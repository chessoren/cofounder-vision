"""Cofounder takeover: an agentic perception → decision → action loop on OpenCV 5.

The agent replays a workflow it learned from one screen recording. It only gets
pixels, and it can only act through five tools:

  look(step)            OpenCV 5: am I on the expected screen? where is the control
                        I clicked in the demonstration, and how many look like it?
  click(x, y, step)     acts, then verifies with OpenCV 5 that the screen changed
                        the way it should (local change, navigation, nothing, or
                        something unexpected such as a dialog).
  wait(seconds)         gives a slow app time, then reports whether anything moved.
  ask_human(question)   hands control back to a person.
  finish(status, why)   ends the run.

The visual evidence decides what happens next: a low match score leads to a
re-look and then a question; "no change" leads to a wait and a re-check before any
retry; an unexpected screen leads to the human. Guards live in the tools, not in
the prompt: click() refuses coordinates that OpenCV did not return in the last
look(). So neither the model nor a bug can click somewhere it has not seen.

Two brains drive the same tools:
  --brain llm      a Strands agent (local Ollama model; Bedrock when available)
  --brain policy   a deterministic controller implementing the same rules, used as
                   the evaluation baseline

    uv run python agent.py --workflow workflows/approve-invoice.json --brain policy --runs 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from display import VirtualDisplay
from vision_client import Vision, locate_all, unb64

CONFIRM_BEFORE = {2}  # step indices that are irreversible (Approve): need a human OK once per batch


@dataclass
class Session:
    workflow: dict
    display: VirtualDisplay
    vision: Vision
    out: Path
    human: str = "auto"  # "auto" (scripted answers for evaluation) or "dialog" (a real person, macOS dialog)
    trace: list = field(default_factory=list)
    last_look: dict = field(default_factory=dict)
    last_screen: np.ndarray | None = None
    batch_approved: bool = False
    clicks: int = 0
    human_turns: int = 0
    refused_clicks: int = 0
    step_ok: dict = field(default_factory=dict)
    stopped: bool = False
    t0: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ helpers
    def step(self, i: int) -> dict:
        return self.workflow["steps"][i]

    def log(self, kind: str, **data) -> dict:
        event = {"t": round(time.time() - self.t0, 3), "event": kind, **data}
        self.trace.append(event)
        with (self.out / "trace.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        return event

    def snap(self, name: str, img: np.ndarray, boxes=(), point=None, color=(222, 130, 61)) -> str:
        vis = img.copy()
        for b in boxes:
            x, y, w, h = b
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        if point:
            cv2.circle(vis, point, 9, (61, 130, 222), -1, cv2.LINE_AA)
        path = self.out / "frames" / f"{len(self.trace):04d}-{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), vis)
        return str(path.relative_to(self.out))

    # -------------------------------------------------------------------- tools
    def look(self, step: int) -> dict:
        s = self.step(step)
        screen = self.display.screenshot()
        self.last_screen = screen
        target = unb64(s["target_png_b64"])
        expected = unb64(s["screen_before_png_b64"])
        same = self.vision.same_screen(screen, expected)
        loc = self.vision.locate(screen, target, min_score=0.9)
        cands = locate_all(screen, target, min_score=0.9)
        rep = s.get("repeated_control")
        suggested = None
        if rep and cands:
            pick = cands[min(rep["rank"], len(cands) - 1)] if rep["rank"] < len(cands) else None
            if pick:
                suggested = {"x": pick["x"], "y": pick["y"], "why": f"same position as in the demonstration ({rep['rank'] + 1} of {len(cands)})"}
        elif loc.get("found") and not loc.get("ambiguous"):
            suggested = {"x": loc["x"], "y": loc["y"], "why": f"best match, score {loc['score']:.2f}"}
        allowed = [{"x": c["x"], "y": c["y"], "w": c["w"], "h": c["h"]} for c in cands]
        if loc.get("found"):
            allowed.append({"x": loc["x"], "y": loc["y"], "w": loc.get("w", 40), "h": loc.get("h", 20)})
        self.last_look = {"step": step, "allowed": allowed}
        frame = self.snap(f"look-s{step}", screen,
                          boxes=[(c["x"] - c["w"] // 2, c["y"] - c["h"] // 2, c["w"], c["h"]) for c in cands],
                          point=(suggested["x"], suggested["y"]) if suggested else None)
        # "Expected screen" = the screen looks like the demonstration's, OR the control the
        # person clicked there is visible (content differs between items; layout does not).
        result = {
            "step": step,
            "on_expected_screen": bool(same.get("same")) or bool(cands) or bool(loc.get("found")),
            "screen_similarity": round(float(same.get("similarity", 0)), 3),
            "target_found": bool(loc.get("found")) or bool(cands),
            "best_score": round(float(loc.get("score", 0)), 3),
            "method": loc.get("method"),
            "ambiguous": bool(loc.get("ambiguous")) and not suggested,
            "identical_controls_on_screen": len(cands),
            "suggested_click": suggested,
            "perception": {"where": loc.get("where"), "ms": loc.get("roundtrip_ms")},
        }
        self.log("look", **result, frame=frame)
        return result

    def click(self, x: int, y: int, step: int) -> dict:
        ok = self.last_look.get("step") == step and any(
            abs(x - a["x"]) <= max(6, a["w"] // 2) and abs(y - a["y"]) <= max(6, a["h"] // 2) for a in self.last_look.get("allowed", [])
        )
        if not ok:
            self.refused_clicks += 1
            self.log("click_refused", step=step, x=x, y=y, reason="not a control OpenCV found in the last look()")
            return {"done": False, "refused": True, "reason": "Only click coordinates returned by look() for this step."}
        if step in CONFIRM_BEFORE and not self.batch_approved:
            answer = self.ask_human(
                "I'm about to approve supplier invoices on your behalf, the way you did in the recording. Go ahead for this batch?",
                ["Approve this batch", "Stop"])
            if answer["answer"] != "Approve this batch":
                return {"done": False, "stopped_by_human": True}
            self.batch_approved = True
        before = self.last_screen if self.last_screen is not None else self.display.screenshot()
        self.display.click(x, y)
        self.clicks += 1
        time.sleep(0.9)
        after = self.display.screenshot()
        self.last_screen = after
        roi = [max(0, x - 120), max(0, y - 60), 240, 120]
        v = self.vision.verify(before, after, roi=roi)
        nxt = step + 1 if step + 1 < len(self.workflow["steps"]) else 0
        expected_next = unb64(self.step(nxt)["screen_before_png_b64"])
        same_next = self.vision.same_screen(after, expected_next)
        next_target_visible = bool(locate_all(after, unb64(self.step(nxt)["target_png_b64"]), min_score=0.9))
        frame = self.snap(f"after-s{step}", after, boxes=[tuple(r) for r in v.get("regions", [])[:8]], point=(x, y), color=(99, 157, 31))
        # Compare the effect with the demonstration: a dialog covering the screen changes far
        # more pixels than the navigation the person saw. That is "unexpected", whatever else matches.
        demo = self.step(step).get("demo_effect") or {}
        g = float(v.get("global_change", 0))
        expected_g = float(demo.get("global_change", g))
        effect_as_demonstrated = g <= max(0.25, 3 * expected_g + 0.1)
        result = {
            "step": step,
            "effect_as_demonstrated": effect_as_demonstrated,
            "demo_global_change": round(expected_g, 4),
            "verdict": v.get("verdict"),
            "changed": bool(v.get("changed")),
            "local_change": round(float(v.get("local_change", 0)), 4),
            "global_change": round(float(v.get("global_change", 0)), 4),
            "now_on_next_expected_screen": effect_as_demonstrated and (bool(same_next.get("same")) or next_target_visible),
            "next_screen_similarity": round(float(same_next.get("similarity", 0)), 3),
            "perception": {"where": v.get("where"), "ms": v.get("roundtrip_ms")},
        }
        self.log("click", x=x, y=y, **result, frame=frame)
        self._before_click = before
        self.step_ok[step] = bool(result["changed"] and effect_as_demonstrated
                                  and (result["now_on_next_expected_screen"] or result["verdict"] == "changed"))
        return result

    def wait(self, seconds: float) -> dict:
        seconds = float(min(max(seconds, 0.2), 5.0))
        before = getattr(self, "_before_click", None)
        time.sleep(seconds)
        now = self.display.screenshot()
        self.last_screen = now
        result = {"waited": seconds}
        if before is not None:
            v = self.vision.verify(before, now)
            result.update({"changed_since_last_click": bool(v.get("changed")), "verdict": v.get("verdict"),
                           "local_change": round(float(v.get("local_change", 0)), 4), "global_change": round(float(v.get("global_change", 0)), 4)})
        self.log("wait", **result, frame=self.snap("wait", now))
        return result

    def ask_human(self, question: str, options: list[str] | None = None) -> dict:
        options = options or ["Continue", "Stop"]
        self.human_turns += 1
        if self.human == "dialog":
            script = ('display dialog "' + question.replace('"', "'") + '" with title "Cofounder needs you" buttons {'
                      + ", ".join('"' + o + '"' for o in options[:3]) + '} default button 1')
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            answer = r.stdout.strip().split(":", 1)[-1] if r.returncode == 0 else options[-1]
        else:
            answer = self.scripted_human(question, options)
        self.log("ask_human", question=question, options=options, answer=answer, mode=self.human)
        return {"answer": answer}

    def scripted_human(self, question: str, options: list[str]) -> str:
        """Evaluation stand-in for a person: approves the batch, and when a security
        dialog appears, deals with it themselves (clicks it away), then says continue.
        For a stuck control, a person tells the agent to stop."""
        q = question.lower()
        if "approve" in q and "batch" in q:
            return options[0]
        if "dialog" in q or "unexpected" in q or "did not see" in q:
            self.display.click(447, 429)  # the person answers the dialog on screen
            time.sleep(0.6)
            return options[0]
        return options[-1]

    def finish(self, status: str, why: str) -> dict:
        self.stopped = True
        self.log("finish", status=status, why=why)
        return {"finished": status}


# ---------------------------------------------------------------------- brains
def run_policy(sess: Session, runs: int) -> dict:
    """Deterministic controller: the rules the LLM prompt asks for, written as code."""
    n_steps = len(sess.workflow["steps"])
    completed = 0
    for run in range(runs):
        for step in range(n_steps):
            seen = sess.look(step)
            tries = 0
            while not seen["suggested_click"] and tries < 2:
                if step == 0 and run > 0 and seen["identical_controls_on_screen"] == 0 and seen["on_expected_screen"]:
                    sess.finish("done", "no items left to process")
                    return {"completed_runs": completed}
                sess.wait(1.5)
                seen = sess.look(step)
                tries += 1
            if not seen["suggested_click"]:
                if step == 0 and run > 0:
                    sess.finish("done", "nothing left that matches the demonstration")
                    return {"completed_runs": completed}
                a = sess.ask_human(f"I can't find the control for step {step + 1} (best score {seen['best_score']:.2f}). What should I do?", ["Stop"])
                sess.finish("escalated", f"step {step + 1}: target not found")
                return {"completed_runs": completed}
            res = sess.click(seen["suggested_click"]["x"], seen["suggested_click"]["y"], step)
            if res.get("stopped_by_human"):
                sess.finish("stopped", "human declined")
                return {"completed_runs": completed}
            attempts = 0
            while not (res.get("changed") and res.get("effect_as_demonstrated", True)
                       and (res.get("now_on_next_expected_screen") or res.get("verdict") == "changed")):
                if res.get("changed") and not (res.get("now_on_next_expected_screen") and res.get("effect_as_demonstrated", True)):
                    a = sess.ask_human("Something I did not see in the demonstration appeared (unexpected dialog or screen). Can you check?",
                                       ["Continue", "Stop"])
                    if a["answer"] != "Continue":
                        sess.finish("stopped", "unexpected screen")
                        return {"completed_runs": completed}
                    after = sess.look((step + 1) % n_steps)
                    if after["on_expected_screen"]:
                        break
                    sess.finish("escalated", "screen still unexpected after human help")
                    return {"completed_runs": completed}
                w = sess.wait(2.0)
                if w.get("changed_since_last_click"):
                    break
                attempts += 1
                if attempts > 1:
                    sess.ask_human(f"I clicked step {step + 1} twice and nothing changed on screen. Should I stop?", ["Stop"])
                    sess.finish("escalated", f"step {step + 1}: no visible effect")
                    return {"completed_runs": completed}
                seen = sess.look(step)
                if not seen["suggested_click"]:
                    continue
                res = sess.click(seen["suggested_click"]["x"], seen["suggested_click"]["y"], step)
        completed += 1
        sess.log("run_complete", run=run)
    sess.finish("done", f"{completed} runs completed")
    return {"completed_runs": completed}


SYSTEM = """You are Cofounder, replaying ONE step of a workflow a person showed you once in a screen recording.
You only see the screen through tools. Rules:
1. Call look(step). If suggested_click is not null, call click with exactly those x and y.
2. If suggested_click is null: call wait(1.5), then look(step) again (at most twice).
   - If it is still null and this is step 0 of a later run, reply NOTHING_LEFT.
   - Otherwise call ask_human, then reply STOP.
3. After click():
   - If changed is true, effect_as_demonstrated is true and (now_on_next_expected_screen is true or verdict is "changed"), reply DONE.
   - If changed is false: call wait(2). If changed_since_last_click is true, reply DONE. Otherwise look(step) and click once more; if that also has no effect, call ask_human and reply STOP.
   - If changed is true but effect_as_demonstrated or now_on_next_expected_screen is false, something unexpected appeared: call ask_human. If the answer is "Continue", reply DONE; otherwise reply STOP.
Never invent coordinates. Reply with a single word: DONE, NOTHING_LEFT or STOP."""

STEP_BUDGET = 10


def run_llm(sess: Session, runs: int, model: str) -> dict:
    """A Strands agent decides each step from the visual evidence its tools return.

    The outer loop is plain code (which step, which run); inside a step, the model chooses
    every tool call. Short, per-step conversations keep a small local model on track, and a
    per-step tool budget bounds the autonomy.
    """
    from strands import Agent, tool
    from strands.models.ollama import OllamaModel
    from strands.types.exceptions import MaxTokensReachedException

    calls = {"n": 0}

    def budget():
        calls["n"] += 1
        if calls["n"] > STEP_BUDGET:
            raise RuntimeError("tool budget for this step exhausted")

    @tool
    def look(step: int) -> dict:
        """Look at the screen for a workflow step: expected screen?, target found?, suggested click.

        Args:
            step: workflow step index.
        """
        budget()
        return sess.look(step)

    @tool
    def click(x: int, y: int, step: int) -> dict:
        """Click at coordinates returned by look(), then verify the effect with OpenCV.

        Args:
            x: x from look().suggested_click.
            y: y from look().suggested_click.
            step: workflow step index.
        """
        budget()
        return sess.click(x, y, step)

    @tool
    def wait(seconds: float) -> dict:
        """Wait, then report whether the screen changed since the last click.

        Args:
            seconds: 0.2 to 5.
        """
        budget()
        return sess.wait(seconds)

    @tool
    def ask_human(question: str) -> dict:
        """Hand control to the person and get their answer.

        Args:
            question: a short, specific question.
        """
        budget()
        return sess.ask_human(question)

    n = len(sess.workflow["steps"])
    completed = 0
    llm = OllamaModel(host="http://127.0.0.1:11434", model_id=model, temperature=0.1, max_tokens=8192)
    for run in range(runs):
        for step in range(n):
            calls["n"] = 0
            sess.step_ok.pop(step, None)
            agent = Agent(model=llm, system_prompt=SYSTEM, tools=[look, click, wait, ask_human], callback_handler=None)
            prompt = (f"Run {run + 1} of up to {runs}. Step {step} of {n - 1}"
                      + (" (this is step 0 of a later run: the batch may be finished)" if run > 0 and step == 0 else "") + ". Go.")
            try:
                reply = str(agent(prompt)).strip().upper()
            except MaxTokensReachedException:
                reply = str(agent("Reply with one word: DONE, NOTHING_LEFT or STOP.")).strip().upper()
            except RuntimeError as err:
                reply = "BUDGET"
                sess.log("budget_exhausted", step=step, error=str(err))
            sess.log("llm_step", run=run, step=step, reply=reply[:40], step_ok=sess.step_ok.get(step, False), tool_calls=calls["n"])
            if "NOTHING" in reply and step == 0 and run > 0:
                sess.finish("done", f"{completed} runs completed; nothing left to process")
                return {"completed_runs": completed}
            if not sess.step_ok.get(step) and "DONE" not in reply:
                if not any(e["event"] == "ask_human" for e in sess.trace[-6:]):
                    sess.ask_human(f"I could not complete step {step + 1}. Should I stop?", ["Stop"])
                sess.finish("escalated", f"step {step + 1}: model reply {reply[:20]}")
                return {"completed_runs": completed}
        completed += 1
        sess.log("run_complete", run=run)
    sess.finish("done", f"{completed} runs completed")
    return {"completed_runs": completed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default="workflows/approve-invoice.json")
    ap.add_argument("--brain", choices=["policy", "llm"], default="policy")
    ap.add_argument("--model", default="gemma4:e2b")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--query", default="", help="demo app state/fault injection, e.g. 'chaos=modal'")
    ap.add_argument("--human", choices=["auto", "dialog"], default="auto")
    ap.add_argument("--out", default="")
    ap.add_argument("--local", action="store_true", help="run OpenCV in-process instead of on AWS")
    a = ap.parse_args()

    out = Path(a.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}-{a.brain}")
    out.mkdir(parents=True, exist_ok=True)
    display = VirtualDisplay()
    workflow = json.loads(Path(a.workflow).read_text())
    sess = Session(workflow, display, Vision(prefer_cloud=not a.local), out, human=a.human)
    display.load(a.query)
    time.sleep(1.0)
    sess.log("start", brain=a.brain, model=a.model if a.brain == "llm" else None, query=a.query, runs=a.runs,
             perception="aws-lambda-arm64" if sess.vision.url else "local")
    result = run_policy(sess, a.runs) if a.brain == "policy" else run_llm(sess, a.runs, a.model)
    summary = {**result, "clicks": sess.clicks, "human_turns": sess.human_turns, "refused_clicks": sess.refused_clicks,
               "seconds": round(time.time() - sess.t0, 1), "out": str(out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
