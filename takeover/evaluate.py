"""Agent evaluation: does the visual evidence lead to the right next action?

Runs the takeover agent against the Ledgerly demo app under controlled conditions
and scores each run from the app's own final state (never from the agent's claims).

Scenarios (demo-app fault injection):
  clean      6 invoices, 1 already approved during the demonstration (5 to do)
  zoom125    the whole UI rendered 25% larger than in the demonstration
  slow       "Match PO" responds after 2.5 s (the first verify says no_change)
  modal      a security dialog interrupts every approval (needs the human)
  disabled   "Approve" never enables on INV-2043 (the agent must stop and ask)

    uv run python evaluate.py --brains policy,llm --scenarios clean,zoom125,slow,modal,disabled
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCENARIOS = {
    "clean": {"query": "reset=1", "expect": "all 6 approved (5 by the agent), finish done"},
    "zoom125": {"query": "reset=1&zoom=1.25", "expect": "all approved despite 1.25x scale"},
    "slow": {"query": "reset=1&chaos=slow", "expect": "waits on no_change instead of re-clicking; all approved"},
    "modal": {"query": "reset=1&chaos=modal&p=1", "expect": "asks the human on every unexpected dialog; all approved"},
    "disabled": {"query": "reset=1&chaos=disabled", "expect": "approves 2, then stops and asks on INV-2043; no wrong approval"},
}

def app_state() -> list[str]:
    """Ground truth from the app itself (evaluation-only endpoint the agent never uses)."""
    import urllib.request

    with urllib.request.urlopen("http://127.0.0.1:7788/eval-state", timeout=10) as r:
        return json.loads(r.read()).get("done", [])


LOCAL = False


def run(brain: str, scenario: str, model: str, out_root: Path) -> dict:
    out = out_root / f"{scenario}-{brain}"
    if out.exists():
        subprocess.run(["rm", "-rf", str(out)])
    # Start state = what the demonstration left: INV-2041 already approved.
    cmd = [sys.executable, "agent.py", "--brain", brain, "--model", model, "--runs", "6",
           "--query", SCENARIOS[scenario]["query"] + "&done=INV-2041", "--out", str(out)] + (["--local"] if LOCAL else [])
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {"error": p.stderr[-800:]}
    trace = [json.loads(l) for l in (out / "trace.jsonl").read_text().splitlines()] if (out / "trace.jsonl").exists() else []
    finish = next((e for e in reversed(trace) if e["event"] == "finish"), {})
    looks = [e for e in trace if e["event"] == "look"]
    clicks = [e for e in trace if e["event"] == "click"]
    done = app_state()
    approved = len(done)
    return {
        "brain": brain, "scenario": scenario, "expect": SCENARIOS[scenario]["expect"],
        "finish": finish.get("status"), "why": finish.get("why"),
        "approved_total_on_screen": approved,
        "approved_ids": done,
        "clicks": len(clicks), "no_change_verdicts": sum(1 for c in clicks if not c.get("changed")),
        "unexpected_screens": sum(1 for c in clicks if c.get("changed") and not c.get("now_on_next_expected_screen")),
        "waits": sum(1 for e in trace if e["event"] == "wait"),
        "human_turns": sum(1 for e in trace if e["event"] == "ask_human"),
        "refused_clicks": sum(1 for e in trace if e["event"] == "click_refused"),
        "perception_ms_median": statistics.median([e["perception"]["ms"] for e in looks if e.get("perception", {}).get("ms")] or [0]),
        "perception_where": sorted({e.get("perception", {}).get("where") for e in looks if e.get("perception")} - {None}),
        "seconds": round(time.time() - t, 1),
        "out": str(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", default="policy")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--model", default="gemma4:e2b")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--local", action="store_true", help="perception in-process instead of AWS")
    a = ap.parse_args()
    global LOCAL
    LOCAL = a.local
    out_root = Path(a.out)
    out_root.mkdir(exist_ok=True)
    rows = []
    for brain in a.brains.split(","):
        for sc in a.scenarios.split(","):
            r = run(brain, sc, a.model, out_root)
            rows.append(r)
            print(json.dumps(r))
    (out_root / "agent_results.json").write_text(json.dumps(rows, indent=2))
    lines = ["| brain | scenario | expected | finish | approved (of 6) | clicks | no_change | unexpected | waits | human | refused | perception p50 (ms) | where | s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['brain']} | {r['scenario']} | {r['expect']} | {r['finish']} | {r['approved_total_on_screen']} | {r['clicks']} | "
                     f"{r['no_change_verdicts']} | {r['unexpected_screens']} | {r['waits']} | {r['human_turns']} | {r['refused_clicks']} | "
                     f"{r['perception_ms_median']} | {', '.join(r['perception_where'])} | {r['seconds']} |")
    (out_root / "AGENT_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
