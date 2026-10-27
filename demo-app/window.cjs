// A "virtual display" for Cofounder's agentic vision loop.
//
// Opens the Ledgerly demo app in its own window and exposes it on 127.0.0.1 as
// pixels in, input events out — nothing else. The agent never reads the DOM:
// it gets PNG screenshots of this window and can only move and click at pixel
// coordinates, exactly as on a real screen. Capturing the window's own content
// (instead of the whole desktop) also means no other app, notification or
// personal window can ever end up in a recording or a model input.
//
//   electron demo-app/window.cjs [--port 7788] [--w 1180 --h 760] [--hidden]
//
//   GET  /screenshot            -> image/png of the window content (CSS pixels, scale 1)
//   POST /click   {x, y}        -> mouse down/up at (x, y)
//   POST /move    {x, y}        -> mouse move
//   POST /type    {text}        -> keyboard input
//   POST /load    {query}       -> reload the app with a query string (e.g. "reset=1&chaos=modal")
//   GET  /info                  -> {width, height}
const { app, BrowserWindow } = require('electron');
const http = require('http');
const path = require('path');

app.commandLine.appendSwitch('force-device-scale-factor', '1');

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
}
const PORT = Number(arg('port', 7788));
const W = Number(arg('w', 1180));
const H = Number(arg('h', 760));
const hidden = process.argv.includes('--hidden');

let win = null;

function load(query) {
  return win.loadFile(path.join(__dirname, 'index.html'), { search: query || '' });
}

function body(req) {
  return new Promise((resolve) => {
    let b = '';
    req.on('data', (c) => { b += c; });
    req.on('end', () => { try { resolve(JSON.parse(b || '{}')); } catch (_) { resolve({}); } });
  });
}

async function mouse(type, x, y) {
  win.webContents.sendInputEvent({ type, x: Math.round(x), y: Math.round(y), button: 'left', clickCount: 1 });
}

app.whenReady().then(async () => {
  win = new BrowserWindow({
    x: 60, y: 60, useContentSize: true, width: W, height: H, show: !hidden, resizable: false,
    title: 'Ledgerly', backgroundColor: '#f6f7fb',
    webPreferences: { contextIsolation: true, backgroundThrottling: false },
  });
  win.setMenuBarVisibility(false);
  await load(arg('query', 'reset=1'));

  http.createServer(async (req, res) => {
    const send = (code, type, data) => { res.writeHead(code, { 'Content-Type': type }); res.end(data); };
    try {
      const url = new URL(req.url, 'http://x');
      if (req.method === 'GET' && url.pathname === '/screenshot') {
        const img = await win.webContents.capturePage();
        const size = img.getSize();
        const out = size.width !== W ? img.resize({ width: W, height: H }) : img;
        return send(200, 'image/png', out.toPNG());
      }
      // Ground truth for the evaluation harness only; the agent never calls it.
      if (req.method === 'GET' && url.pathname === '/eval-state') {
        const st = await win.webContents.executeJavaScript("localStorage.getItem('ledgerly') || '{\"done\":[]}'");
        return send(200, 'application/json', st);
      }
      if (req.method === 'GET' && url.pathname === '/info') {
        return send(200, 'application/json', JSON.stringify({ width: W, height: H }));
      }
      const b = await body(req);
      if (url.pathname === '/move') { await mouse('mouseMove', b.x, b.y); return send(200, 'application/json', '{"ok":true}'); }
      if (url.pathname === '/click') {
        await mouse('mouseMove', b.x, b.y);
        await mouse('mouseDown', b.x, b.y);
        await new Promise((r) => setTimeout(r, 40));
        await mouse('mouseUp', b.x, b.y);
        return send(200, 'application/json', '{"ok":true}');
      }
      if (url.pathname === '/type') {
        for (const ch of String(b.text || '')) win.webContents.sendInputEvent({ type: 'char', keyCode: ch });
        return send(200, 'application/json', '{"ok":true}');
      }
      if (url.pathname === '/load') { await load(b.query || ''); return send(200, 'application/json', '{"ok":true}'); }
      return send(404, 'application/json', '{"error":"not found"}');
    } catch (err) {
      return send(500, 'application/json', JSON.stringify({ error: String(err && err.message) }));
    }
  }).listen(PORT, '127.0.0.1');
});
app.on('window-all-closed', () => app.quit());
