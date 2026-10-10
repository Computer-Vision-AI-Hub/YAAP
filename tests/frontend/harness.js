/* Frontend test harness — loaded by every *.test.html, which run through
   tests/frontend/run.py (headless Chrome + a local server mapping /static/ to
   frontend/, exactly like the app). boot() builds the page from the real
   frontend/index.html and loads the real api.js/editor.js/app.js, with
   window.fetch replaced by a mock API. A test is:

     test(async t => {
       await t.boot({ routes: [["GET", /\/api\/projects$/, () => []]] });
       t.check("something", condition);
     });

   Results land in document.title as "RESULT {json}" for run.py to read. */
(() => {
  const realFetch = window.fetch.bind(window);
  const PNG_1PX = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

  const results = { passed: [], failed: [], unmatched: [] };
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = () => rej(new Error(`failed to load ${src}`));
      document.body.append(s);
    });
  }

  // routes: [[method, urlRegex, handler(url, body, opts) -> json]], first match wins
  function installMockApi(routes) {
    const calls = [];
    const defaults = [
      ["GET", /\/api\/system\/device$/, () => ({ torch: null, cuda: false, devices: [], message: "" })],
      ["POST", /\/api\/system\/(here|leaving)$/, () => ({ ok: true })],
    ];
    window.fetch = async (url, opts = {}) => {
      const method = (opts.method || "GET").toUpperCase();
      let body = opts.body;
      try { body = typeof body === "string" ? JSON.parse(body) : body; } catch (_) {}
      calls.push({ method, url, body });
      for (const [m, re, handler] of [...routes, ...defaults]) {
        if (m === method && re.test(url)) {
          const out = await handler(url, body, opts);
          return { ok: true, status: 200, json: async () => out };
        }
      }
      results.unmatched.push(`${method} ${url}`);
      return { ok: true, status: 200, json: async () => ({}) };
    };
    navigator.sendBeacon = () => true;
    return calls;
  }

  async function boot({ routes = [], hash = "" } = {}) {
    // the real shell, so tests break if index.html's ids change
    const html = await (await realFetch("/frontend/index.html")).text();
    const doc = new DOMParser().parseFromString(html, "text/html");
    doc.querySelectorAll('head link[rel="stylesheet"][href^="/static/"]').forEach(l =>
      document.head.append(document.importNode(l, true)));
    doc.querySelectorAll("script").forEach(s => s.remove());
    document.body.innerHTML = doc.body.innerHTML;

    const calls = installMockApi(routes);
    if (hash) history.replaceState(null, "", hash);
    for (const src of ["/static/js/api.js", "/static/js/editor.js", "/static/js/app.js"]) await loadScript(src);
    API.imageUrl = () => PNG_1PX;
    API.thumbUrl = () => PNG_1PX;
    if (API.jobPlotUrl) API.jobPlotUrl = () => PNG_1PX;
    document.dispatchEvent(new Event("DOMContentLoaded", { bubbles: true }));
    await sleep(100);
    return calls;
  }

  // navigate through the app's real hashchange routing and let the view render
  async function go(hash, ms = 150) {
    location.hash = hash;
    await sleep(ms);
  }

  function check(name, cond, detail) {
    if (cond) results.passed.push(name);
    else results.failed.push(detail === undefined ? name : `${name} — ${JSON.stringify(detail)}`);
  }

  window.test = fn => {
    window.addEventListener("load", async () => {
      try { await fn({ boot, go, sleep, check, PNG_1PX }); }
      catch (e) { results.failed.push(`threw: ${e && e.message} @ ${e && e.stack}`); }
      document.title = "RESULT " + JSON.stringify(results);
    });
  };
})();
