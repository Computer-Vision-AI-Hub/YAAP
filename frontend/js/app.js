/* YAAP application — hash-routed views over the API. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const S = { project: null, images: [], curImage: -1, editor: null, saveTimer: null,
            activeClassId: null, jobPoll: null, verPoll: null,
            // one-shot hints consumed by the next viewAnnotate() call, set by whoever
            // navigates there (e.g. the Review grid) to control which image opens first,
            // how the working list is ordered, and whether to flag already-annotated images
            pendingImageId: null, annotateSort: null, warnIfAnnotated: false };

/* ── plumbing ─────────────────────────────────────────────────────── */
function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`; el.textContent = msg;
  $("#toastHost").append(el);
  setTimeout(() => el.remove(), 4200);
}
const err = e => toast(e.message || String(e), "err");

function nav(view) { location.hash = S.project ? `#/p/${S.project.id}/${view}` : "#/"; }

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", () => {
  $("#brand").onclick = () => { location.hash = "#/"; };
  $$(".rail-btn[data-view]").forEach(b => b.onclick = () => nav(b.dataset.view));
  $("#homeBtn").onclick = () => { location.hash = "#/"; };
  loadDevice();
  route();
});

async function loadDevice() {
  try {
    const d = await API.device();
    const chip = $("#deviceChip");
    if (!d.torch) { chip.textContent = "no torch"; chip.className = "chip mono warn"; chip.title = d.message; }
    else if (d.cuda) { chip.textContent = d.devices[0].split("—")[1]?.trim() || "CUDA"; chip.className = "chip mono ok"; chip.title = d.devices.join("\n"); }
    else { chip.textContent = `CPU · torch ${d.torch}`; chip.className = "chip mono"; }
  } catch (_) { $("#deviceChip").textContent = "offline"; }
}

async function route() {
  clearInterval(S.jobPoll); clearInterval(S.verPoll);
  const m = location.hash.match(/^#\/p\/(\d+)\/(\w+)/);
  if (!m) { S.project = null; $("#rail").hidden = true; $("#topbarMid").innerHTML = ""; return viewHome(); }
  const [, pid, view] = m;
  try {
    if (!S.project || S.project.id !== +pid) S.project = await API.project(+pid);
    else S.project = await API.project(+pid);
  } catch (e) { err(e); location.hash = "#/"; return; }
  $("#rail").hidden = false;
  $$(".rail-btn").forEach(b => b.classList.toggle("active", b.dataset.view === view));
  $("#topbarMid").innerHTML = `
    <span class="proj-crumb">${esc(S.project.name)}</span>
    <span class="task-badge ${S.project.task_type}">${S.project.task_type}</span>
    <span class="chip mono">${S.project.image_count} imgs · ${S.project.annotated_count} labeled</span>`;
  ({ upload: viewUpload, annotate: viewAnnotate, review: viewReview, quickreview: viewQuickReview,
     classes: viewClasses, autolabel: viewAutolabel,
     generate: viewGenerate, versions: viewVersions, train: viewTrain }[view] || viewReview)();
}

/* ── home: projects ───────────────────────────────────────────────── */
async function viewHome() {
  const stage = $("#stage");
  let projects = [];
  try { projects = await API.projects(); } catch (e) { err(e); }
  stage.innerHTML = `<div class="stage-pad">
    <h1 class="page">Projects</h1>
    <p class="sub">Everything lives on this machine — images, labels, models and training runs
      are stored under <span class="mono">data/</span>. Nothing leaves your network.</p>
    <div class="proj-grid">
      ${projects.map(p => `
        <div class="proj-card" data-id="${p.id}">
          <button class="del" title="Delete project">✕</button>
          <h2>${esc(p.name)}</h2>
          <div class="desc">${esc(p.description)}</div>
          <div class="proj-stats">
            <span><b>${p.image_count}</b> images</span>
            <span><b>${p.annotated_count}</b> labeled</span>
            <span><b>${p.class_count}</b> classes</span>
            <span><b>${p.version_count}</b> versions</span>
          </div>
          <div style="margin-top:12px"><span class="task-badge ${p.task_type}">${p.task_type}</span></div>
        </div>`).join("")}
      <div class="proj-card new-card" id="newProj">＋ New project</div>
    </div></div>`;

  $$(".proj-card[data-id]").forEach(c => {
    c.onclick = e => { if (!e.target.matches(".del")) location.hash = `#/p/${c.dataset.id}/review`; };
    $(".del", c).onclick = async () => {
      if (!confirm("Delete this project and all of its data on disk?")) return;
      try { await API.deleteProject(+c.dataset.id); viewHome(); } catch (e) { err(e); }
    };
  });
  $("#newProj").onclick = showNewProject;
}

function showNewProject() {
  openModal(`
    <h1 class="page" style="font-size:19px">New project</h1>
    <label class="fld">Name</label><input class="input" id="npName" placeholder="e.g. Ultrasound nerves">
    <label class="fld">Description</label><input class="input" id="npDesc" placeholder="optional">
    <label class="fld">Task type — this decides annotation tools and export format</label>
    <select class="input" id="npTask">
      <option value="detect">Object detection · bounding boxes → YOLO txt</option>
      <option value="segment">Instance segmentation · polygons → YOLO-seg txt</option>
      <option value="classify">Classification · one label per image → folders</option>
    </select>
    <div class="row" style="margin-top:20px;justify-content:flex-end">
      <button class="btn ghost" onclick="closeModal()">Cancel</button>
      <button class="btn primary" id="npGo">Create project</button>
    </div>`);
  $("#npGo").onclick = async () => {
    try {
      const p = await API.createProject({ name: $("#npName").value || "Untitled",
        description: $("#npDesc").value, task_type: $("#npTask").value });
      closeModal(); location.hash = `#/p/${p.id}/classes`;
      toast("Project created — add your classes, then upload images.", "ok");
    } catch (e) { err(e); }
  };
}

function openModal(html) { $("#modalBody").innerHTML = html; $("#modal").showModal(); }
function closeModal() { $("#modal").close(); }

/* ── upload / dataset browser ─────────────────────────────────────── */
async function viewUpload() {
  const stage = $("#stage");
  const p = S.project;
  stage.innerHTML = `<div class="stage-pad">
    <h1 class="page">Upload</h1>
    <p class="sub">Drop images below. They are copied into this project's local folder and
      thumbnailed for the browser — originals are never modified.</p>
    <div class="dropzone" id="dz">Drop images here or click to browse
      <div class="mono" style="font-size:11px;margin-top:8px;color:var(--faint)">jpg · png · bmp · webp · tiff</div>
    </div>
    <input type="file" id="fileIn" multiple accept="image/*" hidden>

    <h3 class="sect">Dataset in / out</h3>
    <div class="card row" style="align-items:flex-start;gap:22px;flex-wrap:wrap">
      <div>
        <div style="font-weight:600;margin-bottom:6px">Download</div>
        <p class="sub" style="margin-bottom:8px;max-width:260px">Every current image + annotation,
          as-is — no split, no version, no augmentation.</p>
        <label class="fld" style="margin:0 0 8px"><input type="checkbox" id="rawApproved"> Approved images only</label>
        <button class="btn" id="rawGo">⬇ Download dataset (zip)</button>
      </div>
      <div>
        <div style="font-weight:600;margin-bottom:6px">Import</div>
        <p class="sub" style="margin-bottom:8px;max-width:260px">Load a dataset zip — YOLO
          detect/segment layout, a classify folder-per-class layout, or plain images.</p>
        <input type="file" id="impFile" accept=".zip" class="input" style="margin-bottom:8px">
        <button class="btn" id="impGo">Import zip</button>
      </div>
      <div>
        <div style="font-weight:600;margin-bottom:6px">Merge</div>
        <p class="sub" style="margin-bottom:8px;max-width:260px">Copy another project's images,
          classes and annotations into this one.</p>
        <select class="input" id="mrgProject" style="margin-bottom:8px"></select>
        <button class="btn" id="mrgGo">Merge in</button>
      </div>
    </div>
  </div>`;

  const dz = $("#dz"), fi = $("#fileIn");
  dz.onclick = () => fi.click();
  dz.ondragover = e => { e.preventDefault(); dz.classList.add("drag"); };
  dz.ondragleave = () => dz.classList.remove("drag");
  dz.ondrop = e => { e.preventDefault(); dz.classList.remove("drag"); doUpload([...e.dataTransfer.files]); };
  fi.onchange = () => doUpload([...fi.files]);

  async function doUpload(files) {
    if (!files.length) return;
    dz.textContent = `Uploading 0 / ${files.length}…`;
    let n = 0;
    try {
      const res = await API.upload(p.id, files, () => { dz.textContent = `Uploading ${++n} / ${files.length}…`; });
      toast(`Uploaded ${res.created.length} image(s)` +
            (res.skipped.length ? `, skipped ${res.skipped.length}` : ""), "ok");
    } catch (e) { err(e); }
    route();
  }

  $("#rawGo").onclick = () => {
    const a = document.createElement("a");
    a.href = API.exportRawUrl(p.id, $("#rawApproved").checked);
    a.click();
  };

  $("#impGo").onclick = async () => {
    const f = $("#impFile").files[0];
    if (!f) return toast("Choose a .zip file first.", "err");
    const btn = $("#impGo"); btn.disabled = true;
    try {
      const res = await API.importDataset(p.id, f);
      toast(`Imported ${res.images_created} image(s)` +
            (res.images_skipped ? `, skipped ${res.images_skipped}` : ""), "ok");
      route();
    } catch (e) { err(e); }
    btn.disabled = false;
  };

  try {
    const others = (await API.projects()).filter(x => x.id !== p.id && x.task_type === p.task_type);
    $("#mrgProject").innerHTML = others.length
      ? others.map(x => `<option value="${x.id}">${esc(x.name)} (${x.image_count} imgs)</option>`).join("")
      : `<option value="">No other ${p.task_type} projects</option>`;
    $("#mrgGo").disabled = !others.length;
  } catch (e) { err(e); }
  $("#mrgGo").onclick = async () => {
    const sid = +$("#mrgProject").value;
    if (!sid) return;
    if (!confirm("Copy all images, classes and annotations from that project into this one?")) return;
    const btn = $("#mrgGo"); btn.disabled = true;
    try {
      const res = await API.mergeProject(p.id, { source_project_id: sid });
      toast(`Merged ${res.images_merged} image(s)` +
            (res.images_skipped ? `, skipped ${res.images_skipped}` : ""), "ok");
      route();
    } catch (e) { err(e); }
    btn.disabled = false;
  };
}

/* ── review / quality control ────────────────────────────────────── */
// "reviewable" is a pseudo-filter (not a real status): images that still need
// a QC decision — freshly annotated, or rejected and waiting on a re-review
// after being fixed. Unannotated (nothing to review yet) and approved
// (already resolved) are both excluded, so this queue actually drains to
// empty as you work through it instead of staying populated forever.
const QC_FILTERS = [
  ["reviewable", "To review"], ["all", "All"], ["unannotated", "Unannotated"],
  ["annotated", "Annotated"], ["review", "Needs review"], ["approved", "Approved"],
];
const qcMatch = (im, filter) => filter === "all" ? true
  : filter === "reviewable" ? (im.status === "annotated" || im.status === "review")
  : im.status === filter;

async function viewReview() {
  const p = S.project;
  let images;
  try { images = await API.images(p.id); } catch (e) { return err(e); }
  if (!images.length) { nav("upload"); return; }   // nothing to review yet — go straight to Upload
  S.qcFilter = S.qcFilter || "reviewable";

  function render() {
    const counts = {};
    for (const [k] of QC_FILTERS) counts[k] = images.filter(im => qcMatch(im, k)).length;
    const list = images.filter(im => qcMatch(im, S.qcFilter));

    $("#stage").innerHTML = `<div class="stage-pad">
      <h1 class="page">Quality control</h1>
      <p class="sub">Every image in one place — filter by review state, jump into an image to fix a
        wrong label or box, approve it once it checks out, or reject it to flag it for re-work.</p>
      <div class="row" style="margin-bottom:16px" id="qcFilters">
        ${QC_FILTERS.map(([k, label]) => `
          <button class="btn small ${k === S.qcFilter ? "primary" : "ghost"}" data-f="${k}">${label} <span class="mono">${counts[k] ?? 0}</span></button>`).join("")}
      </div>
      <div class="img-grid" id="qcGrid"></div>
    </div>`;

    $$("#qcFilters [data-f]").forEach(b => b.onclick = () => {
      S.qcFilter = b.dataset.f;
      if (b.dataset.f === "reviewable") { startQuickReview(); return; }
      render();
    });

    const host = $("#qcGrid");
    if (!list.length) { host.innerHTML = `<div class="empty" style="grid-column:1/-1">No images in this state.</div>`; return; }
    host.innerHTML = list.map(im => `
      <div class="img-tile qc-tile" data-id="${im.id}" title="${esc(im.filename)}">
        <img loading="lazy" src="${API.thumbUrl(p.id, im.id)}" alt="">
        <span class="st ${im.status}"></span>
        <div class="qc-actions">
          <button class="btn small" data-open="${im.id}" title="Open in editor">✎</button>
          <button class="btn small ${im.status === "review" ? "danger" : ""}" data-reject="${im.id}"
            title="${im.status === "review" ? "Rejected — clear flag" : "Reject — flag for re-work"}">✕</button>
          <button class="btn small ${im.status === "approved" ? "primary" : ""}" data-approve="${im.id}"
            title="${im.status === "approved" ? "Un-approve" : "Approve"}">✓</button>
        </div>
        <span class="nm">${esc(im.filename)} · ${im.annotation_count} ann.</span>
      </div>`).join("");

    $$(".qc-tile", host).forEach(t => t.onclick = e => {
      if (e.target.closest("button")) return;
      if (S.qcFilter === "reviewable") startQuickReview(+t.dataset.id);
      else openReviewImage(+t.dataset.id);
    });
    $$("[data-open]", host).forEach(b => b.onclick = () => openReviewImage(+b.dataset.open));
    $$("[data-approve]", host).forEach(b => b.onclick = async () => {
      const im = images.find(x => x.id === +b.dataset.approve);
      const next = im.status === "approved" ? "annotated" : "approved";
      try { im.status = (await API.patchImage(p.id, im.id, { status: next })).status; render(); }
      catch (e) { err(e); }
    });
    $$("[data-reject]", host).forEach(b => b.onclick = async () => {
      const im = images.find(x => x.id === +b.dataset.reject);
      const next = im.status === "review" ? "annotated" : "review";
      try { im.status = (await API.patchImage(p.id, im.id, { status: next })).status; render(); }
      catch (e) { err(e); }
    });
  }

  function openReviewImage(imgId) {
    // viewAnnotate() re-fetches images itself, so hand it a hint instead of
    // pre-setting S.images/S.curImage directly (that assignment would just get
    // overwritten by the fresh fetch before it's ever read).
    S.pendingImageId = imgId;
    S.annotateSort = S.qcFilter === "unannotated" ? "unannotatedFirst" : null;
    S.warnIfAnnotated = S.qcFilter === "unannotated";
    nav("annotate");
  }

  render();
}

function startQuickReview(imgId) {
  S.pendingImageId = imgId ?? null;
  nav("quickreview");
}

/* ── quick review: accept/reject only, no editing, auto-advances the queue ── */
async function viewQuickReview() {
  const p = S.project;
  let images;
  try { images = await API.images(p.id); } catch (e) { return err(e); }

  let queue = images.filter(im => qcMatch(im, "reviewable"));
  if (S.pendingImageId != null) {
    const i = queue.findIndex(x => x.id === S.pendingImageId);
    if (i > 0) queue = queue.slice(i);
    S.pendingImageId = null;
  }
  if (!queue.length) {
    toast("Nothing left to review.", "ok");
    nav("review");
    return;
  }

  let idx = 0;

  function onKey(e) {
    if (e.target.matches("input,textarea,select")) return;
    const k = e.key.toLowerCase();
    if (k === "a") $("#qrAccept")?.click();
    else if (k === "r") $("#qrReject")?.click();
  }
  window.addEventListener("keydown", onKey);
  window.addEventListener("hashchange", function cleanup() {
    window.removeEventListener("keydown", onKey);
    window.removeEventListener("hashchange", cleanup);
  });

  async function renderCurrent() {
    const im = queue[idx];
    let anns = [];
    try { anns = await API.annotations(p.id, im.id); } catch (e) { /* show image without overlays */ }

    const strokeW = Math.max(2, Math.round(im.width / 350));
    const fontSize = Math.min(42, Math.max(14, Math.round(im.width / 40)));
    const shapes = anns.map(a => {
      const c = p.classes.find(x => x.id === a.class_id);
      const color = c ? c.color : "#999";
      const name = (c ? c.name : "?") + (a.source === "model" ? ` ·${Math.round(a.confidence * 100)}%` : "");
      let shape, lx, ly;
      if (a.kind === "bbox") {
        const d = a.data;
        shape = `<rect x="${d.x}" y="${d.y}" width="${d.w}" height="${d.h}" fill="${color}33" stroke="${color}" stroke-width="${strokeW}"/>`;
        lx = d.x; ly = d.y;
      } else if (a.kind === "polygon") {
        const pts = a.data.points.map(pt => pt.join(",")).join(" ");
        shape = `<polygon points="${pts}" fill="${color}33" stroke="${color}" stroke-width="${strokeW}"/>`;
        [lx, ly] = a.data.points[0];
      } else return "";
      // label tag — clamped so it never sits above the image's own top edge
      const labelY = Math.max(fontSize + 4, ly);
      const labelW = Math.max(name.length * fontSize * 0.62, fontSize * 1.6);
      const label = `<rect x="${lx}" y="${labelY - fontSize - 4}" width="${labelW}" height="${fontSize + 6}" fill="${color}"/>
        <text x="${lx + 4}" y="${labelY - 5}" font-size="${fontSize}" font-family="monospace"
          font-weight="600" fill="#0a0a0a">${esc(name)}</text>`;
      return shape + label;
    }).join("");

    $("#stage").innerHTML = `<div class="stage-pad" style="max-width:900px">
      <h1 class="page">Review queue</h1>
      <p class="sub">${idx + 1} of ${queue.length} · ${esc(im.filename)} · ${anns.length} annotation(s)</p>
      <div class="qr-frame">
        <svg viewBox="0 0 ${im.width} ${im.height}" preserveAspectRatio="xMidYMid meet">
          <image href="${API.imageUrl(p.id, im.id)}" width="${im.width}" height="${im.height}"></image>
          ${shapes}
        </svg>
      </div>
      <div class="row" style="margin-top:18px;justify-content:center;gap:16px">
        <button class="btn danger qr-btn" id="qrReject">✕ Reject <span class="kbd">R</span></button>
        <button class="btn primary qr-btn" id="qrAccept">✓ Accept <span class="kbd">A</span></button>
      </div>
    </div>`;

    $("#qrAccept").onclick = () => decide("approved");
    $("#qrReject").onclick = () => decide("review");
  }

  async function decide(status) {
    const im = queue[idx];
    try { await API.patchImage(p.id, im.id, { status }); } catch (e) { return err(e); }
    idx++;
    if (idx >= queue.length) {
      toast("Review queue complete!", "ok");
      nav("review");
      return;
    }
    renderCurrent();
  }

  await renderCurrent();
}

/* ── classes ──────────────────────────────────────────────────────── */
async function viewClasses() {
  const p = S.project;
  $("#stage").innerHTML = `<div class="stage-pad" style="max-width:640px">
    <h1 class="page">Classes</h1>
    <p class="sub">The class order here becomes the class index in exported YOLO labels —
      keep it stable once you start training.</p>
    <div class="row" style="margin-bottom:16px">
      <input class="input" id="clsName" placeholder="New class name" style="flex:1">
      <button class="btn primary" id="clsAdd">Add class</button>
    </div>
    <div id="clsList"></div></div>`;

  function render() {
    $("#clsList").innerHTML = S.project.classes.map((c, i) => `
      <div class="class-row" data-id="${c.id}">
        <input type="color" value="${c.color}" title="Class color">
        <span class="swatch" style="background:${c.color}"></span>
        <span class="nm" contenteditable spellcheck="false">${esc(c.name)}</span>
        <span class="idx">idx ${i} · key ${i < 9 ? i + 1 : "—"}</span>
        <button class="btn ghost small">✕</button>
      </div>`).join("") || `<div class="empty">No classes yet — add one above.</div>`;
    $$(".class-row").forEach(row => {
      const cid = +row.dataset.id;
      $("input[type=color]", row).onchange = async e => {
        try { await API.updateClass(p.id, cid, { color: e.target.value }); refresh(); } catch (e2) { err(e2); }
      };
      $(".nm", row).onblur = async e => {
        try { await API.updateClass(p.id, cid, { name: e.target.textContent.trim() }); refresh(); } catch (e2) { err(e2); }
      };
      $("button", row).onclick = async () => {
        if (!confirm("Delete class and all of its annotations?")) return;
        try { await API.deleteClass(p.id, cid); refresh(); } catch (e2) { err(e2); }
      };
    });
  }
  async function refresh() { S.project = await API.project(p.id); render(); }
  $("#clsAdd").onclick = async () => {
    const name = $("#clsName").value.trim();
    if (!name) return;
    try { await API.addClass(p.id, { name }); $("#clsName").value = ""; refresh(); } catch (e) { err(e); }
  };
  $("#clsName").onkeydown = e => { if (e.key === "Enter") $("#clsAdd").click(); };
  render();
}

/* ── annotate ─────────────────────────────────────────────────────── */
async function viewAnnotate() {
  const p = S.project;
  try { S.images = await API.images(p.id); } catch (e) { return err(e); }
  if (!S.images.length) {
    $("#stage").innerHTML = `<div class="stage-pad"><div class="empty">Upload images first.</div></div>`; return;
  }
  if (!p.classes.length && p.task_type !== "detect") { /* classes needed for all tasks actually */ }
  if (!p.classes.length) {
    $("#stage").innerHTML = `<div class="stage-pad"><div class="empty">
      Add at least one class before annotating. <br><br>
      <button class="btn primary" onclick="nav('classes')">Go to classes</button></div></div>`;
    return;
  }
  if (S.annotateSort === "unannotatedFirst") {
    // stable sort: unannotated images first, original relative order preserved
    // within each group — so browsing naturally flows unannotated → annotated
    S.images = [...S.images].sort((a, b) =>
      (a.status === "unannotated" ? 0 : 1) - (b.status === "unannotated" ? 0 : 1));
  }
  S.annotateSort = null;   // one-shot — only applies to this entry, not future navigation

  if (S.pendingImageId != null) {
    const i = S.images.findIndex(x => x.id === S.pendingImageId);
    S.curImage = i >= 0 ? i : 0;
    S.pendingImageId = null;
  } else if (S.curImage < 0 || S.curImage >= S.images.length) {
    S.curImage = 0;
  }
  const warnIfAnnotated = S.warnIfAnnotated;
  S.warnIfAnnotated = false;   // one-shot — consumed here, reused by loadImage() below for this session

  S.activeClassId = S.activeClassId ?? p.classes[0].id;
  if (!p.classes.some(c => c.id === S.activeClassId)) S.activeClassId = p.classes[0].id;

  const isCls = p.task_type === "classify";
  $("#stage").innerHTML = `
  <div class="editor">
    <div class="ed-canvas-wrap">
      <canvas id="edCanvas"></canvas>
      ${isCls ? "" : `
      <div class="ed-toolbar">
        <button class="tool-btn" data-tool="select" title="Select / edit (V)">➤</button>
        <button class="tool-btn active" data-tool="bbox" title="Bounding box (B)" ${p.task_type === "segment" ? "" : ""}>▭</button>
        <button class="tool-btn" data-tool="polygon" title="Polygon (P)" ${p.task_type === "detect" ? "style='display:none'" : ""}>⬠</button>
        <button class="tool-btn sam" data-tool="sam" title="SAM assist — click / shift-click / drag box (S)">✦</button>
        <button class="tool-btn" data-tool="pan" title="Pan (H, or hold Space)">✥</button>
      </div>
      <div class="sam-strip" id="samStrip" hidden>
        <select class="input" id="samModel" title="Model variant"></select>
        <label class="sam-simplify" title="Polygon simplification — higher = fewer vertices, coarser edges">
          simplify
          <input type="range" id="samSimplify" min="0.001" max="0.03" step="0.001" value="0.008">
          <span class="mono" id="samSimplifyVal">0.008</span>
        </label>
        <span class="sam-hint mono" id="samHint">click ＋ · shift-click − · drag box</span>
        <button class="btn small primary" id="samAccept" hidden>✓ Accept <span class="kbd">↵</span></button>
        <button class="btn small" id="samDiscard" hidden>✕ <span class="kbd">Esc</span></button>
      </div>
      <div class="sam-label-pick" id="samLabelPick" hidden></div>`}
      <div class="ed-topstrip">
        <span class="fn mono" id="edFn"></span>
        <span class="savestate" id="saveState">saved</span>
        <span class="chip warn" id="edAnnotatedWarn" hidden>⚠ already annotated</span>
        <button class="btn small" id="edReject" title="Flag this image as needing re-work">✕ Reject</button>
        <button class="btn small" id="edApprove" title="Mark this image as QC-approved">✓ Approve</button>
      </div>
      <div class="ed-hud" id="edHud">—</div>
    </div>
    <aside class="ed-side">
      ${isCls ? `<h4>Label this image</h4><div class="cls-panel" id="clsPanel"></div>`
              : `<h4>Classes <span class="kbd">1–9</span></h4><div class="ed-classes" id="edClasses"></div>
                 <h4>Annotations</h4><div class="ed-anns" id="edAnns"></div>`}
      <div class="ed-nav">
        <button class="btn" id="prevImg" title="Previous (←)">←</button>
        <span class="chip mono" id="imgPos" style="align-self:center"></span>
        <button class="btn" id="nextImg" title="Next (→)">→</button>
      </div>
    </aside>
  </div>`;

  const editor = S.editor = new Editor($("#edCanvas"), {
    classes: () => S.project.classes,
    getActiveClass: () => S.project.classes.find(c => c.id === S.activeClassId),
    onChange: () => { markDirty(); renderAnnList(); },
    onSelect: () => renderAnnList(),
    onToolKey: setTool,
    onClassKey: i => { const c = p.classes[i]; if (c) { S.activeClassId = c.id; renderSide(); } },
    onSamPrompt: runSamPrompt,
    onSamAccept: acceptSam,
    onSamState: updateSamStrip,
  });
  if (p.task_type === "segment") editor.setTool("polygon");
  if (isCls) editor.setTool("pan");

  $$(".tool-btn").forEach(b => b.onclick = () => setTool(b.dataset.tool));
  function setTool(t) {
    editor.setTool(t);
    $$(".tool-btn").forEach(b => b.classList.toggle("active", b.dataset.tool === t));
    const strip = $("#samStrip");
    if (strip) { strip.hidden = t !== "sam"; if (t === "sam") initSam(); }
  }

  /* ── SAM assist ─────────────────────────────────────────────── */
  let samReady = false, samBusy = false, samInfo = null;
  function fillVariants() {
    const models = samInfo?.models?.length ? samInfo.models : [{ name: "sam3.pt", label: "sam3.pt" }];
    $("#samModel").innerHTML = models.map(m => `<option value="${m.name}">${esc(m.label)}</option>`).join("");
    const stored = localStorage.getItem("yaap.samModel");
    if (stored && models.some(m => m.name === stored)) $("#samModel").value = stored;
  }
  async function initSam() {
    if (samReady) return;
    try {
      samInfo = await API.samModels();
      fillVariants();
      $("#samModel").onchange = () => localStorage.setItem("yaap.samModel", $("#samModel").value);

      const storedSimplify = localStorage.getItem("yaap.samSimplify");
      if (storedSimplify) { $("#samSimplify").value = storedSimplify; $("#samSimplifyVal").textContent = storedSimplify; }
      $("#samSimplify").oninput = () => { $("#samSimplifyVal").textContent = $("#samSimplify").value; };
      $("#samSimplify").onchange = () => {
        localStorage.setItem("yaap.samSimplify", $("#samSimplify").value);
        if (editor.samPts.length || editor.samBox) runSamPrompt(editor);   // re-segment with the new value
      };

      if (!samInfo.ultralytics)
        toast("SAM needs ultralytics installed on the server (pip install ultralytics).", "err");
      samReady = true;
    } catch (e) { err(e); }
  }
  function samHint(t) { const h = $("#samHint"); if (h) h.textContent = t; }
  function updateSamStrip(ed) {
    const has = ed.samPending.length > 0;
    const a = $("#samAccept"), d = $("#samDiscard");
    if (a) { a.hidden = !has; d.hidden = !has && !ed.samPts.length && !ed.samBox; }
    if (!has) samHint("click ＋ · shift-click − · drag box");
    renderSamLabelPick();
  }

  /* small class picker that pops up right next to a fresh SAM mask —
     click a class to label + accept it in one step, no sidebar detour. */
  function renderSamLabelPick() {
    const box = $("#samLabelPick");
    if (!box) return;
    if (!editor.samPending.length || !p.classes.length) { box.hidden = true; return; }
    box.innerHTML = p.classes.map(c => `
      <button class="sam-label-chip" data-id="${c.id}">
        <span class="swatch" style="background:${c.color}"></span>${esc(c.name)}</button>`).join("");
    $$(".sam-label-chip", box).forEach(b => b.onclick = () => { box.hidden = true; acceptSam(+b.dataset.id); });
    const r0 = editor.samPending[0];
    const [sx, sy] = editor.toScr(r0.bbox.x + r0.bbox.w / 2, r0.bbox.y);
    box.hidden = false;
    box.style.left = `${sx}px`;
    box.style.top = `${Math.max(4, sy)}px`;
  }
  async function runSamPrompt(ed) {
    if (samBusy) return;
    const im = S.images[S.curImage];
    const prompt = ed.samBox
      ? { type: "box", box: [ed.samBox.x0, ed.samBox.y0, ed.samBox.x1, ed.samBox.y1] }
      : { type: "point", points: ed.samPts.map(q => [q[0], q[1]]), labels: ed.samPts.map(q => q[2]) };
    await callSam(prompt);
  }
  async function callSam(prompt) {
    const im = S.images[S.curImage];
    samBusy = true; samHint("segmenting…");
    try {
      const r = await API.samPredict(p.id, im.id,
        { model_name: $("#samModel").value, prompt, simplify: +$("#samSimplify").value });
      editor.samPending = r.results || [];
      samHint(editor.samPending.length
        ? `${editor.samPending.length} mask(s) — refine with more clicks, then Accept`
        : "no mask found — add clicks or adjust the box");
    } catch (e) { samHint(""); err(e); }
    samBusy = false;
    updateSamStrip(editor); editor.draw();
  }
  $("#samAccept") && ($("#samAccept").onclick = () => acceptSam());
  $("#samDiscard") && ($("#samDiscard").onclick = () => editor.clearSam());
  function acceptSam(clsId) {
    if (!editor.samPending.length) return;
    const cls = p.classes.find(c => c.id === (clsId ?? S.activeClassId));
    if (!cls) return toast("Pick a class first.", "err");
    for (const r of editor.samPending) {
      if (p.task_type === "detect")            // segmentation → detection format
        editor.anns.push({ class_id: cls.id, kind: "bbox", source: "model",
          confidence: r.score, data: { ...r.bbox } });
      else                                     // instance segmentation format
        editor.anns.push({ class_id: cls.id, kind: "polygon", source: "model",
          confidence: r.score, data: { points: r.polygon.map(q => [q[0], q[1]]) } });
    }
    toast(`Added ${editor.samPending.length} ${p.task_type === "detect" ? "box(es)" : "polygon(s)"} as “${cls.name}”.`, "ok");
    editor.clearSam(false);
    markDirty(); renderAnnList(); editor.draw();
  }

  $("#prevImg").onclick = () => go(S.curImage - 1);
  $("#nextImg").onclick = () => go(S.curImage + 1);
  window.onkeydown = e => {
    if (e.target.matches("input,textarea,[contenteditable]")) return;
    if (e.key === "ArrowLeft") go(S.curImage - 1);
    if (e.key === "ArrowRight") go(S.curImage + 1);
  };

  let dirty = false;
  function markDirty() {
    dirty = true;
    const st = $("#saveState"); st.textContent = "unsaved"; st.className = "savestate dirty";
    clearTimeout(S.saveTimer);
    S.saveTimer = setTimeout(save, 700);
  }
  async function save() {
    if (!dirty) return;
    const im = S.images[S.curImage];
    try {
      await API.saveAnnotations(p.id, im.id, editor.anns.map(a =>
        ({ class_id: a.class_id, kind: a.kind, data: a.data, source: a.source, confidence: a.confidence })));
      dirty = false;
      const st = $("#saveState"); st.textContent = "saved"; st.className = "savestate saved";
      im.annotation_count = editor.anns.length;
      im.status = editor.anns.length ? "annotated" : "unannotated";
    } catch (e) { err(e); }
  }

  async function go(i) {
    await save();
    if (i < 0 || i >= S.images.length) return;
    S.curImage = i;
    await loadImage();
  }

  async function loadImage() {
    const im = S.images[S.curImage];
    $("#edFn").textContent = im.filename;
    $("#imgPos").textContent = `${S.curImage + 1}/${S.images.length}`;
    const warnEl = $("#edAnnotatedWarn");
    if (warnEl) warnEl.hidden = !(warnIfAnnotated && im.status !== "unannotated");
    const [el, anns] = await Promise.all([
      new Promise((res, rej) => { const x = new Image(); x.onload = () => res(x); x.onerror = rej; x.src = API.imageUrl(p.id, im.id); }),
      API.annotations(p.id, im.id),
    ]);
    editor.load(el, anns);
    $("#edHud").textContent = `${im.width}×${im.height}px`;
    renderApprove(); renderReject();
    renderSide();
  }

  function renderApprove() {
    const im = S.images[S.curImage];
    const btn = $("#edApprove"); if (!btn) return;
    btn.textContent = im.status === "approved" ? "✓ Approved" : "✓ Approve";
    btn.classList.toggle("primary", im.status === "approved");
  }
  $("#edApprove") && ($("#edApprove").onclick = async () => {
    const im = S.images[S.curImage];
    try {
      await API.patchImage(p.id, im.id, { status: im.status === "approved" ? "annotated" : "approved" });
      im.status = im.status === "approved" ? "annotated" : "approved";
      renderApprove(); renderReject();
      toast(im.status === "approved" ? "Marked as approved." : "Approval cleared.", "ok");
    } catch (e) { err(e); }
  });

  function renderReject() {
    const im = S.images[S.curImage];
    const btn = $("#edReject"); if (!btn) return;
    btn.textContent = im.status === "review" ? "✕ Rejected" : "✕ Reject";
    btn.classList.toggle("danger", im.status === "review");
  }
  $("#edReject") && ($("#edReject").onclick = async () => {
    const im = S.images[S.curImage];
    try {
      const next = im.status === "review" ? "annotated" : "review";
      await API.patchImage(p.id, im.id, { status: next });
      im.status = next;
      renderApprove(); renderReject();
      toast(im.status === "review" ? "Flagged for review." : "Rejection cleared.", "ok");
    } catch (e) { err(e); }
  });

  function renderSide() {
    if (isCls) return renderClsPanel();
    $("#edClasses").innerHTML = p.classes.map((c, i) => `
      <button class="ed-class ${c.id === S.activeClassId ? "active" : ""}" data-id="${c.id}">
        <span class="swatch" style="background:${c.color}"></span>${esc(c.name)}
        <span class="key">${i < 9 ? i + 1 : ""}</span>
      </button>`).join("");
    $$(".ed-class").forEach(b => b.onclick = () => {
      S.activeClassId = +b.dataset.id;
      if (editor.sel >= 0) { editor.anns[editor.sel].class_id = S.activeClassId; markDirty(); editor.draw(); }
      renderSide();
    });
    renderAnnList();
  }

  function renderAnnList() {
    const host = $("#edAnns"); if (!host) return renderClsPanel();
    host.innerHTML = editor.anns.map((a, i) => {
      const c = p.classes.find(x => x.id === a.class_id) || {};
      return `<div class="ed-ann ${i === editor.sel ? "sel" : ""}" data-i="${i}">
        <span class="swatch" style="background:${c.color || "#888"}"></span>
        <select class="relabel" data-i="${i}" title="Change label">
          ${p.classes.map(cl => `<option value="${cl.id}" ${cl.id === a.class_id ? "selected" : ""}>${esc(cl.name)}</option>`).join("")}
        </select>
        <span class="kind">${a.kind === "bbox" ? "box" : "poly"}</span>
        ${a.source === "model" ? `<span class="src-model">model ${Math.round(a.confidence * 100)}%</span>` : ""}
        <button class="rm" title="Remove">✕</button></div>`;
    }).join("") || `<div class="empty" style="padding:16px;border:none">Draw with <span class="kbd">B</span>${p.task_type !== "detect" ? " or <span class='kbd'>P</span>" : ""}</div>`;
    $$(".ed-ann", host).forEach(row => {
      row.onclick = e => {
        const i = +row.dataset.i;
        if (e.target.matches(".rm")) { editor.anns.splice(i, 1); editor.select(-1); markDirty(); editor.draw(); renderAnnList(); return; }
        if (e.target.matches(".relabel")) return;         // handled by its own onchange
        editor.select(i);
        editor.draw(); renderAnnList();
      };
    });
    $$(".relabel", host).forEach(sel => {
      sel.onclick = e => e.stopPropagation();
      sel.onchange = () => {
        const i = +sel.dataset.i;
        editor.anns[i].class_id = +sel.value;
        markDirty(); editor.draw(); renderAnnList();
      };
    });
  }

  function renderClsPanel() {
    const host = $("#clsPanel"); if (!host) return;
    const cur = editor.anns.find(a => a.kind === "classification");
    host.innerHTML = p.classes.map(c => `
      <button class="cls-big ${cur && cur.class_id === c.id ? "active" : ""}" data-id="${c.id}">
        <span class="swatch" style="background:${c.color}"></span>${esc(c.name)}</button>`).join("");
    $$(".cls-big", host).forEach(b => b.onclick = () => {
      editor.anns = [{ class_id: +b.dataset.id, kind: "classification", data: {}, source: "manual", confidence: 1 }];
      markDirty(); renderClsPanel();
    });
  }

  await loadImage();
}

/* ── auto-label ───────────────────────────────────────────────────── */
async function viewAutolabel() {
  const p = S.project;
  let models;
  try { models = await API.models(p.id, p.task_type); } catch (e) { return err(e); }
  const opts = [
    ...models.weights.map(w => `<option value="weight:${w.id}">★ ${esc(w.name)} (${w.source})</option>`),
    ...models.pretrained.map(m => `<option value="${m}">${m}</option>`),
    ...(models.rtdetr || []).map(m => `<option value="${m}">${m} (RT-DETR)</option>`),
    ...(models.rfdetr_available ? models.rfdetr : []).map(m => `<option value="${m}">${m} (RF-DETR)</option>`),
  ].join("");

  $("#stage").innerHTML = `<div class="stage-pad" style="max-width:680px">
    <h1 class="page">Auto-label</h1>
    <p class="sub">Run a YOLO${(models.rtdetr || []).length ? ", RT-DETR" : ""}${models.rfdetr_available ? " or RF-DETR" : ""} model over your images and
      turn its predictions into editable annotations. Model output is marked
      <span style="color:var(--teal)">for review</span> so you can verify each image in the editor.
      ${!models.rfdetr_available && p.task_type === "detect" ? "RF-DETR is not installed (<span class='mono'>pip install rfdetr</span>)." : ""}</p>
    <div class="card">
      <label class="fld">Model — pretrained aliases download once, then run fully offline</label>
      <select class="input" id="alModel">${opts}</select>
      <div class="grid2">
        <div><label class="fld">Confidence threshold</label>
          <input class="input" id="alConf" type="number" min="0.05" max="0.95" step="0.05" value="0.4"></div>
        <div><label class="fld">Scope</label>
          <select class="input" id="alScope">
            <option value="all">All images</option>
            <option value="unannotated">Only images without annotations</option>
          </select></div>
      </div>
      <label class="fld"><input type="checkbox" id="alReplace"> Replace existing annotations on each image</label>
      <label class="fld"><input type="checkbox" id="alCreate" checked> Create project classes from model class names</label>
      <div class="row" style="margin-top:16px">
        <button class="btn primary" id="alGo">Run auto-label</button>
        <span class="chip mono" id="alStatus"></span>
      </div>
    </div>
    <h3 class="sect">Bring your own weights</h3>
    <div class="card row">
      <input type="file" id="wtFile" accept=".pt" class="input" style="flex:1">
      <button class="btn" id="wtGo">Upload .pt</button>
    </div></div>`;

  $("#alGo").onclick = async () => {
    const btn = $("#alGo"); btn.disabled = true;
    $("#alStatus").textContent = "running… (first run may download weights)";
    try {
      const imgs = await API.images(p.id);
      const ids = $("#alScope").value === "unannotated"
        ? imgs.filter(i => !i.annotation_count).map(i => i.id) : [];
      const res = await API.autolabel(p.id, {
        model_name: $("#alModel").value, conf: +$("#alConf").value, image_ids: ids,
        replace_existing: $("#alReplace").checked, create_missing_classes: $("#alCreate").checked,
      });
      $("#alStatus").textContent = "";
      toast(`Labeled ${res.images_labeled}/${res.images_total} images · +${res.annotations_added} annotations`, "ok");
    } catch (e) { $("#alStatus").textContent = ""; err(e); }
    btn.disabled = false;
  };
  $("#wtGo").onclick = async () => {
    const f = $("#wtFile").files[0];
    if (!f) return;
    try { await API.uploadWeight(f, p.task_type, p.id); toast("Weights uploaded.", "ok"); viewAutolabel(); }
    catch (e) { err(e); }
  };
}

/* ── generate (preprocess + augment + version) ────────────────────── */
const AUG_DEFS = [
  ["hflip", "Horizontal flip", "check"], ["vflip", "Vertical flip", "check"],
  ["rotate90", "Rotate 90°", "check"], ["rotate", "Rotate ±deg", "range", 0, 45, 1],
  ["shear", "Shear ±deg", "range", 0, 25, 1],
  ["brightness", "Brightness", "range", 0, 0.6, 0.05], ["contrast", "Contrast", "range", 0, 0.6, 0.05],
  ["hue", "Hue shift", "range", 0, 0.3, 0.02], ["saturation", "Saturation", "range", 0, 0.6, 0.05],
  ["blur", "Blur", "range", 0, 1, 0.1], ["noise", "Noise", "range", 0, 1, 0.1],
  ["cutout", "Cutout", "range", 0, 1, 0.1], ["clahe", "CLAHE", "check"], ["gray_p", "Random grayscale", "check"],
];

async function viewGenerate() {
  const p = S.project;
  $("#stage").innerHTML = `<div class="stage-pad" style="max-width:760px">
    <h1 class="page">Generate dataset version</h1>
    <p class="sub">Creates a frozen, ultralytics-ready snapshot: split into train/val/test,
      preprocessed, augmented (train split only — val/test always stay clean) and packaged
      with a <span class="mono">data.yaml</span> — ready for <span class="mono">yolo train</span>.</p>

    <div class="card">
      <label class="fld" style="margin-top:0"><input type="checkbox" id="genApproved" checked>
        Only include <span style="color:var(--blue)">approved</span> images — verify annotations in
        the Review tab first</label>
      ${p.task_type === "classify" ? "" : `
      <h3 class="sect">Export format</h3>
      <select class="input" id="genFormat" style="max-width:360px">
        <option value="yolo">YOLO — data.yaml + txt labels, for YOLO / RT-DETR via ultralytics</option>
        <option value="coco">COCO — _annotations.coco.json per split, for RF-DETR / HuggingFace / detectron2</option>
      </select>`}
      <h3 class="sect">Split</h3>
      <div class="row">
        <div style="flex:1"><label class="fld">Train %</label><input class="input" id="spTrain" type="number" value="70" min="0" max="100"></div>
        <div style="flex:1"><label class="fld">Valid %</label><input class="input" id="spVal" type="number" value="20" min="0" max="100"></div>
        <div style="flex:1"><label class="fld">Test %</label><input class="input" id="spTest" type="number" value="10" min="0" max="100"></div>
      </div>
      <h3 class="sect">Preprocess</h3>
      <div class="row">
        <div style="flex:1"><label class="fld">Resize (px, 0 = keep)</label><input class="input" id="ppSize" type="number" value="640"></div>
        <div style="flex:1"><label class="fld">Fit mode</label>
          <select class="input" id="ppMode"><option value="stretch">Stretch</option><option value="letterbox">Letterbox (pad)</option></select></div>
        <div style="flex:1"><label class="fld">Grayscale</label><select class="input" id="ppGray"><option value="">No</option><option value="1">Yes</option></select></div>
      </div>
      <h3 class="sect">Augmentation — applied to the train split</h3>
      <div class="row"><label class="fld" style="margin:0">Outputs per training image</label>
        <input class="input" id="augMult" type="number" value="3" min="1" max="10" style="width:80px"></div>
      <div id="augOps" style="margin-top:8px">
        ${AUG_DEFS.map(d => d[2] === "check"
          ? `<div class="aug-op"><span>${d[1]}</span><span></span><input type="checkbox" data-op="${d[0]}"></div>`
          : `<div class="aug-op"><span>${d[1]}</span>
               <input type="range" data-op="${d[0]}" min="${d[3]}" max="${d[4]}" step="${d[5]}" value="0">
               <span class="val">0</span></div>`).join("")}
      </div>
      <div class="row" style="margin-top:14px">
        <select class="input" id="pvImage" style="max-width:280px"></select>
        <button class="btn" id="pvGo">Preview augmentations</button>
      </div>
      <img id="augPreview" alt="augmentation preview">

      <label class="fld" style="margin-top:20px"><input type="checkbox" id="showSnippet">
        Show a matching <span class="mono">model.train()</span> snippet with ultralytics'
        built-in augmentation disabled for whatever's enabled above — avoids double-augmenting</label>
      <div id="snippetBox" style="display:none">
        <div class="row" style="justify-content:flex-end;margin-bottom:6px">
          <button class="btn small" id="snippetCopy">Copy</button>
        </div>
        <pre class="logbox" id="snippetText" style="max-height:none"></pre>
      </div>

      <div class="row" style="margin-top:20px">
        <input class="input" id="verName" placeholder="Version name (optional)" style="max-width:220px">
        <button class="btn primary" id="verGo">⚙ Generate version</button>
      </div>
    </div></div>`;

  $$("#augOps input[type=range]").forEach(r =>
    r.oninput = () => { r.parentElement.querySelector(".val").textContent = r.value; renderSnippet(); });
  $$("#augOps input[type=checkbox]").forEach(c => c.onchange = renderSnippet);
  $("#ppSize").oninput = renderSnippet;
  $("#verName").oninput = renderSnippet;
  $("#showSnippet").onchange = renderSnippet;
  $("#snippetCopy").onclick = () => {
    navigator.clipboard.writeText($("#snippetText").textContent);
    toast("Snippet copied.", "ok");
  };

  try {
    S.images = await API.images(p.id);
    $("#pvImage").innerHTML = S.images.map(im => `<option value="${im.id}">${esc(im.filename)}</option>`).join("");
  } catch (e) { err(e); }

  const collectOps = () => {
    const ops = {};
    $$("#augOps [data-op]").forEach(el => {
      const v = el.type === "checkbox" ? el.checked : +el.value;
      if (v) ops[el.dataset.op] = v;
    });
    return ops;
  };

  function buildSnippet() {
    const ops = collectOps();
    const disableKeys = new Set();
    Object.keys(ops).forEach(k => (AUG_OVERLAP[k] || []).forEach(uk => disableKeys.add(uk)));
    const imgsz = +$("#ppSize").value || 640;
    const name = ($("#verName").value || "train").trim().replace(/\s+/g, "_") || "train";

    const lines = ["data=DATA_YAML", "epochs=100", "batch=8", "patience=6",
      `imgsz=${imgsz}`, 'project="runs"', `name="${name}"`];
    if (disableKeys.size) {
      lines.push("# ultralytics' built-in augmentation disabled — already applied by YAAP's Generate step");
      for (const k of disableKeys) lines.push(`${k}=${k === "auto_augment" ? "None" : 0}`);
    }
    const body = lines.map(l => l.startsWith("#") ? `    ${l}` : `    ${l},`).join("\n");
    return `from ultralytics import YOLO\n\n` +
      `DATA_YAML = "<copy the path shown for this version in the Versions tab>"\n\n` +
      `model = YOLO("yolo26s.pt")\nresults = model.train(\n${body}\n)`;
  }

  function renderSnippet() {
    const show = $("#showSnippet").checked;
    $("#snippetBox").style.display = show ? "block" : "none";
    if (show) $("#snippetText").textContent = buildSnippet();
  }

  $("#pvGo").onclick = async () => {
    if (!S.images.length) return toast("Upload images first.", "err");
    try {
      const url = await API.augPreview(p.id, { image_id: +$("#pvImage").value, ops: collectOps(), count: 4 });
      const img = $("#augPreview"); img.src = url; img.style.display = "block";
    } catch (e) { err(e); }
  };

  $("#verGo").onclick = async () => {
    try {
      await API.createVersion(p.id, {
        name: $("#verName").value,
        format: $("#genFormat")?.value || "yolo",
        splits: { train: +$("#spTrain").value, val: +$("#spVal").value, test: +$("#spTest").value },
        preprocess: { resize: +$("#ppSize").value, mode: $("#ppMode").value, grayscale: !!$("#ppGray").value },
        augment: { multiplier: +$("#augMult").value, ops: collectOps() },
        approved_only: $("#genApproved").checked,
      });
      toast("Version is building — check the Versions tab.", "ok");
      nav("versions");
    } catch (e) { err(e); }
  };
}

/* ── versions ─────────────────────────────────────────────────────── */
async function viewVersions() {
  const p = S.project;
  $("#stage").innerHTML = `<div class="stage-pad" style="max-width:820px">
    <h1 class="page">Versions</h1>
    <p class="sub">Frozen datasets in ultralytics layout. Download the zip or train on it directly
      from the Train tab — the folder path inside <span class="mono">data/</span> is shown per version.</p>
    <div id="verList"></div></div>`;

  async function render() {
    let versions = [];
    try { versions = await API.versions(p.id); } catch (e) { return err(e); }
    $("#verList").innerHTML = versions.map(v => {
      const st = v.stats || {}, src = st.source_images || {}, total = st.source_images_total ?? 0;
      const pct = n => total ? Math.round((n / total) * 100) : 0;
      const meta = v.status === "ready"
        ? `${(st.format || "yolo").toUpperCase()} · ${total} image(s) · train ${src.train ?? 0} (${pct(src.train ?? 0)}%) · ` +
          `val ${src.val ?? 0} (${pct(src.val ?? 0)}%) · test ${src.test ?? 0} (${pct(src.test ?? 0)}%)` +
          (st.annotations != null ? ` · ${st.annotations} labels` : "") : (st.error || "");
      return `<div class="v-row">
        <span class="vname">${esc(v.name)}</span>
        <span class="pill ${v.status}">${v.status}</span>
        <span class="meta">${esc(meta)}</span>
        ${v.status === "ready" ? `<a class="btn small" href="${API.versionZip(p.id, v.id)}">⬇ zip</a>` : ""}
        <button class="btn ghost small" data-del="${v.id}">✕</button>
      </div>`;
    }).join("") || `<div class="empty">No versions yet — use Generate.</div>`;
    $$("[data-del]").forEach(b => b.onclick = async () => {
      try { await API.deleteVersion(p.id, +b.dataset.del); render(); } catch (e) { err(e); }
    });
  }
  await render();
  S.verPoll = setInterval(render, 3000);
}

/* ── ultralytics train() hyperparameters + built-in augmentation ─────
   https://docs.ultralytics.com/modes/train/#train-settings
   https://docs.ultralytics.com/modes/train/#augmentation-settings
   Every row is opt-in — unchecked/"(default)" means the key is simply not
   sent, so ultralytics' own default applies. Types:
     num       — checkbox "override" + number input
     opt-select — checkbox "override" + a fixed choice of strings
     bool3     — a single tri-state select: (default) / On / Off           */
const TRAIN_HYPERPARAMS = [
  ["optimizer", "Optimizer", "opt-select", ["auto", "SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"]],
  ["lr0", "Initial LR (lr0)", "num", 0.01, 0, 1, 0.0001],
  ["lrf", "Final LR fraction (lrf)", "num", 0.01, 0, 1, 0.0001],
  ["momentum", "Momentum", "num", 0.937, 0, 1, 0.001],
  ["weight_decay", "Weight decay", "num", 0.0005, 0, 1, 0.0001],
  ["warmup_epochs", "Warmup epochs", "num", 3.0, 0, 20, 0.5],
  ["warmup_momentum", "Warmup momentum", "num", 0.8, 0, 1, 0.01],
  ["warmup_bias_lr", "Warmup bias LR", "num", 0.1, 0, 1, 0.01],
  ["box", "Box loss gain", "num", 7.5, 0, 20, 0.1],
  ["cls", "Class loss gain", "num", 0.5, 0, 20, 0.1],
  ["dfl", "DFL loss gain", "num", 1.5, 0, 20, 0.1],
  ["label_smoothing", "Label smoothing", "num", 0.0, 0, 1, 0.01],
  ["nbs", "Nominal batch size", "num", 64, 1, 256, 1],
  ["close_mosaic", "Close mosaic — last N epochs", "num", 10, 0, 100, 1],
  ["mask_ratio", "Mask downsample ratio (segment)", "num", 4, 1, 16, 1],
  ["dropout", "Dropout (classify)", "num", 0.0, 0, 1, 0.05],
  ["cos_lr", "Cosine LR schedule", "bool3"],
  ["overlap_mask", "Overlap masks (segment)", "bool3"],
  ["amp", "Automatic mixed precision", "bool3"],
  ["multi_scale", "Multi-scale training", "bool3"],
  ["val", "Validate during training", "bool3"],
];

// Maps a YAAP Generate-step augmentation op to the ultralytics train()
// kwarg(s) it overlaps with — used both to render this box and to build the
// "disable what YAAP already did" snippet in the Generate view.
const AUG_OVERLAP = {
  hflip: ["fliplr"], vflip: ["flipud"], rotate90: ["degrees"], rotate: ["degrees"],
  shear: ["shear"], brightness: ["hsv_v"], contrast: ["hsv_v"], hue: ["hsv_h"],
  saturation: ["hsv_s"], cutout: ["erasing"],
};

// Model families offered on the Google Colab tab — same ultralytics train()
// call shape either way, just a different Model subclass + pretrained alias.
const COLAB_MODELS = {
  yolo: { label: "YOLO", importClass: "YOLO", weights: "yolo26s.pt",
    guideUrl: "https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb",
    guideLabel: "Open Ultralytics' Colab notebook ↗" },
  rtdetr: { label: "RT-DETR", importClass: "RTDETR", weights: "rtdetr-l.pt",
    guideUrl: "https://docs.ultralytics.com/models/rtdetr",
    guideLabel: "Open Ultralytics' RT-DETR guide ↗" },
};

const TRAIN_AUG_DEFS = [
  ["hsv_h", "HSV — Hue", "num", 0.015, 0, 1, 0.001],
  ["hsv_s", "HSV — Saturation", "num", 0.7, 0, 1, 0.01],
  ["hsv_v", "HSV — Value", "num", 0.4, 0, 1, 0.01],
  ["degrees", "Rotation ± deg", "num", 0.0, 0, 180, 1],
  ["translate", "Translate", "num", 0.1, 0, 1, 0.01],
  ["scale", "Scale gain", "num", 0.5, 0, 1, 0.01],
  ["shear", "Shear ± deg", "num", 0.0, 0, 180, 1],
  ["perspective", "Perspective", "num", 0.0, 0, 0.001, 0.0001],
  ["flipud", "Flip up-down probability", "num", 0.0, 0, 1, 0.05],
  ["fliplr", "Flip left-right probability", "num", 0.5, 0, 1, 0.05],
  ["bgr", "BGR channel-swap probability", "num", 0.0, 0, 1, 0.05],
  ["mosaic", "Mosaic probability", "num", 1.0, 0, 1, 0.05],
  ["mixup", "MixUp probability", "num", 0.0, 0, 1, 0.05],
  ["cutmix", "CutMix probability", "num", 0.0, 0, 1, 0.05],
  ["copy_paste", "Copy-paste probability", "num", 0.0, 0, 1, 0.05],
  ["copy_paste_mode", "Copy-paste mode", "opt-select", ["flip", "mixup"]],
  ["auto_augment", "Auto augment (classify)", "opt-select", ["randaugment", "autoaugment", "augmix", "none"]],
  ["erasing", "Random erasing probability", "num", 0.4, 0, 1, 0.05],
  ["crop_fraction", "Crop fraction (classify)", "num", 1.0, 0.1, 1, 0.05],
];

function renderHpRow([key, label, type, ...rest]) {
  if (type === "bool3") {
    return `<div class="hp-op"><span>${label}</span>
      <select class="input" data-hp="${key}">
        <option value="">(default)</option><option value="1">On</option><option value="0">Off</option>
      </select></div>`;
  }
  if (type === "opt-select") {
    const [opts] = rest;
    return `<div class="hp-op"><label><input type="checkbox" data-hpov="${key}">${esc(label)}</label>
      <select class="input" data-hp="${key}" disabled>${opts.map(o => `<option value="${o}">${o}</option>`).join("")}</select></div>`;
  }
  const [def, min, max, step] = rest;
  return `<div class="hp-op"><label><input type="checkbox" data-hpov="${key}">${esc(label)}</label>
    <input class="input" type="number" data-hp="${key}" value="${def}" min="${min}" max="${max}" step="${step}" disabled></div>`;
}

function wireHpRows(host) {
  $$("[data-hpov]", host).forEach(cb => cb.onchange = () => {
    const inp = $(`[data-hp="${cb.dataset.hpov}"]`, host);
    inp.disabled = !cb.checked;
  });
}

function collectTrainExtra(host) {
  const extra = {};
  [...TRAIN_HYPERPARAMS, ...TRAIN_AUG_DEFS].forEach(([key, , type]) => {
    if (type === "bool3") {
      const sel = $(`[data-hp="${key}"]`, host);
      if (sel && sel.value !== "") extra[key] = sel.value === "1";
    } else {
      const cb = $(`[data-hpov="${key}"]`, host);
      if (cb && cb.checked) {
        const el = $(`[data-hp="${key}"]`, host);
        if (key === "auto_augment" && el.value === "none") extra[key] = null;
        else extra[key] = el.tagName === "SELECT" ? el.value : +el.value;
      }
    }
  });
  return extra;
}

/* ── train ────────────────────────────────────────────────────────── */
async function viewTrain() {
  const p = S.project;
  let versions = [], models;
  try { [versions, models] = await Promise.all([API.versions(p.id), API.models(p.id, p.task_type)]); }
  catch (e) { return err(e); }
  const ready = versions.filter(v => v.status === "ready");
  const verOpts = ready.map(v => `<option value="${v.id}">${esc(v.name)}</option>`).join("")
    || `<option value="">No versions yet — use Generate</option>`;

  $("#stage").innerHTML = `<div class="stage-pad" style="max-width:820px">
    <h1 class="page">Train</h1>
    <p class="sub">Two ways to train a model from a generated version: locally through this app, or on
      Google's free GPUs via Colab.</p>

    <div class="row" style="margin-bottom:16px" id="trainTabs">
      <button class="btn small primary" data-tab="local">🖥 Local train</button>
      <button class="btn small ghost" data-tab="colab">☁ Google Colab</button>
    </div>

    <div id="trainLocalPane">
      <div class="card" style="border-color:var(--amber-dim);background:rgba(255,210,63,.06);margin-bottom:14px">
        <b style="color:var(--amber)">⚠ Known issue</b> — local training doesn't currently complete
        successfully. If you hit an error, open the job's <span class="mono">log</span> below (once
        started/failed jobs appear under Jobs) so it can be diagnosed with the actual traceback — until
        then, use the Google Colab tab instead.
      </div>
      <div class="card">
        <div class="grid2">
          <div><label class="fld">Dataset version</label>
            <select class="input" id="trVer">${verOpts}</select></div>
          <div><label class="fld">Architecture / starting weights</label>
            <select class="input" id="trArch">
              ${models.pretrained.map(m => `<option ${m === "yolo11n.pt" ? "selected" : ""}>${m}</option>`).join("")}
              ${(models.rtdetr || []).map(m => `<option value="${m}">${m} (RT-DETR)</option>`).join("")}
              ${models.weights.map(w => `<option value="weight:${w.id}">★ ${esc(w.name)}</option>`).join("")}
            </select></div>
          <div><label class="fld">Epochs</label><input class="input" id="trEpochs" type="number" value="100"></div>
          <div><label class="fld">Image size</label><input class="input" id="trImgsz" type="number" value="640"></div>
          <div><label class="fld">Batch</label><input class="input" id="trBatch" type="number" value="16"></div>
          <div><label class="fld">Device</label>
            <select class="input" id="trDevice"><option value="auto">auto (GPU if available)</option>
              <option value="cpu">cpu</option><option value="cuda:0">cuda:0</option></select></div>
        </div>

        <details class="hp-details" id="trHpDetails">
          <summary>Optional hyperparameters &amp; augmentations (ultralytics <span class="mono">train()</span> settings)</summary>
          <div class="hp-body">
            <p class="sub" style="margin:4px 0 12px">Everything here is opt-in — leave a row untouched and
              ultralytics uses its own default. If you already augmented this version in <b>Generate</b>,
              turn the matching rows below off to avoid double-augmenting.</p>
            <h3 class="sect" style="margin-top:0">Hyperparameters</h3>
            <div class="hp-grid">${TRAIN_HYPERPARAMS.map(renderHpRow).join("")}</div>
            <h3 class="sect">Built-in augmentation</h3>
            <div class="hp-grid">${TRAIN_AUG_DEFS.map(renderHpRow).join("")}</div>
          </div>
        </details>

        <div class="row" style="margin-top:16px">
          <button class="btn primary" id="trGo" ${ready.length ? "" : "disabled"}>▶ Start training</button>
          ${ready.length ? "" : `<span class="chip warn">Generate a dataset version first</span>`}
        </div>
      </div>
      <h3 class="sect">Jobs</h3>
      <div id="jobList"></div>
      <div class="joblog-head" id="jobLogHead">
        <span class="mono" style="font-size:11px;color:var(--muted)" id="jobLogTitle"></span>
        <button class="btn small" id="jobLogExpand">⤢ Expand</button>
      </div>
      <div class="logbox" id="jobLog" style="display:none"></div>
    </div>

    <div id="trainColabPane" hidden>
      <div class="card train-colab">
        <h3 class="sect" style="margin-top:0;color:var(--red)">☁ Not local — runs on Google's servers</h3>
        <p class="sub" style="margin-bottom:14px">This path uploads your dataset to Google Colab and trains
          on Google's GPU, not this machine. Everything else in YAAP stays local by design — this is the
          one deliberate exception, so it's marked clearly instead of blending in.</p>
        <label class="fld" style="margin-top:0">Model</label>
        <div class="row" id="colabModelToggle" style="margin-bottom:14px">
          ${Object.entries(COLAB_MODELS).map(([k, m]) => `
            <button class="btn small ${k === "yolo" ? "primary" : "ghost"}" data-fam="${k}">${m.label}</button>`).join("")}
        </div>
        <label class="fld" style="margin-top:0">Dataset version — download its zip from the Versions tab
          first, then upload/extract it in Colab</label>
        <select class="input" id="colabVer" style="max-width:320px">${verOpts}</select>
        <div class="row" style="justify-content:space-between;margin:16px 0 6px;flex-wrap:wrap;gap:8px">
          <span style="font-weight:600;color:var(--muted)">Notebook template</span>
          <div class="row">
            <a class="btn small" id="colabGuideLink" target="_blank" rel="noopener"></a>
            <button class="btn small" id="colabCopy">Copy</button>
          </div>
        </div>
        <pre class="logbox" id="colabSnippet" style="max-height:none"></pre>
        <p class="sub" style="margin-top:10px;margin-bottom:0">🚧 Placeholder, based on the guide linked
          above — swap it for your own notebook code whenever you're ready to share it.</p>
      </div>
    </div>
  </div>`;

  function setTrainTab(tab) {
    $$("#trainTabs [data-tab]").forEach(b => {
      const active = b.dataset.tab === tab;
      b.classList.toggle("primary", active);
      b.classList.toggle("ghost", !active);
    });
    $("#trainLocalPane").hidden = tab !== "local";
    $("#trainColabPane").hidden = tab !== "colab";
  }
  $$("#trainTabs [data-tab]").forEach(b => b.onclick = () => setTrainTab(b.dataset.tab));

  let colabFamily = "yolo";
  function buildColabSnippet() {
    const fam = COLAB_MODELS[colabFamily];
    const v = ready.find(x => x.id === +$("#colabVer").value);
    const stem = (v ? v.name : "dataset").trim().replace(/\s+/g, "_") || "dataset";
    return `# Colab notebook template — see ${fam.guideUrl}
# 🚧 placeholder — replace with your own notebook code when ready.

!pip install ultralytics

from ultralytics import ${fam.importClass}

# Download this version's zip from YAAP's Versions tab, upload/extract it in
# Colab, then point this at the extracted data.yaml
DATA_YAML = "/content/${stem}/data.yaml"

model = ${fam.importClass}("${fam.weights}")
results = model.train(
    data=DATA_YAML,
    epochs=100,
    batch=8,
    patience=6,
    imgsz=640,
    project="runs",
    name="${stem}",
)`;
  }
  function renderColabSnippet() {
    const fam = COLAB_MODELS[colabFamily];
    $("#colabSnippet").textContent = buildColabSnippet();
    $("#colabGuideLink").href = fam.guideUrl;
    $("#colabGuideLink").textContent = fam.guideLabel;
  }
  $$("#colabModelToggle [data-fam]").forEach(b => b.onclick = () => {
    colabFamily = b.dataset.fam;
    $$("#colabModelToggle [data-fam]").forEach(x => {
      const active = x.dataset.fam === colabFamily;
      x.classList.toggle("primary", active);
      x.classList.toggle("ghost", !active);
    });
    renderColabSnippet();
  });
  $("#colabVer") && ($("#colabVer").onchange = renderColabSnippet);
  $("#colabCopy") && ($("#colabCopy").onclick = () => {
    navigator.clipboard.writeText($("#colabSnippet").textContent);
    toast("Snippet copied.", "ok");
  });
  renderColabSnippet();

  wireHpRows($("#trHpDetails"));

  $("#trGo").onclick = async () => {
    try {
      await API.train(p.id, {
        version_id: +$("#trVer").value, model_arch: $("#trArch").value,
        epochs: +$("#trEpochs").value, imgsz: +$("#trImgsz").value,
        batch: +$("#trBatch").value, device: $("#trDevice").value,
        extra: collectTrainExtra($("#trHpDetails")),
      });
      toast("Training started.", "ok"); renderJobs();
    } catch (e) { err(e); }
  };

  let watching = null, expanded = false;
  async function renderJobs() {
    let jobs = [];
    try { jobs = await API.jobs(p.id); } catch (e) { return; }
    $("#jobList").innerHTML = jobs.map(j => `
      <div class="v-row">
        <span class="vname mono">#${j.id}</span>
        <span class="pill ${j.status}">${j.status}</span>
        <span class="meta">${esc(j.model_arch)} · version ${j.version_id}</span>
        <button class="btn small" data-log="${j.id}">log</button>
        ${j.status === "running" ? `<button class="btn small danger" data-stop="${j.id}">stop</button>` : ""}
        <button class="btn ghost small" data-deljob="${j.id}" title="Delete job">✕</button>
      </div>`).join("") || `<div class="empty">No training jobs yet.</div>`;
    $$("[data-log]").forEach(b => b.onclick = () => { watching = +b.dataset.log; pollLog(); });
    $$("[data-stop]").forEach(b => b.onclick = async () => {
      try { await API.stopJob(p.id, +b.dataset.stop); renderJobs(); } catch (e) { err(e); }
    });
    $$("[data-deljob]").forEach(b => b.onclick = async () => {
      const jid = +b.dataset.deljob;
      if (!confirm(`Delete job #${jid}? Its log is removed too (trained weights, if any, are kept).`)) return;
      try {
        await API.deleteJob(p.id, jid);
        if (watching === jid) {
          watching = null;
          $("#jobLog").style.display = "none";
          $("#jobLogHead").classList.remove("show");
        }
        renderJobs();
      } catch (e) { err(e); }
    });
  }
  async function pollLog() {
    if (!watching) return;
    try {
      const r = await API.jobLog(p.id, watching, expanded ? 2000 : 200);
      const box = $("#jobLog");
      // only follow the tail if the user hasn't scrolled away from the bottom —
      // otherwise every 3.5s refresh yanks them back down mid-read
      const wasAtBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
      box.style.display = "block";
      $("#jobLogHead").classList.add("show");
      $("#jobLogTitle").textContent = `job #${watching} · ${r.status}` +
        (wasAtBottom ? "" : " · scrolled up — jump to bottom to resume auto-follow");
      box.textContent = r.log || "(no output yet)";
      if (wasAtBottom) box.scrollTop = box.scrollHeight;
    } catch (_) {}
  }
  $("#jobLogExpand").onclick = () => {
    expanded = !expanded;
    $("#jobLog").classList.toggle("expanded", expanded);
    $("#jobLogExpand").textContent = expanded ? "⤡ Collapse" : "⤢ Expand";
    pollLog();
  };
  await renderJobs();
  S.jobPoll = setInterval(() => { renderJobs(); pollLog(); }, 3500);
}
