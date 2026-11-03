/* YAAP application — hash-routed views over the API. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const S = { project: null, images: [], curImage: -1, editor: null, saveTimer: null,
            activeClassId: null, jobPoll: null, verPoll: null,
            // one-shot hints consumed by the next viewAnnotate() call, set by whoever
            // navigates there (e.g. the Review grid) to control which image opens first,
            // which QC section's images to confine Prev/Next + the counter to, and
            // whether to flag already-annotated images
            pendingImageId: null, annotateScope: null, warnIfAnnotated: false,
            // one-shot: confine the next Quick Review to these image ids (a Bootstrap round)
            reviewOnly: null,
            // set by viewAnnotate() while it's the active view; lets the Home button
            // (and reaching the last image) run the same save+push-to-review confirm
            // as the editor's own Exit button, instead of just jumping away silently
            onLeaveAnnotate: null };

/* ── plumbing ─────────────────────────────────────────────────────── */
function toast(msg, kind = "", ms = 4200) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`; el.textContent = msg;
  $("#toastHost").append(el);
  setTimeout(() => el.remove(), ms);
}
const err = e => toast(e.message || String(e), "err");

function nav(view) { location.hash = S.project ? `#/p/${S.project.id}/${view}` : "#/"; }

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", () => {
  $("#brand").onclick = () => { location.hash = "#/"; };
  // the rail's "Home" button (data-view="review") doubles as the project
  // dashboard — while the annotator is open it owns leaving (saves, then may
  // confirm pushing the current image to review) before landing there; every
  // other rail button navigates straight to its view as usual.
  $$(".rail-btn[data-view]").forEach(b => b.onclick = () => {
    if (b.dataset.view === "review" && S.onLeaveAnnotate) return S.onLeaveAnnotate();
    nav(b.dataset.view);
  });
  $("#homeBtn").onclick = () => { location.hash = "#/"; };
  $("#viewerClose").onclick = closeViewer;
  // click on the dark area around the content closes it, like most lightboxes
  $("#viewer").onclick = e => { if (e.target.id === "viewer" || e.target.id === "viewerBody") closeViewer(); };
  // Esc closes natively; "close" fires asynchronously, so skip the wipe if
  // another viewer has opened in the meantime
  $("#viewer").addEventListener("close", () => { if (!$("#viewer").open) $("#viewerBody").innerHTML = ""; });
  loadDevice();
  route();
  fetch("/api/system/here", { method: "POST" }).catch(() => {});
});
// tells launch.sh's host-side watcher we're actually leaving (tab closed,
// navigated away, or reloading) — not just backgrounded — so it can free the
// container's resources after a grace period if we don't come back
window.addEventListener("pagehide", () => {
  navigator.sendBeacon && navigator.sendBeacon("/api/system/leaving");
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
  S.onLeaveAnnotate = null;   // stale hook would otherwise fire against a view that's no longer live
  if ($("#viewer")?.open) closeViewer();
  // a pending autosave still runs — destroy() only drops the editor's listeners
  S.editor?.destroy(); S.editor = null;
  window.onkeydown = null;   // Annotate's ←/→ image navigation
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
    <span class="chip mono" title="labeled = done or checked by a person; predicted = model output nobody has reviewed yet">${S.project.image_count} imgs · ${S.project.annotated_count} labeled${S.project.predicted_count ? ` · ${S.project.predicted_count} predicted` : ""}</span>`;
  ({ upload: viewUpload, annotate: viewAnnotate, review: viewReview, quickreview: viewQuickReview,
     classes: viewClasses, autolabel: viewAutolabel,
     generate: viewGenerate, versions: viewVersions, train: viewTrain,
     testmodel: viewTestModel }[view] || viewReview)();
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
            ${p.predicted_count ? `<span title="model output nobody has reviewed yet"><b>${p.predicted_count}</b> predicted</span>` : ""}
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
      const name = $("h2", c).textContent;
      if (!confirm(`Delete project "${name}"? Its images, labels, dataset versions, training runs and model ` +
                   "weights are deleted from disk. This can't be undone.")) return;
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

/* full-screen viewer: one <dialog> reused for images, annotated cards and logs */
function openViewer(title, html, { hint = "", bodyClass = "" } = {}) {
  $("#viewerTitle").textContent = title;
  $("#viewerHint").textContent = hint;
  const body = $("#viewerBody");
  body.className = `viewer-body ${bodyClass}`;
  body.innerHTML = html;
  $("#viewer").showModal();
}
function closeViewer() {
  if ($("#viewer").open) $("#viewer").close();
  $("#viewerBody").innerHTML = "";
}
function openImageViewer(title, src) {
  openViewer(title, `<img class="viewer-img" src="${src}" alt="${esc(title)}">`,
    { hint: "click the image to toggle actual size" });
  const img = $("#viewerBody .viewer-img");
  img.onclick = () => { img.classList.toggle("actual"); $("#viewerBody").classList.toggle("scroll"); };
}
// click any .al-frame inside `host` → that image with its overlay, full screen,
// carrying over the grid's hide-boxes / hide-labels state
function enableFrameViewer(host) {
  host.addEventListener("click", e => {
    const frame = e.target.closest(".al-frame");
    if (!frame || !host.contains(frame)) return;
    const name = frame.closest(".al-card")?.querySelector(".al-meta span")?.textContent || "";
    const toggles = ["hide-boxes", "hide-labels"].filter(c => host.classList.contains(c)).join(" ");
    openViewer(name, `<div class="viewer-frame">${frame.innerHTML}</div>`, { bodyClass: toggles });
  });
}

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
      toast(`✓ Uploaded ${res.created.length} image(s)` +
            (res.skipped.length ? `, skipped ${res.skipped.length}` : ""), "ok", 8000);
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
    const btn = $("#impGo"), label = btn.textContent;
    btn.disabled = true; btn.textContent = "Importing…";
    try {
      const res = await API.importDataset(p.id, f);
      toast(`✓ Imported ${res.images_created} image(s)` +
            (res.images_skipped ? `, skipped ${res.images_skipped}` : ""), "ok", 8000);
      route();
    } catch (e) { err(e); }
    btn.disabled = false; btn.textContent = label;
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
// "review" = graduated from auto-label into the main QC queue; "needs_reannotation"
// = rejected by a human, a distinct status so it never overlaps with "To review".
// Fixing + saving a rejected image flips it back to "annotated" (annotations.py),
// so it reappears under "To review" on its own.
const QC_FILTERS = [
  ["unannotated", "Unannotated"], ["reviewable", "To review"],
  ["auto-annotated", "Auto-annotated"], ["needs_reannotation", "Needs re-annotation"],
  ["approved", "Approved"],
];
const qcMatch = (im, filter) => filter === "proposals" ? im.proposal_count > 0 : filter === "reviewable"
  ? (im.status === "annotated" || im.status === "auto-annotated" || im.status === "empty" || im.status === "review")
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
        wrong label or box, approve it once it checks out, reject it to flag it for re-annotation, or
        erase its annotations entirely (🗑) to send it back to unannotated.</p>
      <div class="row" style="margin-bottom:16px" id="qcFilters">
        ${QC_FILTERS.map(([k, label]) => `
          <button class="btn small ${k === S.qcFilter ? "primary" : "ghost"}" data-f="${k}">${label} <span class="mono">${counts[k] ?? 0}</span></button>`).join("")}
      </div>
      <div class="img-grid" id="qcGrid"></div>
    </div>`;

    $$("#qcFilters [data-f]").forEach(b => b.onclick = () => {
      S.qcFilter = b.dataset.f;
      // both of these are single-image browsers, not grids — jump straight in
      if (b.dataset.f === "reviewable") { startQuickReview(null, "review"); return; }
      if (b.dataset.f === "approved") { startQuickReview(null, "approved"); return; }
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
          <button class="btn small ${im.status === "needs_reannotation" ? "danger" : ""}" data-reject="${im.id}"
            title="${im.status === "needs_reannotation" ? "Flagged for re-annotation — clear flag" : "Reject — flag for re-annotation"}">✕</button>
          <button class="btn small ${im.status === "approved" ? "primary" : ""}" data-approve="${im.id}"
            title="${im.status === "approved" ? "Un-approve" : "Approve"}">✓</button>
          <button class="btn small" data-clear="${im.id}" ${im.annotation_count ? "" : "disabled"}
            title="Erase annotations — back to unannotated">🗑</button>
        </div>
        <span class="nm">${esc(im.filename)} · ${im.annotation_count} ann.</span>
      </div>`).join("");

    // already-annotated sections open the image in Quick Review; the ✎ button
    // is the explicit way into the editor from there
    $$(".qc-tile", host).forEach(t => t.onclick = e => {
      if (e.target.closest("button")) return;
      if (S.qcFilter === "reviewable") startQuickReview(+t.dataset.id, "review");
      else if (S.qcFilter === "approved") startQuickReview(+t.dataset.id, "approved");
      else openReviewImage(+t.dataset.id);
    });
    $$("[data-open]", host).forEach(b => b.onclick = () => openReviewImage(+b.dataset.open));
    $$("[data-approve]", host).forEach(b => b.onclick = async () => {
      const im = images.find(x => x.id === +b.dataset.approve);
      const next = im.status === "approved" ? "annotated" : "approved";
      try { im.status = (await API.patchImage(p.id, im.id, { status: next })).status; render(); }
      catch (e) { err(e); }
    });
    $$("[data-clear]", host).forEach(b => b.onclick = async () => {
      const im = images.find(x => x.id === +b.dataset.clear);
      if (!confirm(`Erase all ${im.annotation_count} annotation(s) on "${im.filename}" and reset it to unannotated?`)) return;
      try { await API.saveAnnotations(p.id, im.id, []); }
      catch (e) { return err(e); }
      im.status = "unannotated"; im.annotation_count = 0;
      render();
      toast("Annotations cleared.", "ok");
    });
    $$("[data-reject]", host).forEach(b => b.onclick = async () => {
      const im = images.find(x => x.id === +b.dataset.reject);
      const next = im.status === "needs_reannotation" ? "annotated" : "needs_reannotation";
      try { im.status = (await API.patchImage(p.id, im.id, { status: next })).status; render(); }
      catch (e) { err(e); }
    });
  }

  function openReviewImage(imgId) {
    // viewAnnotate() re-fetches images itself, so hand it a hint instead of
    // pre-setting S.images/S.curImage directly (that assignment would just get
    // overwritten by the fresh fetch before it's ever read).
    S.pendingImageId = imgId;
    S.annotateScope = S.qcFilter;   // confine Prev/Next + the counter to this section's images
    S.warnIfAnnotated = S.qcFilter === "unannotated";
    nav("annotate");
  }

  render();
}

// The pixel box an annotation covers (polygons by their bounding box), and box overlap.
function annBox(a) {
  if (a.kind === "bbox") return a.data;
  if (a.kind !== "polygon" || !a.data.points?.length) return null;
  const xs = a.data.points.map(q => q[0]), ys = a.data.points.map(q => q[1]);
  const x = Math.min(...xs), y = Math.min(...ys);
  return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}
// Overlap of two outlines by area (rasterized) — two triangles splitting one square share a box,
// not an object.
function polyIou(p, q) {
  const all = p.concat(q), xs = all.map(v => v[0]), ys = all.map(v => v[1]);
  const x0 = Math.min(...xs), y0 = Math.min(...ys);
  const s = 128 / Math.max(Math.max(...xs) - x0, Math.max(...ys) - y0, 1);
  const W = Math.ceil((Math.max(...xs) - x0) * s) + 2, H = Math.ceil((Math.max(...ys) - y0) * s) + 2;
  const fill = poly => {
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const g = c.getContext("2d");
    g.beginPath();
    poly.forEach(([x, y], i) => i ? g.lineTo((x - x0) * s, (y - y0) * s) : g.moveTo((x - x0) * s, (y - y0) * s));
    g.closePath(); g.fill();
    return g.getImageData(0, 0, W, H).data;
  };
  const A = fill(p), B = fill(q);
  let inter = 0, union = 0;
  for (let i = 3; i < A.length; i += 4) {
    const a = A[i] > 127, b = B[i] > 127;
    if (a && b) inter++;
    if (a || b) union++;
  }
  return union ? inter / union : 0;
}
function boxIou(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const u = a.w * a.h + b.w * b.h - ix * iy;
  return u > 0 ? ix * iy / u : 0;
}

// Shared by Quick Review and the auto-label preview grid: annotations -> SVG shapes + labels.
function annotationShapesSvg(anns, classes, imgWidth) {
  const strokeW = Math.max(2, Math.round(imgWidth / 350));
  const fontSize = Math.min(42, Math.max(14, Math.round(imgWidth / 40)));
  return anns.map(a => {
    const c = classes.find(x => x.id === a.class_id);
    const color = c ? c.color : (a.color || "#999");
    const name = (c ? c.name : (a.class_name || "?")) + (a.source === "model" ? ` ·${Math.round(a.confidence * 100)}%` : "");
    let shape, lx, ly;
    if (a.kind === "bbox") {
      const d = a.data;
      shape = `<rect x="${d.x}" y="${d.y}" width="${d.w}" height="${d.h}" fill="${color}33" stroke="${color}" stroke-width="${strokeW}"/>`;
      lx = d.x; ly = d.y;
    } else if (a.kind === "polygon") {
      const pts = a.data.points.map(pt => pt.join(",")).join(" ");
      shape = `<polygon points="${pts}" fill="${color}33" stroke="${color}" stroke-width="${strokeW}"/>`;
      [lx, ly] = a.data.points[0];
    } else if (a.kind === "classification") {
      shape = ""; lx = 0; ly = 0;                   // whole-image class: just the tag, top-left
    } else return "";
    // label tag — clamped so it never sits above the image's own top edge
    const labelY = Math.max(fontSize + 4, ly);
    const labelW = Math.max(name.length * fontSize * 0.62, fontSize * 1.6);
    const label = `<rect x="${lx}" y="${labelY - fontSize - 4}" width="${labelW}" height="${fontSize + 6}" fill="${color}"/>
      <text x="${lx + 4}" y="${labelY - 5}" font-size="${fontSize}" font-family="monospace"
        font-weight="600" fill="#0a0a0a">${esc(name)}</text>`;
    return `<g class="ann-box">${shape}</g><g class="ann-label">${label}</g>`;
  }).join("");
}

function startQuickReview(imgId, mode = "review") {
  S.pendingImageId = imgId ?? null;
  S.reviewMode = mode;
  nav("quickreview");
}

/* ── quick review: a single-image browser with two modes.
   "review"   — accept/reject, auto-advances, drains the "To review" queue.
   "approved" — prev/next arrows to browse, plus a revert button; nothing
                auto-advances except a revert (which removes that image from
                the approved set, so it makes sense to move on). Both share
                zoom/pan (wheel or +/− buttons) on the image itself. ── */
async function viewQuickReview() {
  const p = S.project;
  const mode = S.reviewMode || "review";
  S.reviewMode = null;
  let images;
  try { images = await API.images(p.id); } catch (e) { return err(e); }

  let queue = images.filter(im => qcMatch(im, mode === "approved" ? "approved" : "reviewable"));
  if (S.reviewOnly) {
    const only = new Set(S.reviewOnly);
    queue = queue.filter(im => only.has(im.id));
    S.reviewOnly = null;
  }
  if (S.pendingImageId != null) {
    const i = queue.findIndex(x => x.id === S.pendingImageId);
    if (i > 0) queue = queue.slice(i);
    S.pendingImageId = null;
  }
  if (!queue.length) {
    toast(mode === "approved" ? "No approved images yet." : "Nothing left to review.", "ok");
    nav("review");
    return;
  }

  let idx = 0, zoom = 1, shownAnnIds = [];

  function onKey(e) {
    if (e.target.matches("input,textarea,select")) return;
    if (mode === "review") {
      const k = e.key.toLowerCase();
      if (k === "a") $("#qrAccept")?.click();
      else if (k === "r") $("#qrReject")?.click();
    } else {
      if (e.key === "ArrowLeft") $("#qrPrev")?.click();
      else if (e.key === "ArrowRight") $("#qrNext")?.click();
    }
  }
  window.addEventListener("keydown", onKey);
  window.addEventListener("hashchange", function cleanup() {
    window.removeEventListener("keydown", onKey);
    window.removeEventListener("hashchange", cleanup);
  });

  function setZoom(z) {
    zoom = Math.min(4, Math.max(0.5, Math.round(z * 100) / 100));
    const svg = $(".qr-frame svg");
    if (svg) svg.style.transform = `scale(${zoom})`;
    const lvl = $("#qrZoomLevel"); if (lvl) lvl.textContent = Math.round(zoom * 100) + "%";
  }

  async function renderCurrent() {
    const im = queue[idx];
    let anns = [];
    try { anns = await API.annotations(p.id, im.id); } catch (e) { /* show image without overlays */ }
    shownAnnIds = anns.map(a => a.id);

    const shapes = annotationShapesSvg(anns, p.classes, im.width);
    zoom = 1;

    const actionsHtml = mode === "review" ? `
      <div class="row" style="margin-top:18px;justify-content:center;gap:16px">
        <button class="btn danger qr-btn" id="qrReject">✕ Reject <span class="kbd">R</span></button>
        <button class="btn primary qr-btn" id="qrAccept">✓ Accept <span class="kbd">A</span></button>
      </div>` : `
      <div class="row" style="margin-top:18px;justify-content:center;gap:16px">
        <button class="btn qr-btn" id="qrPrev" ${idx === 0 ? "disabled" : ""} title="Previous (←)">← Prev</button>
        <button class="btn danger qr-btn" id="qrRevert">↺ Revert to Needs re-annotation</button>
        <button class="btn qr-btn" id="qrNext" ${idx === queue.length - 1 ? "disabled" : ""} title="Next (→)">Next →</button>
      </div>`;

    $("#stage").innerHTML = `<div class="stage-pad" style="max-width:900px">
      <h1 class="page">${mode === "approved" ? "Approved images" : "Review queue"}</h1>
      <p class="sub">${idx + 1} of ${queue.length} · ${esc(im.filename)} · ${anns.length} annotation(s)</p>
      <div class="qr-frame">
        <svg viewBox="0 0 ${im.width} ${im.height}" preserveAspectRatio="xMidYMid meet">
          <image href="${API.imageUrl(p.id, im.id)}" width="${im.width}" height="${im.height}"></image>
          ${shapes}
        </svg>
      </div>
      <div class="row" style="justify-content:center;align-items:center;gap:6px;margin-top:10px">
        <button class="btn small" id="qrZoomOut" title="Zoom out">−</button>
        <span class="mono" id="qrZoomLevel" style="min-width:42px;text-align:center">100%</span>
        <button class="btn small" id="qrZoomIn" title="Zoom in">+</button>
        <button class="btn small ghost" id="qrZoomReset" title="Reset zoom">Fit</button>
      </div>
      ${actionsHtml}
    </div>`;

    $("#qrZoomIn").onclick = () => setZoom(zoom + 0.25);
    $("#qrZoomOut").onclick = () => setZoom(zoom - 0.25);
    $("#qrZoomReset").onclick = () => setZoom(1);
    $(".qr-frame").addEventListener("wheel", e => {
      e.preventDefault();
      setZoom(zoom + (e.deltaY < 0 ? 0.15 : -0.15));
    }, { passive: false });

    if (mode === "review") {
      $("#qrAccept").onclick = () => decide("approved");
      $("#qrReject").onclick = () => decide("needs_reannotation");
    } else {
      $("#qrPrev").onclick = () => { if (idx > 0) { idx--; renderCurrent(); } };
      $("#qrNext").onclick = () => { if (idx < queue.length - 1) { idx++; renderCurrent(); } };
      $("#qrRevert").onclick = () => revert();
    }
  }

  async function decide(status) {
    const im = queue[idx];
    // Accept approves what was shown — a Bootstrap round may have replaced it meanwhile
    const body = status === "approved" ? { status, expect_annotations: shownAnnIds } : { status };
    try { await API.patchImage(p.id, im.id, body); }
    catch (e) {
      if (e.status === 409) { toast(e.message); return renderCurrent(); }
      return err(e);
    }
    idx++;
    if (idx >= queue.length) {
      toast("Review queue complete!", "ok");
      nav("review");
      return;
    }
    renderCurrent();
  }

  async function revert() {
    const im = queue[idx];
    try { await API.patchImage(p.id, im.id, { status: "needs_reannotation" }); } catch (e) { return err(e); }
    toast(`"${im.filename}" sent back to Needs re-annotation.`, "ok");
    queue.splice(idx, 1);
    if (!queue.length) {
      toast("No more approved images.", "ok");
      nav("review");
      return;
    }
    if (idx >= queue.length) idx = queue.length - 1;
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
    <div id="clsList"></div>
    <div class="card" id="clsProposals" style="margin-top:16px" hidden></div></div>`;
  let waiting = 0;                       // images with SAM-outlined objects that still need a class
  API.images(p.id).then(ims => { waiting = ims.filter(im => im.proposal_count > 0).length; render(); }).catch(() => {});

  function render() {
    $("#clsList").innerHTML = S.project.classes.map((c, i) => `
      <div class="class-row" data-id="${c.id}">
        <input type="color" value="${c.color}" title="Class color">
        <span class="swatch" style="background:${c.color}"></span>
        <span class="nm" contenteditable spellcheck="false">${esc(c.name)}</span>
        <span class="idx">idx ${i} · key ${i < 9 ? i + 1 : "—"}</span>
        <button class="btn ghost small">✕</button>
      </div>`).join("") || `<div class="empty">No classes yet — add one above.</div>`;
    const box = $("#clsProposals");
    box.hidden = !(waiting && S.project.classes.length);
    if (!box.hidden) {
      box.innerHTML = `<div class="row" style="align-items:center;justify-content:space-between;gap:12px">
        <span class="sub" style="margin:0">SAM outlined objects in <b>${waiting}</b> image(s) — they're waiting for a class.</span>
        <button class="btn primary" id="clsLabelFound">Label them →</button></div>`;
      $("#clsLabelFound").onclick = () => { S.annotateScope = "proposals"; nav("annotate"); };
    }
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
  if (S.annotateScope) {
    // confine Prev/Next and the "N/total" counter to just this QC section's
    // images — a snapshot taken now, not re-filtered as statuses change
    // during the session (so the count doesn't shift under you mid-edit)
    const scoped = S.images.filter(im => qcMatch(im, S.annotateScope));
    if (!scoped.length) {
      S.annotateScope = null;
      $("#stage").innerHTML = `<div class="stage-pad"><div class="empty">
        No images to annotate. <br><br>
        <button class="btn primary" onclick="location.hash='#/'">Return to Home</button></div></div>`;
      return;
    }
    S.images = scoped;
  }
  S.annotateScope = null;   // one-shot — only applies to this entry, not future navigation

  // active learning: ranked images exist → offer to walk them most-uncertain first
  // (unlabeled images only; everything else keeps its order, after them)
  const hasScores = S.images.some(im => im.uncertainty != null);
  let uncOn = false;
  try { uncOn = hasScores && localStorage.getItem("yaap.uncertainFirst") === "1"; } catch (_) {}
  if (uncOn) {
    const ranked = S.images.filter(im => im.uncertainty != null && !im.annotation_count)
      .sort((a, b) => b.uncertainty - a.uncertainty);
    const rest = S.images.filter(im => !ranked.includes(im));
    S.images = [...ranked, ...rest];
    if (S.pendingImageId == null) S.curImage = 0;     // the top of the ranking is always the next one to label
  }

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
  // class-by-class labeling state of a "find all" result (see startWalk) — declared before the
  // Editor is built: its constructor already calls back into updateSamStrip(), which reads it
  let walk = null;
  $("#stage").innerHTML = `
  <div class="editor">
    <div class="ed-canvas-wrap">
      <canvas id="edCanvas"></canvas>
      ${isCls ? "" : `
      <div class="ed-toolbar">
        <button class="tool-btn" data-tool="select" title="Select / edit (V)">➤</button>
        <button class="tool-btn active" data-tool="bbox" title="Bounding box (B)">▭</button>
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
        <button class="btn small" id="samFindAll" hidden title="Find every object like the boxed one in this image (E) — ctrl-click a match to exclude it">E</button>
        <label class="sam-simplify" id="samConfWrap" hidden title="Minimum match score">min score
          <input class="input" id="samConf" type="number" min="5" max="95" step="5" value="25" style="width:56px;display:inline-block">%</label>
        <button class="btn small primary" id="samAccept" hidden>✓ Accept <span class="kbd">↵</span></button>
        <button class="btn small" id="samDiscard" hidden>✕ <span class="kbd">Esc</span></button>
      </div>
      <div class="sam-label-pick" id="samLabelPick" hidden></div>`}
      <div class="ed-topstrip">
        <span class="fn mono" id="edFn"></span>
        <span class="savestate" id="saveState">saved</span>
        <span class="chip warn" id="edAnnotatedWarn" hidden>⚠ already annotated</span>
        <button class="btn small primary" id="edProposals" hidden title="This image has objects SAM outlined that still need a class">⌖ Label found objects</button>
        <label class="chip" id="edUncWrap" hidden title="Order images by model uncertainty (rank them on the Auto-label page)">
          <input type="checkbox" id="edUnc"> Most uncertain first</label>
        <button class="btn small" id="edReject" title="Flag this image as needing re-work">✕ Reject</button>
        <button class="btn small" id="edApprove" title="Mark this image as QC-approved">✓ Approve</button>
        <button class="btn small ghost" id="edExit" title="Save and exit the annotator">⏏ Exit</button>
      </div>
      <div class="ed-hud" id="edHud">—</div>
    </div>
    <aside class="ed-side">
      ${isCls ? `<h4>Label this image</h4><div class="cls-panel" id="clsPanel"></div>`
              : `<h4>Classes <span class="kbd">1–9</span></h4><div class="ed-classes" id="edClasses"></div>
                 <h4>Annotations</h4><div class="ed-anns" id="edAnns"></div>
                 <button class="btn small ghost" id="edMarkEmpty" style="width:100%;margin-top:8px"
                   title="No objects in this image — mark it empty so it stops showing as unannotated">∅ Mark empty</button>`}
      <div class="ed-nav">
        <button class="btn" id="prevImg" title="Previous (←)">←</button>
        <span class="chip mono" id="imgPos" style="align-self:center"></span>
        <button class="btn" id="nextImg" title="Next (→)">→</button>
      </div>
    </aside>
  </div>`;

  S.editor?.destroy();
  const editor = S.editor = new Editor($("#edCanvas"), {
    classes: () => S.project.classes,
    getActiveClass: () => S.project.classes.find(c => c.id === S.activeClassId),
    onChange: () => { markDirty(); renderAnnList(); },
    onSelect: () => renderAnnList(),
    onToolKey: setTool,
    onClassKey: i => { const c = p.classes[i]; if (c) { S.activeClassId = c.id; renderSide(); } },
    onSamPrompt: runSamPrompt,
    onSamAccept: acceptSam,
    onSamFindAll: () => findAllLikeThis(),
    onSamToggle: ed => { updateSamStrip(ed); samHint(`${ed.samPending.filter(r => !r.off).length}/${ed.samPending.length} match(es) kept — ctrl-click to toggle, then Accept`); },
    onSamState: updateSamStrip,
    onPendingShape: renderLabelPick,
  });
  // through the local setTool so the toolbar highlight matches the tool
  if (p.task_type === "segment") setTool("polygon");
  if (isCls) setTool("pan");

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
      else {                       // load SAM in the background so the first click / E isn't slow
        samHint("warming up SAM…");
        API.samWarmup({ model_name: $("#samModel").value })
          .then(() => samHint("click ＋ · shift-click − · drag box"))
          .catch(() => samHint("click ＋ · shift-click − · drag box"));
      }
      samReady = true;
    } catch (e) { err(e); }
  }
  function samHint(t) { const h = $("#samHint"); if (h) h.textContent = t; }
  function updateSamStrip(ed) {
    const has = ed.samPending.length > 0;
    if (!has) walk = null;
    const a = $("#samAccept"), d = $("#samDiscard");
    if (a) { a.hidden = !has; d.hidden = !has && !ed.samPts.length && !ed.samBox; }
    if ($("#samFindAll")) { $("#samFindAll").hidden = !ed.samBox; $("#samConfWrap").hidden = !ed.samBox; }
    if (!has) samHint("click ＋ · shift-click − · drag box");
    renderLabelPick();
  }

  /* small class picker that pops up right next to a fresh SAM mask, or a
     freshly-drawn manual bbox/polygon (editor.pendingShape) — click a class
     to label + finish it in one step, no sidebar detour either way. */
  /* class-by-class walkthrough of a "find all" result: each match is highlighted in turn,
     you pick its class (click or 1–9) or delete it if it's wrong; Done keeps what's labeled. */
  function startWalk(fromProposals = false) {
    if (!editor.samPending.length) return;
    if (!p.classes.length) return toast("Add at least one class first.", "err");
    editor.samPending.forEach(r => { r.cls = null; r.cur = false; r.off = false; delete r.color; delete r.label; });
    walk = { i: 0, proposals: fromProposals };
    focusWalk(0);
  }
  function walkLabeled() { return editor.samPending.filter(r => r.cls); }
  function focusWalk(i) {
    const pend = editor.samPending;
    walk.i = Math.max(0, Math.min(pend.length - 1, i));
    pend.forEach((r, k) => { r.cur = k === walk.i; });
    samHint(`label box ${walk.i + 1}/${pend.length} — pick a class (1–9) · Del removes the box`);
    renderLabelPick();
    revealBox(pend[walk.i].bbox);
    editor.draw();
  }
  // pan (never zoom) so the whole box is on screen and not under the fixed class panel (bottom-right)
  function revealBox(b) {
    const cw = editor.cv.clientWidth, ch = editor.cv.clientHeight, v = editor.view, m = 30;
    const panel = $("#samLabelPick");
    const pl = cw - (panel ? panel.offsetWidth : 0) - 24, pt = ch - (panel ? panel.offsetHeight : 0) - 24;
    let [x0, y0] = editor.toScr(b.x, b.y), [x1, y1] = editor.toScr(b.x + b.w, b.y + b.h);
    let dx = 0, dy = 0;
    if (x0 < m) dx = m - x0; else if (x1 > cw - m) dx = Math.max(cw - m - x1, m - x0);
    if (y0 < m + 20) dy = m + 20 - y0; else if (y1 > ch - m) dy = Math.max(ch - m - y1, m + 20 - y0);
    x0 += dx; x1 += dx; y0 += dy; y1 += dy;
    if (x1 > pl && y1 > pt) {                       // overlaps the panel → slide left if room, else up
      if (x0 - (x1 - pl) - 10 >= m) dx -= x1 - pl + 10; else dy -= y1 - pt + 10;
    }
    if (dx || dy) { v.tx += dx; v.ty += dy; }
  }
  function walkNextUnlabeled(from) {
    const pend = editor.samPending, n = pend.length;
    for (let k = 1; k <= n; k++) { const j = (from + k) % n; if (!pend[j].cls) return j; }
    return -1;
  }
  function walkAssign(cid) {
    const r = editor.samPending[walk.i], c = p.classes.find(x => x.id === cid);
    if (!r || !c) return;
    r.cls = cid; r.color = c.color; r.label = c.name;
    const nx = walkNextUnlabeled(walk.i);
    if (nx < 0) return finishWalk();
    focusWalk(nx);
  }
  function walkDelete() {
    editor.samPending.splice(walk.i, 1);
    if (!editor.samPending.length) { walk = null; editor.clearSam(); return; }
    const nx = walkNextUnlabeled(walk.i - 1);
    if (nx < 0) return finishWalk();
    focusWalk(nx);
  }
  function finishWalk() {
    if (!walk) return;                      // already finished (the last class pick ends it)
    const done = walkLabeled(), fromProposals = walk.proposals;
    for (const r of done) pushSamAnn(r, p.classes.find(c => c.id === r.cls));
    const dropped = editor.samPending.length - done.length;
    toast(`Added ${done.length} labeled ${p.task_type === "detect" ? "box(es)" : "polygon(s)"}` +
          (dropped ? ` · ${dropped} unlabeled left out` : ""), "ok");
    walk = null;
    editor.clearSam(false);
    markDirty(); renderAnnList(); editor.draw();
    if (fromProposals) afterProposalWalk();
  }
  function cancelWalk() {
    const n = walkLabeled().length;
    if (n && !confirm(`Discard ${n} labeled box(es)?`)) return;
    walk = null; editor.clearSam();
  }
  function renderWalkPanel(box) {
    const pend = editor.samPending, r = pend[walk.i];
    const b = r.bbox;
    box.className = "sam-label-pick walk";
    box.innerHTML = `
      <div class="sam-walk-head">Box ${walk.i + 1} / ${pend.length} · ${Math.round(r.score * 100)}% · ${walkLabeled().length} labeled</div>
      ${p.classes.map((c, i) => `<button class="sam-label-chip" data-id="${c.id}">
        <span class="swatch" style="background:${c.color}"></span>${esc(c.name)}${i < 9 ? ` <span class="kbd">${i + 1}</span>` : ""}</button>`).join("")}
      <div class="sam-walk-actions">
        <button class="btn small danger" id="walkDel" title="This box is wrong — erase it (Del / X)">🗑 Erase box</button>
        <button class="btn small" id="walkPrev" title="Previous (←)">←</button>
        <button class="btn small" id="walkNext" title="Skip to next (→)">Skip →</button>
        <button class="btn small primary" id="walkDone" title="Keep the labeled boxes and finish (Enter)">✓ Done</button>
        <button class="btn small ghost" id="walkCancel" title="Discard all (Esc)">✕</button>
      </div>`;
    $$(".sam-label-chip", box).forEach(b2 => b2.onclick = () => walkAssign(+b2.dataset.id));
    $("#walkDel").onclick = walkDelete;
    $("#walkPrev").onclick = () => focusWalk(walk.i - 1);
    $("#walkNext").onclick = () => focusWalk(walk.i + 1);
    $("#walkDone").onclick = finishWalk;
    $("#walkCancel").onclick = cancelWalk;
    // fixed spot (bottom-right of the canvas, see .sam-label-pick.walk) — always the same place, never over the box
    box.style.left = box.style.top = box.style.maxHeight = box.style.maxWidth = "";
    box.hidden = false;
    box.scrollTop = 0;
  }
  function walkKey(e) {
    if (walk && !editor.samPending.length) walk = null;      // image changed / cleared underneath us
    if (!walk || e.target.matches("input,textarea,select,[contenteditable]") || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    let handled = true;
    if (/^[1-9]$/.test(k)) { const c = p.classes[+k - 1]; if (c) walkAssign(c.id); }
    else if (k === "delete" || k === "backspace" || k === "x") walkDelete();
    else if (e.key === "ArrowRight" || e.key === "Tab" && !e.shiftKey) focusWalk(walk.i + 1);
    else if (e.key === "ArrowLeft" || e.key === "Tab" && e.shiftKey) focusWalk(walk.i - 1);
    else if (e.key === "Enter") finishWalk();
    else if (e.key === "Escape") cancelWalk();
    else handled = false;
    if (handled) { e.preventDefault(); e.stopPropagation(); }
  }
  window.addEventListener("keydown", walkKey, true);
  window.addEventListener("hashchange", function stopWalkKeys() {
    window.removeEventListener("keydown", walkKey, true);
    window.removeEventListener("hashchange", stopWalkKeys);
  });

  function renderLabelPick() {
    const box = $("#samLabelPick");
    if (!box) return;
    if (walk && editor.samPending.length) return renderWalkPanel(box);
    box.className = "sam-label-pick";
    const pending = editor.pendingShape, mask = editor.samPending[0];
    if (!p.classes.length || (!pending && !mask)) { box.hidden = true; return; }
    box.innerHTML = p.classes.map(c => `
      <button class="sam-label-chip" data-id="${c.id}">
        <span class="swatch" style="background:${c.color}"></span>${esc(c.name)}</button>`).join("");
    if (pending) {
      $$(".sam-label-chip", box).forEach(b => b.onclick = () => { box.hidden = true; editor.assignPendingClass(+b.dataset.id); });
    } else {
      $$(".sam-label-chip", box).forEach(b => b.onclick = () => { box.hidden = true; acceptSam(+b.dataset.id); });
    }
    const [ix, iy] = pending ? pending.anchor : [mask.bbox.x + mask.bbox.w / 2, mask.bbox.y];
    const [sx, sy] = editor.toScr(ix, iy);
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
  async function findAllLikeThis() {
    const b = editor.samBox;
    if (!b || samBusy) return toast("Draw a box around one example first.", "err");
    await callSam({ type: "exemplar", box: [b.x0, b.y0, b.x1, b.y1] },
                  { conf: Math.min(0.95, Math.max(0.05, (+$("#samConf").value || 25) / 100)) });
  }
  $("#samFindAll") && ($("#samFindAll").onclick = () => findAllLikeThis());
  async function callSam(prompt, extra = {}) {
    const im = S.images[S.curImage];
    const exemplar = prompt.type === "exemplar";
    samBusy = true; samHint(exemplar ? "searching for similar objects…" : "segmenting…");
    try {
      const r = await API.samPredict(p.id, im.id,
        { model_name: $("#samModel").value, prompt, simplify: +$("#samSimplify").value, ...extra });
      let found = r.results || [], already = 0;
      if (exemplar) {                    // "find all like this": skip objects that already have a box
        const taken = editor.anns.filter(a => annBox(a));
        const same = (m, a) => {             // segment projects compare outlines, detect ones boxes
          const o = boxIou(m.bbox, annBox(a));
          return o > 0 && a.kind === "polygon" && m.polygon?.length ? polyIou(m.polygon, a.data.points) : o;
        };
        const fresh = found.filter(m => !taken.some(a => same(m, a) >= 0.7));
        already = found.length - fresh.length;
        found = fresh;
      }
      editor.samPending = found;
      const note = already ? ` · ${already} already labeled` : "";
      samHint(editor.samPending.length
        ? (exemplar ? `${editor.samPending.length} match(es)${note} — ctrl-click one to exclude, then Accept`
                    : `${editor.samPending.length} mask(s) — refine with more clicks, then Accept`)
        : already ? `every match is already labeled (${already})` : "no mask found — add clicks or adjust the box");
    } catch (e) { samHint(""); err(e); }
    samBusy = false;
    updateSamStrip(editor); editor.draw();
    if (exemplar && editor.samPending.length) startWalk();
  }
  $("#samAccept") && ($("#samAccept").onclick = () => acceptSam());
  $("#samDiscard") && ($("#samDiscard").onclick = () => editor.clearSam());
  function pushSamAnn(r, cls) {
    if (p.task_type === "detect")            // segmentation → detection format
      editor.anns.push({ class_id: cls.id, kind: "bbox", source: "model",
        confidence: r.score, data: { ...r.bbox } });
    else                                     // instance segmentation format
      editor.anns.push({ class_id: cls.id, kind: "polygon", source: "model",
        confidence: r.score, data: { points: r.polygon.map(q => [q[0], q[1]]) } });
  }
  function acceptSam(clsId) {
    if (walk) return finishWalk();
    const kept = editor.samPending.filter(r => !r.off);
    if (!kept.length) return;
    const cls = p.classes.find(c => c.id === (clsId ?? S.activeClassId));
    if (!cls) return toast("Pick a class first.", "err");
    for (const r of kept) pushSamAnn(r, cls);
    toast(`Added ${kept.length} ${p.task_type === "detect" ? "box(es)" : "polygon(s)"} as “${cls.name}”.`, "ok");
    editor.clearSam(false);
    markDirty(); renderAnnList(); editor.draw();
  }

  if (hasScores) {                  // "Most uncertain first" — needs a ranking (Auto-label → Active learning)
    $("#edUncWrap").hidden = false;
    $("#edUnc").checked = uncOn;
    $("#edUnc").onchange = async e => {
      try { localStorage.setItem("yaap.uncertainFirst", e.target.checked ? "1" : "0"); } catch (_) {}
      await save();
      // on: start at the top of the ranking; off: stay on the image you're on
      S.pendingImageId = e.target.checked ? null : S.images[S.curImage]?.id ?? null;
      S.curImage = 0;
      route();
    };
  }
  $("#prevImg").onclick = () => go(S.curImage - 1);
  $("#nextImg").onclick = () => go(S.curImage + 1);
  window.onkeydown = e => {
    if (e.target.matches("input,textarea,[contenteditable]")) return;
    if (e.key === "ArrowLeft") go(S.curImage - 1);
    if (e.key === "ArrowRight") go(S.curImage + 1);
  };

  let dirty = false;
  // the image the editor is showing — save() writes here, not to whatever
  // S.images[S.curImage] points at by the time the debounced save fires
  let editingImage = null;
  function markDirty() {
    dirty = true;
    const st = $("#saveState"); st.textContent = "unsaved"; st.className = "savestate dirty";
    clearTimeout(S.saveTimer);
    S.saveTimer = setTimeout(save, 700);
    renderMarkEmpty();   // drawing/deleting anything should reset the "empty" flag right away, not after the debounce
  }
  async function save() {
    if (!dirty) return;
    const im = editingImage;
    try {
      await API.saveAnnotations(p.id, im.id, editor.anns.map(a =>
        ({ class_id: a.class_id, kind: a.kind, data: a.data, source: a.source, confidence: a.confidence })));
      dirty = false;
      const st = $("#saveState");   // gone if the save finished after leaving the view
      if (st) { st.textContent = "saved"; st.className = "savestate saved"; }
      im.annotation_count = editor.anns.length;
      im.status = editor.anns.length ? "annotated" : "unannotated";
    } catch (e) { err(e); }
  }

  // Shared by the Exit button, the Home/brand nav, and running past the last
  // image with Next: save whatever's pending, then land on the project's Home
  // dashboard. No status change needed here — save() above already leaves a
  // finished image as "annotated", which is exactly the status that puts it
  // in "To review" (qcMatch). Setting it to "review" here would be wrong: that
  // status means "flagged for re-annotation" (the Reject button), so doing it
  // on a normal exit used to mislabel perfectly fine images as rejected.
  async function leaveAnnotator() {
    await save();
    nav("review");   // the project's Home dashboard, not the global projects list
  }
  S.onLeaveAnnotate = leaveAnnotator;
  $("#edExit").onclick = leaveAnnotator;

  async function go(i) {
    await save();
    if (i < 0) return;
    if (i >= S.images.length) { await leaveAnnotator(); return; }
    S.curImage = i;
    await loadImage();
  }

  async function loadImage() {
    const im = editingImage = S.images[S.curImage];
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
    renderApprove(); renderReject(); renderMarkEmpty();
    renderSide();
    renderProposalsBtn();
    if (im.proposal_count > 0 && !isCls) startProposalWalk();     // objects waiting for a class → ask right away
  }
  function renderProposalsBtn() {
    const b = $("#edProposals"), im = S.images[S.curImage];
    if (b && im) { b.hidden = !(im.proposal_count > 0); b.textContent = `⌖ Label found objects (${im.proposal_count})`; }
  }
  $("#edProposals") && ($("#edProposals").onclick = () => startProposalWalk());
  async function startProposalWalk() {
    const im = editingImage;
    let props = [];
    try { props = await API.proposals(p.id, im.id); } catch (e) { return err(e); }
    if (editingImage !== im || !props.length) return;
    editor.samPending = props.map(r => ({ polygon: r.polygon, bbox: r.bbox, score: r.score, label: "" }));
    startWalk(true);
  }
  // after a proposal walk: drop the leftover proposals and jump to the next image that still has some
  async function afterProposalWalk() {
    const im = editingImage;
    try { await API.clearProposals(p.id, im.id); } catch (e) { return err(e); }
    im.proposal_count = 0; renderProposalsBtn();
    const next = S.images.findIndex((x, i) => i > S.curImage && x.proposal_count > 0);
    if (next >= 0) await go(next);
    else toast("All found objects are labeled.", "ok");
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
      renderApprove(); renderReject(); renderMarkEmpty();
      toast(im.status === "approved" ? "Marked as approved." : "Approval cleared.", "ok");
    } catch (e) { err(e); }
  });

  function renderReject() {
    const im = S.images[S.curImage];
    const btn = $("#edReject"); if (!btn) return;
    btn.textContent = im.status === "needs_reannotation" ? "✕ Rejected" : "✕ Reject";
    btn.classList.toggle("danger", im.status === "needs_reannotation");
  }
  $("#edReject") && ($("#edReject").onclick = async () => {
    const im = S.images[S.curImage];
    try {
      const next = im.status === "needs_reannotation" ? "annotated" : "needs_reannotation";
      await API.patchImage(p.id, im.id, { status: next });
      im.status = next;
      renderApprove(); renderReject(); renderMarkEmpty();
      toast(im.status === "needs_reannotation" ? "Flagged for re-annotation." : "Rejection cleared.", "ok");
    } catch (e) { err(e); }
  });

  function renderMarkEmpty() {
    const im = S.images[S.curImage];
    const btn = $("#edMarkEmpty"); if (!btn) return;
    // live anns count, not just im.status — the moment something is drawn it's
    // no longer empty, even before the debounced save confirms it server-side
    const isEmpty = im.status === "empty" && editor.anns.length === 0;
    btn.textContent = isEmpty ? "∅ Marked empty — undo" : "∅ Mark empty";
    btn.classList.toggle("primary", isEmpty);
  }
  $("#edMarkEmpty") && ($("#edMarkEmpty").onclick = async () => {
    const im = S.images[S.curImage];
    if (im.status === "empty") {
      try { await API.patchImage(p.id, im.id, { status: "unannotated" }); }
      catch (e) { return err(e); }
      im.status = "unannotated";
      renderApprove(); renderReject(); renderMarkEmpty();
      toast("Empty flag cleared.", "ok");
      return;
    }
    if (editor.anns.length &&
        !confirm(`This image has ${editor.anns.length} annotation(s). Clear them and mark it empty (no objects)?`)) return;
    clearTimeout(S.saveTimer);
    editor.anns = [];
    editor.select(-1);
    editor.draw();
    try {
      await API.saveAnnotations(p.id, im.id, []);
      await API.patchImage(p.id, im.id, { status: "empty" });
    } catch (e) { return err(e); }
    dirty = false;
    const st = $("#saveState"); st.textContent = "saved"; st.className = "savestate saved";
    im.annotation_count = 0; im.status = "empty";
    renderApprove(); renderReject(); renderMarkEmpty(); renderAnnList();
    toast("Marked as empty — no objects to annotate.", "ok");
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
    <h3 class="sect">Bootstrap — label a few, YAAP labels the rest</h3>
    <div class="card" id="bsCard">
      <p class="sub" style="margin-top:0">Label about 10 images by hand. Bootstrap trains a small model on them
        (about a minute on a GPU) and pre-labels every image nobody has labeled yet. Accept or reject its work in
        <b>Review</b>, then run another round — each round learns from everything you labeled or accepted.</p>
      <div class="bs-stats" id="bsStats">…</div>
      <details class="hp-details"><summary>Options</summary><div class="hp-body">
        <div class="grid2">
          <div><label class="fld">Base model</label><select class="input" id="bsModel"></select></div>
          <div><label class="fld">Epochs</label>
            <input class="input" id="bsEpochs" type="number" min="0" max="500" step="10" placeholder="auto"></div>
          <div><label class="fld">Min confidence</label>
            <input class="input" id="bsConf" type="number" min="0.05" max="0.95" step="0.05" value="0.4"></div>
          <div><label class="fld" title="Boxes on the same object are kept once: the same class overlapping by IoU 0.5 or more, or any class overlapping by at least this value. The higher score wins; a box that is already on the image always wins. 1 = keep everything.">Duplicate overlap (IoU)</label>
            <input class="input" id="bsOverlap" type="number" min="0.1" max="1" step="0.05" value="0.7"
              title="Boxes on the same object are kept once: the same class overlapping by IoU 0.5 or more, or any class overlapping by at least this value. The higher score wins; a box that is already on the image always wins. 1 = keep everything."></div>
        </div>
        <label class="fld"><input type="checkbox" id="bsRefresh" checked> Also re-label images whose predictions
          nobody has reviewed yet (auto-annotated) with the new model</label>
        <label class="fld"><input type="checkbox" id="bsRejected" checked> Also re-label rejected images
          (<i>Needs re-annotation</i>) with the new model — only boxes a model made are replaced</label>
      </div></details>
      <div class="row" style="margin-top:14px;align-items:center;gap:10px">
        <button class="btn primary" id="bsGo" disabled>⚡ Bootstrap</button>
        <span class="chip" id="bsHint" hidden></span>
      </div>
      <div id="bsProgress"></div>
    </div>
    <h3 class="sect">Run a model</h3>
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
            <option value="rejected">Rejected images (needs re-annotation)</option>
          </select></div>
        <div><label class="fld" title="Boxes on the same object are kept once: the same class overlapping by IoU 0.5 or more, or any class overlapping by at least this value. The higher score wins; a box that is already on the image always wins. 1 = keep everything.">Duplicate overlap (IoU)</label>
          <input class="input" id="alOverlap" type="number" min="0.1" max="1" step="0.05" value="0.7"
            title="Boxes on the same object are kept once: the same class overlapping by IoU 0.5 or more, or any class overlapping by at least this value. The higher score wins; a box that is already on the image always wins. 1 = keep everything."></div>
      </div>
      <label class="fld"><input type="checkbox" id="alReplace"> Replace existing annotations on each image</label>
      <label class="fld" title="Off: only your project's classes are kept — predictions of any other class are dropped"><input type="checkbox" id="alCreate"> Also add new project classes for model classes I don't have (e.g. all 80 COCO classes)</label>
      <div class="row" style="margin-top:16px">
        <button class="btn primary" id="alGo">Run auto-label</button>
        <span class="chip mono" id="alStatus"></span>
      </div>
      <div id="alProgress"></div>
    </div>
    <div id="alPreview"></div>
    ${p.task_type === "classify" ? "" : `
    <h3 class="sect">Find all by concept — SAM 3</h3>
    <div class="card">
      <p class="sub" style="margin-top:0">Type what to look for (e.g. “USB port”) and SAM 3 finds every instance across
        your images. Results are saved as model predictions with their scores — check the least confident first in
        <b>Review</b>. (To find look-alikes of one specific object, use <span class="kbd">E</span> in the editor.)</p>
      <div class="row" id="faMode" style="margin-bottom:12px;gap:8px">
        <button class="btn small primary" data-mode="classes" title="SAM looks for each class by its name/phrase and labels matches automatically">By class</button>
        <button class="btn small ghost" data-mode="generic" title="SAM outlines every object without classes; you then give each one a class in the editor">Generic — outline everything, label later</button>
      </div>
      <div id="faGeneric" hidden>
        <label class="fld">Generic phrases (comma-separated nouns SAM should look for)</label>
        <input class="input" id="faPhrases" value="component, part, object">
        <p class="sub" style="margin:6px 0 0">Nothing is labeled automatically: every outlined object waits as a proposal. Open the
          editor on those images and YAAP asks for each object's class, one by one.</p>
      </div>
      <div id="faByClass">
      <div class="row" style="justify-content:space-between;align-items:center;margin-bottom:6px">
        <label class="fld" style="margin:0">Classes to find — tick the ones you want, edit the phrase if the class name isn't descriptive</label>
        <button class="btn small ghost" id="faAll">All / none</button>
      </div>
      <div id="faRows">${p.classes.map(c => `
        <div class="fa-row" data-cid="${c.id}">
          <input type="checkbox" checked>
          <span class="swatch" style="background:${c.color}"></span>
          <span class="fa-name" title="${esc(c.name)}">${esc(c.name)}</span>
          <input class="input" value="${esc(c.name)}" placeholder="phrase to find">
        </div>`).join("")}</div>
      </div>
      <div class="grid2" style="margin-top:10px">
        <div><label class="fld">Min score</label>
          <input class="input" id="faConf" type="number" min="0.05" max="0.95" step="0.05" value="0.4"></div>
        <div><label class="fld">Scope</label>
          <select class="input" id="faScope"><option value="unannotated">Only images without annotations</option>
            <option value="all">All images</option></select></div>
        <div><label class="fld" title="Boxes on the same object are kept once: the same class overlapping by IoU 0.5 or more, or any class overlapping by at least this value. The higher score wins; a box that is already on the image always wins. 1 = keep everything.">Duplicate overlap (IoU)</label>
          <input class="input" id="faOverlap" type="number" min="0.1" max="1" step="0.05" value="0.7"
            title="Boxes on the same object are kept once: the same class overlapping by IoU 0.5 or more, or any class overlapping by at least this value. The higher score wins; a box that is already on the image always wins. 1 = keep everything."></div>
      </div>
      <div class="row" style="margin-top:16px">
        <button class="btn primary" id="faGo">⌖ Find in dataset</button>
        <span class="chip warn" id="faNoClass" hidden>Add a class first — or use Generic</span>
      </div>
    </div>`}
    <h3 class="sect">Active learning — most informative images first</h3>
    <div class="card">
      <p class="sub" style="margin-top:0">Rank the images you haven't labeled by how unsure a model is about them, then switch
        on <b>“Most uncertain first”</b> in the editor so you label what helps the next training run most.
        Needs a model — train one first, or use a pretrained alias.</p>
      <div class="row" style="gap:10px"><button class="btn primary" id="rkGo">Rank unlabeled images</button>
        <span class="chip mono" id="rkState"></span></div>
    </div>
    <h3 class="sect">Bring your own weights</h3>
    <div class="card row">
      <input type="file" id="wtFile" accept=".pt" class="input" style="flex:1">
      <button class="btn" id="wtGo">Upload .pt</button>
    </div></div>`;

  $("#alGo").onclick = async () => {
    const btn = $("#alGo"); btn.disabled = true;
    $("#alStatus").textContent = "starting… (first run may download weights)";
    $("#alPreview").innerHTML = "";
    try {
      const imgs = await API.images(p.id), scope = $("#alScope").value;
      const ids = scope === "unannotated" ? imgs.filter(i => !i.annotation_count).map(i => i.id)
        : scope === "rejected" ? imgs.filter(i => i.status === "needs_reannotation").map(i => i.id) : [];
      if (scope !== "all" && !ids.length) {          // an empty list would mean "all images"
        $("#alStatus").textContent = ""; btn.disabled = false;
        return toast(scope === "rejected" ? "No rejected images." : "Every image already has annotations.");
      }
      const job = await API.autolabel(p.id, {
        model_name: $("#alModel").value, conf: +$("#alConf").value, image_ids: ids,
        replace_existing: $("#alReplace").checked, create_missing_classes: $("#alCreate").checked,
        max_overlap: +$("#alOverlap").value || 0.7,
      });
      $("#alStatus").textContent = "";
      watchAutolabelJob(p, job.id);
    } catch (e) { $("#alStatus").textContent = ""; btn.disabled = false; err(e); }
  };
  // rejected images still hold the boxes that were rejected — the new ones should replace them
  $("#alScope").onchange = () => { if ($("#alScope").value === "rejected") $("#alReplace").checked = true; };
  $("#faAll") && ($("#faAll").onclick = () => {
    const boxes = $$("#faRows input[type=checkbox]");
    const on = boxes.some(b => !b.checked);
    boxes.forEach(b => { b.checked = on; });
  });
  let faMode = "classes";
  function setFaMode(mode) {
    faMode = mode;
    $$("#faMode [data-mode]").forEach(x => {
      x.classList.toggle("primary", x.dataset.mode === mode); x.classList.toggle("ghost", x.dataset.mode !== mode);
    });
    $("#faByClass").hidden = mode !== "classes";
    $("#faGeneric").hidden = mode !== "generic";
    const blocked = mode === "classes" && !p.classes.length;     // generic mode needs no classes
    $("#faGo").disabled = blocked;
    $("#faNoClass").hidden = !blocked;
  }
  if ($("#faMode")) {
    $$("#faMode [data-mode]").forEach(b => b.onclick = () => setFaMode(b.dataset.mode));
    setFaMode(p.classes.length ? "classes" : "generic");          // no classes yet: outline everything
  }
  $("#faGo") && ($("#faGo").onclick = async () => {
    const body = { mode: faMode, conf: +$("#faConf").value, only_unannotated: $("#faScope").value === "unannotated",
                   max_overlap: +$("#faOverlap").value || 0.7 };
    if (faMode === "generic") {
      body.phrases = $("#faPhrases").value.split(",").map(s => s.trim()).filter(Boolean);
      if (!body.phrases.length) return toast("Give at least one phrase.", "err");
    } else {
      body.targets = $$("#faRows .fa-row").filter(r => $("input[type=checkbox]", r).checked)
        .map(r => ({ class_id: +r.dataset.cid, text: $("input.input", r).value.trim() }));
      if (!body.targets.length) return toast("Tick at least one class.", "err");
      if (body.targets.some(t => !t.text)) return toast("Every ticked class needs a phrase.", "err");
    }
    const btn = $("#faGo"); btn.disabled = true; $("#alGo").disabled = true;
    $("#alPreview").innerHTML = "";
    try {
      const job = await API.samFindAll(p.id, body);
      watchAutolabelJob(p, job.id, faMode === "generic" ? "generic" : "find", btn);
    } catch (e) { btn.disabled = false; $("#alGo").disabled = false; err(e); }
  });
  async function refreshRankState() {
    try {
      const s = await API.alState(p.id);
      $("#rkState").textContent = s.scored
        ? `${s.scored} image(s) ranked · ${s.model_name} · ${new Date(s.scored_at + "Z").toLocaleString()}` : "not ranked yet";
    } catch (_) {}
  }
  refreshRankState();
  $("#rkGo").onclick = async () => {
    const btn = $("#rkGo"); btn.disabled = true; $("#alGo").disabled = true;
    $("#alPreview").innerHTML = "";
    try {
      const job = await API.alScore(p.id, { model_name: $("#alModel").value });
      watchAutolabelJob(p, job.id, "score", btn, refreshRankState);
    } catch (e) { btn.disabled = false; $("#alGo").disabled = false; err(e); }
  };
  $("#wtGo").onclick = async () => {
    const f = $("#wtFile").files[0];
    if (!f) return;
    try { await API.uploadWeight(f, p.task_type, p.id); toast("Weights uploaded.", "ok"); viewAutolabel(); }
    catch (e) { err(e); }
  };
  enableFrameViewer($("#alPreview"));

  /* Bootstrap: train on the labeled images → label the rest → review → next round */
  const bsCard = $("#bsCard");
  const bsActive = r => !!r && (r.status === "training" || r.status === "labeling");
  let bs = null, bsGen = 0;
  async function loadBootstrap() {
    try { bs = await API.bootstrapState(p.id); } catch (e) { $("#bsStats").textContent = e.message; return; }
    if (!document.body.contains(bsCard)) return;
    if (!$("#bsModel").options.length)
      $("#bsModel").innerHTML = bs.models.map(m => `<option ${m === bs.default_model ? "selected" : ""}>${m}</option>`).join("");
    $("#bsEpochs").placeholder = `auto (${bs.auto_epochs})`;
    renderBsState();
    if (bs.last) renderBsRun(bs.last);
    if (bsActive(bs.last)) pollBootstrap(bs.last.id);
  }
  function renderBsState() {
    const refresh = $("#bsRefresh").checked, rejected = $("#bsRejected").checked, last = bs.last;
    const targets = bs.unannotated + (refresh ? bs.unreviewed : 0) + (rejected ? bs.rejected : 0);
    $("#bsStats").innerHTML = `
      <span><b>${bs.labeled}</b> labeled image${bs.labeled === 1 ? "" : "s"} to learn from${bs.background ? ` + ${bs.background} empty` : ""}</span>
      <span><b>${bs.unannotated}</b> unlabeled</span>
      ${bs.unreviewed ? `<span><b>${bs.unreviewed}</b> with unreviewed predictions${refresh ? " — get the new model's" : " — kept"}</span>` : ""}
      ${bs.rejected ? `<span><b>${bs.rejected}</b> rejected${rejected ? " — get the new model's" : " — kept for you to fix"}</span>` : ""}
      ${bs.new_since_last > 0 ? `<span class="ok">+${bs.new_since_last} labeled since the last round</span>` : ""}`;
    let hint = "", warn = false;
    if (bs.labeled < bs.min_labeled) { hint = `Label at least ${bs.min_labeled} images first`; warn = true; }
    else if (!targets) hint = "Nothing left to label";
    else if (bs.labeled < bs.recommended) { hint = `Works best with ${bs.recommended}+ labeled images`; warn = true; }
    else if (bs.few_examples?.length) {
      hint = "Few examples of: " + bs.few_examples.map(c => `${c.name} (${c.count})`).join(", ") +
        " — label a few more, or the model will confuse them";
      warn = true;
    }
    else if (last && last.status === "done" && last.to_review > 0 && !bs.new_since_last)
      hint = `Review round ${last.round} first — what you accept trains the next round`;
    $("#bsHint").hidden = !hint;
    $("#bsHint").textContent = hint;
    $("#bsHint").className = warn ? "chip warn" : "chip";
    $("#bsGo").disabled = bsActive(last) || bs.labeled < bs.min_labeled || !targets;
    $("#bsGo").textContent = bs.next_round > 1 ? `⚡ Run round ${bs.next_round}` : "⚡ Bootstrap";
  }
  // built once per run + status; polls only update the moving parts, so the log can be
  // scrolled and the Stop button doesn't get swapped out under the pointer
  function renderBsRun(r) {
    const host = $("#bsProgress"), active = bsActive(r), al = r.autolabel;
    const pct = r.status === "training" ? r.epoch / Math.max(1, r.epochs)
      : al ? al.processed / Math.max(1, al.total) : 0;
    const msg = r.status === "failed" ? r.error : r.message;
    const key = `${r.id}:${r.status}:${!!(al && al.log)}`;
    if (host.dataset.key === key) {
      $(".bs-msg", host).textContent = msg;
      const bar = $(".bs-bar i", host); if (bar) bar.style.width = `${Math.round(pct * 100)}%`;
      const lb = $(".logbox", host);
      if (lb && al) {
        const follow = lb.scrollHeight - lb.scrollTop - lb.clientHeight < 30;
        if (lb.textContent !== al.log) { lb.textContent = al.log; if (follow) lb.scrollTop = lb.scrollHeight; }
      }
      return;
    }
    host.dataset.key = key;
    const phase = { training: "step 1/2 · training", labeling: "step 2/2 · labeling" }[r.status] || r.status;
    host.innerHTML = `
      <div class="row bs-head">
        <span class="pill ${active ? "running" : r.status}">round ${r.round} · ${phase}</span>
        <span class="mono bs-msg ${r.status === "failed" ? "wrap" : ""}" title="${esc(msg)}">${esc(msg)}</span>
        ${active ? `<button class="btn small danger" id="bsStop">Stop</button>` : ""}
      </div>
      ${active ? `<span class="hl-bar bs-bar"><i style="width:${Math.round(pct * 100)}%"></i></span>` : ""}
      ${r.status === "training" ? `<p class="sub bs-note">${esc(r.model_arch)} on ${r.train_images} labeled image(s) ·
        full log under Train → job #${r.train_job_id}</p>` : ""}
      ${al && al.log ? (active ? `<div class="logbox">${esc(al.log)}</div>`
        : `<details class="bs-log"><summary>Labeling log</summary><div class="logbox">${esc(al.log)}</div></details>`) : ""}
      ${r.status === "done" && r.to_review ? `
        <div class="row" style="margin-top:12px;align-items:center;gap:12px">
          <button class="btn primary" id="bsReview">Review round ${r.round} → <span class="mono">${r.to_review}</span></button>
          <span class="sub" style="margin:0">Accept (<span class="kbd">A</span>) what's right, reject
            (<span class="kbd">R</span>) what's wrong — accepted images train the next round.</span>
        </div>` : ""}`;
    const lb = $(".logbox", host);
    if (lb && active) lb.scrollTop = lb.scrollHeight;
    $("#bsStop") && ($("#bsStop").onclick = async () => {
      if (!confirm(`Stop bootstrap round ${r.round}? Images it already labeled keep their predictions.`)) return;
      try { renderBsRun(await API.stopBootstrap(p.id, r.id)); } catch (e) { err(e); }
      loadBootstrap();
    });
    $("#bsReview") && ($("#bsReview").onclick = () => {
      S.reviewOnly = r.labeled_image_ids;
      startQuickReview(null, "review");
    });
  }
  function pollBootstrap(bid) {
    const gen = ++bsGen;            // a newer poll (or leaving the page) ends this one
    let watched = false;
    const tick = async () => {
      if (gen !== bsGen || !document.body.contains(bsCard)) return;
      let r;
      try { r = await API.bootstrapRun(p.id, bid); } catch (_) { setTimeout(tick, 4000); return; }
      if (gen !== bsGen || !document.body.contains(bsCard)) return;
      if (bs) bs.last = r;
      renderBsRun(r);
      if (bsActive(r)) {
        if (!watched && bs) renderBsState();   // disable ⚡ while it runs
        watched = true; setTimeout(tick, 1500); return;
      }
      if (watched && r.status === "done") {
        toast(`Bootstrap round ${r.round}: ${r.message}`, "ok", 8000);
        refreshModelPicker();
      } else if (watched && r.status === "failed") toast(`Bootstrap round ${r.round} failed: ${r.error}`, "err");
      loadBootstrap();
    };
    tick();
  }
  async function refreshModelPicker() {        // the new "Bootstrap rN" weights for Run a model / Rank
    try {
      const m = await API.models(p.id, p.task_type), sel = $("#alModel"), keep = sel.value;
      const have = new Set([...sel.options].map(o => o.value));
      m.weights.filter(w => !have.has(`weight:${w.id}`)).reverse().forEach(w => {
        sel.insertAdjacentHTML("afterbegin", `<option value="weight:${w.id}">★ ${esc(w.name)} (${w.source})</option>`);
      });
      sel.value = keep;
    } catch (_) {}
  }
  $("#bsRefresh").onchange = $("#bsRejected").onchange = () => { if (bs) renderBsState(); };
  $("#bsGo").onclick = async () => {
    const btn = $("#bsGo");
    btn.disabled = true;
    try {
      const r = await API.bootstrap(p.id, {
        model_arch: $("#bsModel").value, epochs: +$("#bsEpochs").value || 0,
        conf: +$("#bsConf").value || 0.4, refresh_unreviewed: $("#bsRefresh").checked,
        relabel_rejected: $("#bsRejected").checked, max_overlap: +$("#bsOverlap").value || 0.7,
      });
      toast(`Bootstrap round ${r.round} started — training on ${r.train_images} labeled image(s).`, "ok");
      if (bs) bs.last = r;
      renderBsRun(r);
      pollBootstrap(r.id);
    } catch (e) { btn.disabled = false; err(e); }
  };
  loadBootstrap();
}

function renderAlProgress(job) {
  $("#alProgress").innerHTML = `
    <div class="row" style="align-items:center;gap:10px;margin-top:14px">
      <span class="pill ${job.status}">${job.status}</span>
      <span class="mono" style="font-size:12px;color:var(--muted)">${job.processed}/${job.total} images
        · ${job.images_labeled} labeled · +${job.annotations_added} annotations${job.skipped.length ? ` · ${job.skipped.length} skipped` : ""}</span>
    </div>
    <div class="logbox">${esc(job.log || "starting…")}</div>`;
}

// kind: "label" (auto-label), "find" (SAM concept search) → both end in the preview grid;
// "score" (active-learning ranking) writes no annotations, so no grid.
async function watchAutolabelJob(p, jid, kind = "label", btn = null, onDone = null) {
  const done = () => { $("#alGo").disabled = false; if (btn) btn.disabled = false; if (onDone) onDone(); };
  const job = await API.autolabelJob(p.id, jid).catch(err);
  if (!$("#alProgress")) return;          // left the Auto-label page: stop polling
  if (!job) { done(); return; }
  renderAlProgress(job);
  if (job.status === "running") { setTimeout(() => watchAutolabelJob(p, jid, kind, btn, onDone), 1200); return; }

  done();
  if (job.status === "failed") { toast(`${kind === "score" ? "Ranking" : "Auto-label"} failed: ${job.error}`, "err"); return; }
  if (kind === "generic") {
    toast(`Outlined objects in ${job.images_labeled}/${job.total} images (${job.annotations_added} proposals).`, "ok", 8000);
    const noClasses = !p.classes.length;
    $("#alPreview").innerHTML = job.images_labeled
      ? `<div class="card" style="margin-top:16px"><p class="sub" style="margin-top:0">${job.annotations_added} object(s) outlined in
           ${job.images_labeled} image(s). They have no class yet — ${noClasses ? "add your classes, then " : ""}the editor
           walks through them and asks for each class.</p>
         <button class="btn primary" id="goLabel">${noClasses ? "Add classes →" : "Label the found objects →"}</button></div>`
      : `<p class="sub" style="margin-top:16px">Nothing found — try other phrases or a lower min score.</p>`;
    $("#goLabel") && ($("#goLabel").onclick = () => {
      if (noClasses) return nav("classes");
      S.annotateScope = "proposals"; nav("annotate");
    });
    return;
  }
  if (kind === "score") { toast(`Ranked ${job.images_labeled}/${job.total} images by uncertainty.`, "ok", 8000); return; }
  toast(`${kind === "find" ? "Found objects in" : "Labeled"} ${job.images_labeled}/${job.total} images · +${job.annotations_added} annotations`, "ok", 8000);
  await renderAlPreviewGrid(p, jid);
}

async function renderAlPreviewGrid(p, jid) {
  let data;
  try { data = await API.autolabelPreview(p.id, jid); } catch (e) { return err(e); }
  if (!data.images.length) {
    $("#alPreview").innerHTML = `<p class="sub" style="margin-top:16px">Nothing was labeled — try a lower confidence threshold.</p>`;
    return;
  }

  const card = im => {
    const shapes = annotationShapesSvg(im.annotations, p.classes, im.width);
    const approved = im.status === "review";
    return `<div class="al-card ${approved ? "approved" : ""}" data-id="${im.id}">
      <div class="al-frame"><svg viewBox="0 0 ${im.width} ${im.height}" preserveAspectRatio="xMidYMid meet">
        <image href="${API.imageUrl(p.id, im.id)}" width="${im.width}" height="${im.height}"></image>
        ${shapes}
      </svg></div>
      <div class="al-meta">
        <span class="mono" title="${esc(im.filename)}">${esc(im.filename)}</span>
        <span class="mono">${im.annotations.length} obj</span>
      </div>
      <div class="row" style="padding:0 10px 10px">
        <button class="btn small ${approved ? "primary" : ""}" data-approve="${im.id}" ${approved ? "disabled" : ""}>
          ${approved ? "✓ Approved" : "✓ Approve → To review"}</button>
        <a class="btn small ghost" href="#/p/${p.id}/annotate" data-open="${im.id}">Open</a>
      </div>
    </div>`;
  };

  $("#alPreview").innerHTML = `
    <div class="row" style="align-items:center;justify-content:space-between;margin-top:20px">
      <h3 class="sect" style="margin:0">Review auto-labeled images (${data.images.length})</h3>
      <button class="btn primary small" id="alApproveAll">✓ Approve all → To review</button>
    </div>
    <div class="al-grid">${data.images.map(card).join("")}</div>`;

  $$("[data-approve]", $("#alPreview")).forEach(btn => btn.onclick = async () => {
    const iid = +btn.dataset.approve;
    try { await API.patchImage(p.id, iid, { status: "review" }); } catch (e) { return err(e); }
    const im = data.images.find(x => x.id === iid);
    if (im) im.status = "review";
    btn.closest(".al-card").classList.add("approved");
    btn.textContent = "✓ Approved"; btn.classList.add("primary"); btn.disabled = true;
    toast("Approved — moved to the To review queue.", "ok");
  });

  $$("[data-open]", $("#alPreview")).forEach(a => a.onclick = () => { S.pendingImageId = +a.dataset.open; });

  $("#alApproveAll").onclick = async () => {
    const pending = data.images.filter(im => im.status !== "review");
    if (!pending.length) return toast("Everything here is already approved.", "");
    if (!confirm(`Approve all ${pending.length} remaining auto-labeled image(s)? They'll move to the To review queue.`)) return;
    try { await Promise.all(pending.map(im => API.patchImage(p.id, im.id, { status: "review" }))); }
    catch (e) { return err(e); }
    pending.forEach(im => { im.status = "review"; });
    await renderAlPreviewGrid(p, jid);
    toast(`Approved ${pending.length} image(s).`, "ok");
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
        <option value="coco">YOLO + COCO — adds _annotations.coco.json per split, for RF-DETR / HuggingFace / detectron2 (still trainable in-platform)</option>
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

  // Split %s always add up to 100: editing one rebalances the other two,
  // keeping their ratio to each other.
  const spIds = ["#spTrain", "#spVal", "#spTest"];
  spIds.forEach(id => $(id).onchange = () => {
    const el = $(id);
    const v = Math.max(0, Math.min(100, Math.round(+el.value || 0)));
    el.value = v;
    const [a, b] = spIds.filter(o => o !== id).map(o => $(o));
    const rest = 100 - v, oa = +a.value || 0, ob = +b.value || 0;
    const na = oa + ob ? Math.round(rest * oa / (oa + ob)) : Math.round(rest / 2);
    a.value = na;
    b.value = rest - na;
  });

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
function healthHtml(h, task) {
  const bar = (n, max, color = "var(--teal)") =>
    `<span class="hl-bar"><i style="width:${max ? Math.round(100 * n / max) : 0}%;background:${color}"></i></span>`;
  const sc = h.score >= 80 ? "var(--teal)" : h.score >= 50 ? "var(--amber)" : "var(--red)";
  const imgTotal = h.total_images || 1;
  const splitRows = ["train", "val", "test"].map(s => `
    <div class="hl-row"><span>${s}</span>${bar(h.images[s], imgTotal)}
      <span class="mono">${h.images[s]} img · ${Math.round(100 * h.images[s] / imgTotal)}%${task === "classify" ? "" : ` · ${h.empty_images[s]} empty`}</span></div>`).join("");
  const maxCls = Math.max(1, ...h.classes.map(c => c.total));
  const clsRows = h.classes.map(c => `
    <div class="hl-row"><span title="${esc(c.name)}">${esc(c.name)}</span>
      <span class="hl-stack" style="width:${Math.round(100 * c.total / maxCls)}%">
        <i style="flex:${c.train};background:var(--teal)" title="train ${c.train}"></i>
        <i style="flex:${c.val};background:var(--amber)" title="val ${c.val}"></i>
        <i style="flex:${c.test};background:var(--blue)" title="test ${c.test}"></i></span>
      <span class="mono">${c.total} <span style="color:var(--muted)">(${c.train}/${c.val}/${c.test})</span></span></div>`).join("");
  const warn = h.warnings.map(w => `<div class="hl-warn ${w.level}">${w.level === "error" ? "✕" : w.level === "warn" ? "⚠" : "ℹ"} ${esc(w.msg)}</div>`).join("")
    || `<div class="hl-warn ok">✓ No issues found.</div>`;
  const maxH = Math.max(1, ...h.boxes_per_image.hist);
  const hist = h.boxes_per_image.hist.map((n, i) =>
    `<span class="hl-col" title="${i === 10 ? "10+" : i} per image: ${n} image(s)"><i style="height:${Math.round(100 * n / maxH)}%"></i><b>${i === 10 ? "10+" : i}</b></span>`).join("");
  const kv = o => Object.entries(o).map(([k, v]) => `${k} <b>${v}</b>`).join(" · ") || "—";
  return `<div class="health">
    <div class="hl-score" style="color:${sc}">${h.score}<small>/100 health</small></div>
    <div class="hl-warns">${warn}</div>
    <h3 class="sect">Splits</h3>${splitRows}
    <h3 class="sect">Labels per class <span class="sub" style="font-size:11px">(train / val / test)</span></h3>${clsRows || "—"}
    <p class="sub">${h.labels_total} label(s)${h.imbalance_ratio ? ` · imbalance ${h.imbalance_ratio}:1` : ""}</p>
    ${task === "classify" ? "" : `
    <h3 class="sect">Objects per image · mean ${h.boxes_per_image.mean}, max ${h.boxes_per_image.max}</h3>
    <div class="hl-hist">${hist}</div>
    <h3 class="sect">Object size &amp; shape</h3>
    <p class="sub">size: ${kv(h.box_size)}<br>shape: ${kv(h.aspect)}</p>`}
    <h3 class="sect">Duplicates</h3>
    <p class="sub">within a split <b>${h.duplicates.within_split}</b> · across splits (leakage) <b>${h.duplicates.cross_split}</b></p>
  </div>`;
}

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
        ? `${st.format === "coco" ? "YOLO+COCO" : "YOLO"} · ${total} image(s) · train ${src.train ?? 0} (${pct(src.train ?? 0)}%) · ` +
          `val ${src.val ?? 0} (${pct(src.val ?? 0)}%) · test ${src.test ?? 0} (${pct(src.test ?? 0)}%)` +
          (st.annotations != null ? ` · ${st.annotations} labels` : "") : (st.error || "");
      return `<div class="v-row">
        <span class="vname">${esc(v.name)}</span>
        <span class="pill ${v.status}">${v.status}</span>
        <span class="meta">${esc(meta)}</span>
        ${v.status === "ready" ? `<button class="btn small" data-health="${v.id}" title="Dataset health statistics">♥ Health</button>` : ""}
        ${v.status === "ready" ? `<a class="btn small" href="${API.versionZip(p.id, v.id)}">⬇ zip</a>` : ""}
        <button class="btn ghost small" data-del="${v.id}">✕</button>
      </div>`;
    }).join("") || `<div class="empty">No versions yet — use Generate.</div>`;
    $$("[data-del]").forEach(b => b.onclick = async () => {
      try { await API.deleteVersion(p.id, +b.dataset.del); render(); } catch (e) { err(e); }
    });
    $$("[data-health]").forEach(b => b.onclick = async () => {
      const v = versions.find(x => x.id === +b.dataset.health);
      b.disabled = true;
      try { openViewer(`Dataset health · ${v.name}`, healthHtml(await API.versionHealth(p.id, v.id), p.task_type),
                       { hint: "computed from the exported files" }); }
      catch (e) { err(e); }
      b.disabled = false;
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
    <p class="sub">Three ways to train a model from a generated version: locally through this app, on a
      remote GPU machine over SSH, or on Google's free GPUs via Colab.</p>

    <div class="row" style="margin-bottom:16px" id="trainTabs">
      <button class="btn small primary" data-tab="local">🖥 Local train</button>
      <button class="btn small ghost" data-tab="ssh"><span class="ico-white">🔌</span> SSH</button>
      <button class="btn small ghost" data-tab="colab">☁ Google Colab</button>
    </div>

    <div id="trainLocalPane">
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
          <div><label class="fld" title="Stop early if val loss hasn't improved for this many epochs. 0 disables early stopping.">Patience</label>
            <input class="input" id="trPatience" type="number" value="10" min="0"></div>
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
    </div>

    <div id="trainSshPane" hidden>
      <div class="card">
        <h3 class="sect" style="margin-top:0">🔌 Remote GPU over SSH</h3>
        <p class="sub" style="margin-bottom:6px">Trains on another machine; log, stop and results stay here.
          Key-based auth only (no passwords) — hosts from <span class="mono">~/.ssh/config</span> work too.</p>
        <div class="grid2">
          <div><label class="fld">Host</label><input class="input" id="sshHost" placeholder="192.168.1.50 or ssh-config alias"></div>
          <div><label class="fld">User</label><input class="input" id="sshUser"></div>
          <div><label class="fld">Port</label><input class="input" id="sshPort" type="number" value="22"></div>
          <div><label class="fld">Identity file (optional)</label><input class="input" id="sshKey" placeholder="~/.ssh/id_ed25519"></div>
          <div><label class="fld">Remote python</label><input class="input" id="sshPython" placeholder="~/yaap-env/bin/python"></div>
          <div><label class="fld">Remote work dir</label><input class="input" id="sshWorkdir" placeholder="~/yaap_jobs"></div>
        </div>
        <label class="fld" style="display:flex;gap:8px;align-items:center"><input type="checkbox" id="sshCleanup">
          Delete remote job folder after download</label>
        <div class="row" style="margin-top:10px">
          <button class="btn small" id="sshTest">Test connection</button>
          <span class="mono" id="sshTestResult" style="font-size:12px"></span>
        </div>
        <h3 class="sect">Training options</h3>
        <div class="grid2">
          <div><label class="fld">Dataset version</label><select class="input" id="sshVer">${verOpts}</select></div>
          <div><label class="fld">Architecture / starting weights</label>
            <select class="input" id="sshArch">
              ${models.pretrained.map(m => `<option ${m === "yolo11n.pt" ? "selected" : ""}>${m}</option>`).join("")}
              ${(models.rtdetr || []).map(m => `<option value="${m}">${m} (RT-DETR)</option>`).join("")}
              ${models.weights.map(w => `<option value="weight:${w.id}">★ ${esc(w.name)}</option>`).join("")}
            </select></div>
          <div><label class="fld">Epochs</label><input class="input" id="sshEpochs" type="number" value="100"></div>
          <div><label class="fld">Image size</label><input class="input" id="sshImgsz" type="number" value="640"></div>
          <div><label class="fld">Batch</label><input class="input" id="sshBatch" type="number" value="16"></div>
          <div><label class="fld">Patience</label><input class="input" id="sshPatience" type="number" value="10" min="0"></div>
          <div><label class="fld">Device</label>
            <select class="input" id="sshDevice"><option value="auto">auto (GPU if available)</option>
              <option value="cpu">cpu</option><option value="cuda:0">cuda:0</option></select></div>
        </div>
        <details class="hp-details" id="sshHpDetails">
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
          <button class="btn primary" id="sshGo" ${ready.length ? "" : "disabled"}>▶ Start training on SSH host</button>
          ${ready.length ? "" : `<span class="chip warn">Generate a dataset version first</span>`}
        </div>
      </div>
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

    <h3 class="sect">Jobs</h3>
      <div id="jobList"></div>
      <div class="joblog-head" id="jobLogHead">
        <span class="mono" style="font-size:11px;color:var(--muted)" id="jobLogTitle"></span>
        <button class="btn small" id="jobLogFull">⛶ Full screen</button>
      </div>
      <div class="logbox" id="jobLog" style="display:none"></div>
      <div class="row job-plots" id="jobPlots" style="display:none">
        <div style="flex:1;min-width:260px">
          <p class="sub" style="margin:12px 0 4px">Results <span class="plot-hint">· click to open full screen</span></p>
          <img id="jobPlotResults" alt="results.png" style="display:none">
        </div>
        <div style="flex:1;min-width:260px">
          <p class="sub" style="margin:12px 0 4px">Confusion matrix <span class="plot-hint">· click to open full screen</span></p>
          <img id="jobPlotConfusion" alt="confusion_matrix.png" style="display:none">
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
    $("#trainSshPane").hidden = tab !== "ssh";
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
  wireHpRows($("#sshHpDetails"));

  $("#trGo").onclick = async () => {
    try {
      await API.train(p.id, {
        version_id: +$("#trVer").value, model_arch: $("#trArch").value,
        epochs: +$("#trEpochs").value, imgsz: +$("#trImgsz").value,
        batch: +$("#trBatch").value, patience: +$("#trPatience").value,
        device: $("#trDevice").value,
        extra: collectTrainExtra($("#trHpDetails")),
      });
      toast("Training started.", "ok"); renderJobs();
    } catch (e) { err(e); }
  };

  const sshFields = { host: "#sshHost", user: "#sshUser", identity_file: "#sshKey",
    python: "#sshPython", workdir: "#sshWorkdir" };
  const readSsh = () => ({
    ...Object.fromEntries(Object.entries(sshFields).map(([k, s]) => [k, $(s).value.trim()])),
    port: +$("#sshPort").value || 22, cleanup: $("#sshCleanup").checked });
  API.sshSettings().then(s => {
    Object.entries(sshFields).forEach(([k, sel]) => { $(sel).value = s[k] || ""; });
    $("#sshPort").value = s.port; $("#sshCleanup").checked = !!s.cleanup;
  }).catch(() => {});
  $("#sshTest").onclick = async () => {
    const out = $("#sshTestResult");
    out.style.color = "var(--muted)"; out.textContent = "connecting…";
    try {
      const r = await API.sshTest(readSsh());
      out.style.color = r.ok ? "var(--teal)" : "var(--red)";
      out.textContent = r.ok
        ? `✓ torch ${r.torch} · ultralytics ${r.ultralytics} · ` +
          (r.cuda ? `CUDA: ${r.gpus.join(", ")}` : "⚠ CUDA not available")
        : `✗ ${r.error}`;
    } catch (e) { out.style.color = "var(--red)"; out.textContent = `✗ ${e.message || e}`; }
  };
  $("#sshGo").onclick = async () => {
    try {
      await API.saveSshSettings(readSsh());
      await API.train(p.id, {
        mode: "ssh", version_id: +$("#sshVer").value, model_arch: $("#sshArch").value,
        epochs: +$("#sshEpochs").value, imgsz: +$("#sshImgsz").value,
        batch: +$("#sshBatch").value, patience: +$("#sshPatience").value,
        device: $("#sshDevice").value,
        extra: collectTrainExtra($("#sshHpDetails")),
      });
      toast("SSH training started.", "ok"); renderJobs();
    } catch (e) { err(e); }
  };

  let watching = null, plotsKey = null;
  async function renderJobs() {
    let jobs = [];
    try { jobs = await API.jobs(p.id); } catch (e) { return; }
    $("#jobList").innerHTML = jobs.map(j => `
      <div class="v-row">
        <span class="vname mono">#${j.id}</span>
        <span class="pill ${j.status}">${j.status}</span>
        <span class="meta">${esc(j.model_arch)} · ${j.bootstrap_round ? `bootstrap round ${j.bootstrap_round}` : `version ${j.version_id}`}</span>
        ${j.ssh_host ? `<span class="pill ssh-host" title="Trained over SSH">🔌 ${esc(j.ssh_host)}</span>` : ""}
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
      if (!confirm(`Delete job #${jid}? This also deletes its training data on disk — weights, log, results.png, confusion_matrix.png — permanently.`)) return;
      const job = jobs.find(x => x.id === jid);
      const remote = !!(job && job.ssh_host) &&
        confirm(`Also delete job #${jid}'s folder on ${job.ssh_host}?`);
      try {
        await API.deleteJob(p.id, jid, remote);
        if (watching === jid) {
          watching = null;
          $("#jobLog").style.display = "none";
          $("#jobLogHead").classList.remove("show");
          $("#jobPlots").style.display = "none";
          plotsKey = null;
          closeViewer();
        }
        renderJobs();
      } catch (e) { err(e); }
    });
  }
  // only follow the tail if the user hasn't scrolled away from the bottom —
  // otherwise every 3.5s refresh yanks them back down mid-read
  function writeLog(box, text) {
    const wasAtBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
    box.textContent = text;
    if (wasAtBottom) box.scrollTop = box.scrollHeight;
    return wasAtBottom;
  }
  async function pollLog() {
    if (!watching) return;
    const full = $("#viewerLog");
    try {
      const r = await API.jobLog(p.id, watching, full ? 5000 : 200);
      const text = r.log || "(no output yet)";
      const box = $("#jobLog");
      box.style.display = "block";
      $("#jobLogHead").classList.add("show");
      const following = writeLog(box, text);
      $("#jobLogTitle").textContent = `job #${watching} · ${r.status}` +
        (following ? "" : " · scrolled up — jump to bottom to resume auto-follow");
      if (full) {
        writeLog(full, text);
        $("#viewerTitle").textContent = `Training log · job #${watching} · ${r.status}`;
      }
      showPlots(watching, r.status);
    } catch (_) {}
  }
  function showPlots(jid, status) {
    // a finished job's plots never change — load them once instead of
    // re-fetching (and flickering) on every poll
    const key = `${jid}:${status}`;
    if (key === plotsKey && status !== "running") return;
    plotsKey = key;
    const r = $("#jobPlotResults"), c = $("#jobPlotConfusion");
    $("#jobPlots").style.display = "flex";
    r.onload = () => { r.style.display = "block"; };
    r.onerror = () => { r.style.display = "none"; };
    c.onload = () => { c.style.display = "block"; };
    c.onerror = () => { c.style.display = "none"; };
    // cache-bust — results.png updates live as ultralytics trains, so a plain
    // reload while still "running" would otherwise keep showing a stale copy
    const t = Date.now();
    r.src = API.jobPlotUrl(p.id, jid, "results") + `?t=${t}`;
    c.src = API.jobPlotUrl(p.id, jid, "confusion_matrix") + `?t=${t}`;
  }
  $("#jobLogFull").onclick = () => {
    openViewer(`Training log · job #${watching}`, `<pre class="logbox viewer-log" id="viewerLog"></pre>`,
      { hint: "updates live while the job runs" });
    pollLog();
  };
  $("#jobPlotResults").onclick = () => openImageViewer(`job #${watching} · results.png`, $("#jobPlotResults").src);
  $("#jobPlotConfusion").onclick = () => openImageViewer(`job #${watching} · confusion_matrix.png`, $("#jobPlotConfusion").src);
  await renderJobs();
  S.jobPoll = setInterval(() => { renderJobs(); pollLog(); }, 3500);
}

/* ── test model: run inference over a version's test-split images ──── */
async function viewTestModel() {
  const p = S.project;
  let versions = [], models;
  try { [versions, models] = await Promise.all([API.versions(p.id), API.models(p.id, p.task_type)]); }
  catch (e) { return err(e); }
  const ready = versions.filter(v => v.status === "ready");
  const verOpts = ready.map(v => `<option value="${v.id}">${esc(v.name)}</option>`).join("")
    || `<option value="">No versions yet — use Generate</option>`;
  const modelOpts = [
    ...models.weights.map(w => `<option value="weight:${w.id}">★ ${esc(w.name)} (${w.source})</option>`),
    ...models.pretrained.map(m => `<option value="${m}">${m}</option>`),
    ...(models.rtdetr || []).map(m => `<option value="${m}">${m} (RT-DETR)</option>`),
    ...(models.rfdetr_available ? models.rfdetr : []).map(m => `<option value="${m}">${m} (RF-DETR)</option>`),
  ].join("");

  $("#stage").innerHTML = `<div class="stage-pad" style="max-width:980px">
    <h1 class="page">Test model</h1>
    <p class="sub">Pick a dataset version and a model — its own trained job, an uploaded
      weights file, or a pretrained alias — and run it over that version's test-split
      images to eyeball how it's doing. Nothing here is saved; it's pure visualization.</p>
    <div class="card">
      <div class="grid2">
        <div><label class="fld">Dataset version (uses its test split)</label>
          <select class="input" id="tmVer">${verOpts}</select></div>
        <div><label class="fld">Model</label>
          <select class="input" id="tmModel">${modelOpts}</select></div>
        <div><label class="fld">Confidence threshold</label>
          <input class="input" id="tmConf" type="number" min="0.05" max="0.95" step="0.05" value="0.4"></div>
        <div><label class="fld">IoU threshold</label>
          <input class="input" id="tmIou" type="number" min="0.05" max="0.95" step="0.05" value="0.5"></div>
        <div><label class="fld">Images to test</label>
          <input class="input" id="tmLimit" type="number" min="1" max="100" value="20"></div>
      </div>
      <div class="row" style="margin-top:16px">
        <button class="btn primary" id="tmGo" ${ready.length ? "" : "disabled"}>▶ Run test</button>
        ${ready.length ? "" : `<span class="chip warn">Generate a dataset version first</span>`}
        <span class="chip mono" id="tmStatus"></span>
      </div>
    </div>
    <h3 class="sect">Bring your own weights</h3>
    <div class="card row">
      <input type="file" id="wtFile" accept=".pt" class="input" style="flex:1">
      <button class="btn" id="wtGo">Upload .pt</button>
    </div>

    <div class="row" id="tmToggles" style="margin-top:20px;display:none">
      <label class="fld" style="margin:0"><input type="checkbox" id="tmShowBoxes" checked> Show boxes</label>
      <label class="fld" style="margin:0"><input type="checkbox" id="tmShowLabels" checked> Show labels</label>
    </div>
    <div class="al-grid" id="tmPreview" style="margin-top:12px"></div>
    <div class="row" style="justify-content:center;margin-top:14px">
      <button class="btn" id="tmLoadMore" hidden>Load more</button>
    </div>

    <div class="tm-powered">
      <img src="/static/img/cvaih-logo.png" alt="ComputerVisionAIHub">
      <span>Powered by ComputerVisionAIHub</span>
    </div>
  </div>`;

  $("#wtGo").onclick = async () => {
    const f = $("#wtFile").files[0];
    if (!f) return;
    try { await API.uploadWeight(f, p.task_type, p.id); toast("Weights uploaded.", "ok"); viewTestModel(); }
    catch (e) { err(e); }
  };

  $("#tmShowBoxes").onchange = () => $("#tmPreview").classList.toggle("hide-boxes", !$("#tmShowBoxes").checked);
  $("#tmShowLabels").onchange = () => $("#tmPreview").classList.toggle("hide-labels", !$("#tmShowLabels").checked);
  enableFrameViewer($("#tmPreview"));

  let total = 0, shownSoFar = 0;
  function card(im) {
    const shapes = annotationShapesSvg(im.predictions, p.classes, im.width);
    return `<div class="al-card" data-id="${im.id}">
      <div class="al-frame"><svg viewBox="0 0 ${im.width} ${im.height}" preserveAspectRatio="xMidYMid meet">
        <image href="${API.imageUrl(p.id, im.id)}" width="${im.width}" height="${im.height}"></image>
        ${shapes}
      </svg></div>
      <div class="al-meta">
        <span class="mono" title="${esc(im.filename)}">${esc(im.filename)}</span>
        <span class="mono">${im.predictions.length} pred.</span>
      </div>
    </div>`;
  }

  async function runPage(offset) {
    const body = {
      version_id: +$("#tmVer").value, model_name: $("#tmModel").value,
      conf: +$("#tmConf").value, iou: +$("#tmIou").value,
      limit: +$("#tmLimit").value, offset,
    };
    const r = await API.testModel(p.id, body);
    total = r.test_split_total;
    shownSoFar = offset + r.shown;
    const host = $("#tmPreview");
    if (offset === 0) host.innerHTML = "";
    host.insertAdjacentHTML("beforeend", r.images.map(card).join(""));
    $("#tmToggles").style.display = r.images.length ? "flex" : "none";
    $("#tmLoadMore").hidden = shownSoFar >= total;
    if (!r.images.length && offset === 0) {
      host.innerHTML = `<div class="empty" style="grid-column:1/-1">
        ${r.skipped.length ? "Every image in this batch failed to run — check the model/weights." : "Nothing to show."}</div>`;
    }
    if (r.skipped.length) toast(`${r.skipped.length} image(s) skipped (missing file or inference error).`, "");
  }

  $("#tmGo").onclick = async () => {
    if (!$("#tmVer").value) return toast("Generate a dataset version first.", "err");
    const btn = $("#tmGo"); btn.disabled = true;
    $("#tmStatus").textContent = "running…";
    try { await runPage(0); }
    catch (e) { err(e); }
    $("#tmStatus").textContent = ""; btn.disabled = false;
  };
  $("#tmLoadMore").onclick = async () => {
    const btn = $("#tmLoadMore"); btn.disabled = true;
    try { await runPage(shownSoFar); } catch (e) { err(e); }
    btn.disabled = false;
  };
}
