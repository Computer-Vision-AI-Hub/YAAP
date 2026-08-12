/* YAAP API client — thin fetch wrappers. */
const API = {
  async _j(url, opts = {}) {
    const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    return r.json();
  },

  // projects & classes
  projects:      ()            => API._j("/api/projects"),
  createProject: (b)           => API._j("/api/projects", { method: "POST", body: JSON.stringify(b) }),
  project:       (id)          => API._j(`/api/projects/${id}`),
  deleteProject: (id)          => API._j(`/api/projects/${id}`, { method: "DELETE" }),
  addClass:      (pid, b)      => API._j(`/api/projects/${pid}/classes`, { method: "POST", body: JSON.stringify(b) }),
  updateClass:   (pid, cid, b) => API._j(`/api/projects/${pid}/classes/${cid}`, { method: "PATCH", body: JSON.stringify(b) }),
  deleteClass:   (pid, cid)    => API._j(`/api/projects/${pid}/classes/${cid}`, { method: "DELETE" }),

  // images
  images:      (pid)           => API._j(`/api/projects/${pid}/images`),
  deleteImage: (pid, iid)      => API._j(`/api/projects/${pid}/images/${iid}`, { method: "DELETE" }),
  patchImage:  (pid, iid, b)   => API._j(`/api/projects/${pid}/images/${iid}`, { method: "PATCH", body: JSON.stringify(b) }),
  imageUrl:    (pid, iid)      => `/api/projects/${pid}/images/${iid}/file`,
  thumbUrl:    (pid, iid)      => `/api/projects/${pid}/images/${iid}/thumb`,
  async upload(pid, files, onOne) {
    const out = { created: [], skipped: [] };
    for (const f of files) {                       // sequential: keeps UI progress honest
      const fd = new FormData(); fd.append("files", f);
      const r = await fetch(`/api/projects/${pid}/images`, { method: "POST", body: fd });
      const j = await r.json();
      out.created.push(...(j.created || [])); out.skipped.push(...(j.skipped || []));
      onOne && onOne(f.name, j);
    }
    return out;
  },

  // annotations
  annotations:  (pid, iid)     => API._j(`/api/projects/${pid}/images/${iid}/annotations`),
  saveAnnotations: (pid, iid, anns) =>
    API._j(`/api/projects/${pid}/images/${iid}/annotations`,
      { method: "PUT", body: JSON.stringify({ annotations: anns }) }),

  // models / inference
  models:    (pid, task)       => API._j(`/api/models?pid=${pid}&task=${task}`),
  autolabel: (pid, b)          => API._j(`/api/projects/${pid}/autolabel`, { method: "POST", body: JSON.stringify(b) }),
  device:    ()                => API._j("/api/system/device"),
  async uploadWeight(file, task, pid) {
    const fd = new FormData(); fd.append("file", file);
    const r = await fetch(`/api/models/upload?task=${task}&pid=${pid}`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || "upload failed");
    return r.json();
  },

  // SAM
  samModels:  ()               => API._j("/api/sam/models"),
  samPredict: (pid, iid, b)    => API._j(`/api/projects/${pid}/images/${iid}/sam`, { method: "POST", body: JSON.stringify(b) }),

  // datasets
  versions:      (pid)         => API._j(`/api/projects/${pid}/versions`),
  createVersion: (pid, b)      => API._j(`/api/projects/${pid}/versions`, { method: "POST", body: JSON.stringify(b) }),
  deleteVersion: (pid, vid)    => API._j(`/api/projects/${pid}/versions/${vid}`, { method: "DELETE" }),
  versionZip:    (pid, vid)    => `/api/projects/${pid}/versions/${vid}/download`,
  exportRawUrl:  (pid, approvedOnly) => `/api/projects/${pid}/export?approved_only=${!!approvedOnly}`,
  async importDataset(pid, file, defaultSplit) {
    const fd = new FormData(); fd.append("file", file);
    const r = await fetch(`/api/projects/${pid}/import?default_split=${defaultSplit || ""}`,
      { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || "import failed");
    return r.json();
  },
  mergeProject: (pid, b)       => API._j(`/api/projects/${pid}/merge`, { method: "POST", body: JSON.stringify(b) }),
  async augPreview(pid, body) {
    const r = await fetch(`/api/projects/${pid}/augment/preview`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json()).detail || "preview failed");
    return URL.createObjectURL(await r.blob());
  },

  // training
  jobs:     (pid)              => API._j(`/api/projects/${pid}/train`),
  train:    (pid, b)           => API._j(`/api/projects/${pid}/train`, { method: "POST", body: JSON.stringify(b) }),
  jobLog:   (pid, jid, lines)  => API._j(`/api/projects/${pid}/train/${jid}/log${lines ? `?lines=${lines}` : ""}`),
  stopJob:  (pid, jid)         => API._j(`/api/projects/${pid}/train/${jid}/stop`, { method: "POST" }),
  deleteJob: (pid, jid)        => API._j(`/api/projects/${pid}/train/${jid}`, { method: "DELETE" }),
};
