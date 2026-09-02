/* YAAP canvas editor — bbox + polygon drawing, select/move/resize, pan/zoom.
   All annotation coordinates are stored in IMAGE PIXELS; the view transform
   is applied only at draw time. */

class Editor {
  constructor(canvas, opts) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.opts = opts;                 // {classes, getActiveClass, onChange, onSelect}
    this.img = null;
    this.anns = [];                   // {class_id, kind, data, source, confidence}
    this.sel = -1;
    this.tool = "bbox";
    this.view = { s: 1, tx: 0, ty: 0 };
    this.mouse = { x: 0, y: 0, ix: 0, iy: 0, down: false };
    this.draft = null;                // bbox draft {x0,y0,x1,y1}
    this.poly = null;                 // in-progress polygon points
    this.drag = null;                 // {mode:'move'|'handle'|'vertex'|'pan', ...}
    this.spaceHeld = false;
    // SAM prompt state (image coords)
    this.samPts = [];                 // [[x, y, label(1|0)], ...]
    this.samBox = null;               // {x0,y0,x1,y1} while dragging / committed
    this.samPending = [];             // [{polygon:[[x,y]..], score, label}]
    this._samDown = null;
    this._bind();
  }

  /* ── lifecycle ─────────────────────────────────────────────────── */
  load(imgEl, anns) {
    this.img = imgEl;
    this.anns = anns.map(a => ({ ...a }));
    this.sel = -1; this.draft = null; this.poly = null; this.drag = null;
    this.clearSam(false);
    this.fit();
  }

  fit() {
    if (!this.img) return;
    const r = this.cv.getBoundingClientRect();
    this.cv.width = r.width * devicePixelRatio;
    this.cv.height = r.height * devicePixelRatio;
    const s = Math.min(r.width / this.img.width, r.height / this.img.height) * 0.94;
    this.view.s = s;
    this.view.tx = (r.width - this.img.width * s) / 2;
    this.view.ty = (r.height - this.img.height * s) / 2;
    this.draw();
  }

  setTool(t) {
    if (t !== this.tool) this.select(-1);   // switching tools ends any in-progress edit
    this.tool = t; this.poly = null; this.draft = null;
    if (t !== "sam") this.clearSam(false);
    this.draw();
  }

  clearSam(redraw = true) {
    this.samPts = []; this.samBox = null; this.samPending = []; this._samDown = null;
    this.opts.onSamState && this.opts.onSamState(this);
    if (redraw) this.draw();
  }

  /* ── coordinate helpers ────────────────────────────────────────── */
  toImg(sx, sy) { const v = this.view; return [(sx - v.tx) / v.s, (sy - v.ty) / v.s]; }
  toScr(ix, iy) { const v = this.view; return [ix * v.s + v.tx, iy * v.s + v.ty]; }

  /* ── events ────────────────────────────────────────────────────── */
  _bind() {
    const cv = this.cv;
    cv.addEventListener("wheel", e => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const f = Math.exp(-e.deltaY * 0.0012);
      const v = this.view;
      const ns = Math.min(40, Math.max(0.05, v.s * f));
      v.tx = mx - (mx - v.tx) * (ns / v.s);
      v.ty = my - (my - v.ty) * (ns / v.s);
      v.s = ns;
      this.draw();
    }, { passive: false });

    cv.addEventListener("mousedown", e => this._down(e));
    window.addEventListener("mousemove", e => this._move(e));
    window.addEventListener("mouseup", e => this._up(e));
    cv.addEventListener("dblclick", () => { if (this.poly) this._closePoly(); });
    window.addEventListener("keydown", e => this._key(e));
    window.addEventListener("keyup", e => { if (e.code === "Space") this.spaceHeld = false; });
    new ResizeObserver(() => this.fit()).observe(cv.parentElement);
  }

  _pos(e) {
    const r = this.cv.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  _down(e) {
    if (!this.img) return;
    const [sx, sy] = this._pos(e);
    const [ix, iy] = this.toImg(sx, sy);
    this.mouse.down = true;

    if (e.button === 1 || this.tool === "pan" || this.spaceHeld) {
      this.drag = { mode: "pan", sx, sy, tx: this.view.tx, ty: this.view.ty };
      return;
    }
    if (e.button !== 0) return;

    if (this.tool === "sam") {
      this._samDown = { sx, sy, ix, iy, neg: e.shiftKey };
      this.samBox = null;
      return;
    }
    if (this.tool === "bbox") {
      this.draft = { x0: ix, y0: iy, x1: ix, y1: iy };
    } else if (this.tool === "polygon") {
      if (!this.poly) this.poly = [];
      // close when clicking near first vertex
      if (this.poly.length >= 3) {
        const [fx, fy] = this.toScr(this.poly[0][0], this.poly[0][1]);
        if (Math.hypot(sx - fx, sy - fy) < 12) { this._closePoly(); return; }
      }
      this.poly.push([this._clampX(ix), this._clampY(iy)]);
    } else if (this.tool === "select") {
      // 1) handle of selected bbox / vertex of selected polygon
      if (this.sel >= 0) {
        const h = this._hitHandle(sx, sy);
        if (h) {
          this.drag = h;
          if (h.mode === "vertex") Object.assign(this.drag, { downSx: sx, downSy: sy, moved: false });
          return;
        }
        // 1b) click on an edge of the selected polygon → insert a vertex there,
        // and let an immediate drag (no mouseup yet) carry it right away
        const a = this.anns[this.sel];
        if (a && a.kind === "polygon") {
          const at = this._hitEdge(sx, sy);
          if (at !== null) {
            a.data.points.splice(at, 0, [this._clampX(ix), this._clampY(iy)]);
            this.drag = { mode: "vertex", i: at, downSx: sx, downSy: sy, moved: false, inserted: true };
            this._dirty();
            this.draw();
            return;
          }
        }
      }
      // 2) annotation body (topmost first)
      const hit = this._hitAnn(ix, iy);
      this.select(hit);
      if (hit >= 0) {
        const a = this.anns[hit];
        this.drag = { mode: "move", ix, iy, orig: JSON.parse(JSON.stringify(a.data)) };
      }
    }
    this.draw();
  }

  _move(e) {
    const [sx, sy] = this._pos(e);
    const [ix, iy] = this.toImg(sx, sy);
    Object.assign(this.mouse, { x: sx, y: sy, ix, iy });

    if (this.drag) {
      const d = this.drag, v = this.view;
      if (d.mode === "pan") { v.tx = d.tx + (sx - d.sx); v.ty = d.ty + (sy - d.sy); }
      else if (d.mode === "move") {
        const a = this.anns[this.sel], dx = ix - d.ix, dy = iy - d.iy;
        if (a.kind === "bbox") {
          a.data.x = d.orig.x + dx; a.data.y = d.orig.y + dy;
        } else if (a.kind === "polygon") {
          a.data.points = d.orig.points.map(p => [p[0] + dx, p[1] + dy]);
        }
        this._dirty();
      } else if (d.mode === "handle") {
        this._resizeBBox(this.anns[this.sel].data, d.h, ix, iy); this._dirty();
      } else if (d.mode === "vertex") {
        if (!d.moved && Math.hypot(sx - d.downSx, sy - d.downSy) > 4) d.moved = true;
        this.anns[this.sel].data.points[d.i] = [this._clampX(ix), this._clampY(iy)]; this._dirty();
      }
    } else if (this.draft) {
      this.draft.x1 = ix; this.draft.y1 = iy;
    } else if (this._samDown && this.mouse.down) {
      const d = this._samDown;
      if (Math.hypot(sx - d.sx, sy - d.sy) > 6)
        this.samBox = { x0: d.ix, y0: d.iy, x1: ix, y1: iy };
    }
    this.draw();
  }

  _up() {
    this.mouse.down = false;
    if (this._samDown) {
      const d = this._samDown; this._samDown = null;
      if (this.samBox) {                                   // drag → box prompt
        const b = this.samBox;
        this.samBox = { x0: Math.min(b.x0, b.x1), y0: Math.min(b.y0, b.y1),
                        x1: Math.max(b.x0, b.x1), y1: Math.max(b.y0, b.y1) };
        this.samPts = [];
      } else {                                             // click → point prompt
        this.samPts.push([this._clampX(d.ix), this._clampY(d.iy), d.neg ? 0 : 1]);
      }
      this.opts.onSamPrompt && this.opts.onSamPrompt(this);
      this.draw();
      return;
    }
    if (this.draft) {
      const d = this.draft;
      const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
      const w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
      this.draft = null;
      if (w > 4 && h > 4) {
        const cls = this.opts.getActiveClass();
        if (cls) {
          this.anns.push({ class_id: cls.id, kind: "bbox", source: "manual", confidence: 1,
            data: { x: this._clampX(x), y: this._clampY(y),
                    w: Math.min(w, this.img.width - x), h: Math.min(h, this.img.height - y) } });
          this.select(this.anns.length - 1);
          this._dirty();
        }
      }
    }
    if (this.drag && this.drag.mode === "vertex" && !this.drag.moved) {
      // plain click on a pre-existing vertex (no drag) → remove it, never below a
      // triangle; a freshly-inserted vertex (edge click) is left in place instead
      if (!this.drag.inserted) {
        const a = this.anns[this.sel];
        if (a && a.data.points.length > 3) { a.data.points.splice(this.drag.i, 1); this._dirty(); }
      }
      this.drag = null;
      this.draw();
      return;
    }
    if (this.drag && this.drag.mode !== "pan") this._normalizeSel();
    this.drag = null;
    this.draw();
  }

  _key(e) {
    if (e.target.matches("input,textarea,select") || !this.img) return;
    const k = e.key.toLowerCase();
    if (k === "v") this.opts.onToolKey("select");
    else if (k === "b") this.opts.onToolKey("bbox");
    else if (k === "p") this.opts.onToolKey("polygon");
    else if (k === "h") this.opts.onToolKey("pan");
    else if (k === "s") this.opts.onToolKey("sam");
    else if (k === "f") this.fit();
    else if (e.code === "Space") { this.spaceHeld = true; e.preventDefault(); }
    else if (k === "enter" && this.poly) this._closePoly();
    else if (k === "enter" && this.samPending.length) { this.opts.onSamAccept && this.opts.onSamAccept(); }
    else if (k === "escape") {
      if (this.samPts.length || this.samBox || this.samPending.length) this.clearSam();
      else { this.poly = null; this.draft = null; this.select(-1); }
    }
    else if ((k === "delete" || k === "backspace") && this.sel >= 0) {
      this.anns.splice(this.sel, 1); this.select(-1); this._dirty();
    } else if (/^[1-9]$/.test(k)) this.opts.onClassKey(+k - 1);
    this.draw();
  }

  _closePoly() {
    if (this.poly && this.poly.length >= 3) {
      const cls = this.opts.getActiveClass();
      if (cls) {
        this.anns.push({ class_id: cls.id, kind: "polygon", source: "manual",
          confidence: 1, data: { points: this.poly } });
        this.select(-1);           // labeled and done — don't leave it selected/editable
        this._dirty();
      }
    }
    this.poly = null;
    this.draw();
  }

  /* ── hit testing / geometry ────────────────────────────────────── */
  _hitAnn(ix, iy) {
    for (let i = this.anns.length - 1; i >= 0; i--) {
      const a = this.anns[i];
      if (a.kind === "bbox") {
        const d = a.data;
        if (ix >= d.x && ix <= d.x + d.w && iy >= d.y && iy <= d.y + d.h) return i;
      } else if (a.kind === "polygon" && this._inPoly(ix, iy, a.data.points)) return i;
    }
    return -1;
  }

  _inPoly(x, y, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  _handles(d) {
    return [["nw", d.x, d.y], ["n", d.x + d.w / 2, d.y], ["ne", d.x + d.w, d.y],
            ["e", d.x + d.w, d.y + d.h / 2], ["se", d.x + d.w, d.y + d.h],
            ["s", d.x + d.w / 2, d.y + d.h], ["sw", d.x, d.y + d.h], ["w", d.x, d.y + d.h / 2]];
  }

  _hitHandle(sx, sy) {
    const a = this.anns[this.sel];
    if (!a) return null;
    if (a.kind === "bbox") {
      for (const [h, hx, hy] of this._handles(a.data)) {
        const [px, py] = this.toScr(hx, hy);
        if (Math.hypot(sx - px, sy - py) < 8) return { mode: "handle", h };
      }
    } else if (a.kind === "polygon") {
      for (let i = 0; i < a.data.points.length; i++) {
        const [px, py] = this.toScr(a.data.points[i][0], a.data.points[i][1]);
        if (Math.hypot(sx - px, sy - py) < 8) return { mode: "vertex", i };
      }
    }
    return null;
  }

  // Closest point on the selected polygon's boundary to (sx,sy), screen space.
  // Returns the array index to splice a new vertex into (i.e. right after
  // vertex `i`), or null if no edge is close enough.
  _hitEdge(sx, sy) {
    const a = this.anns[this.sel];
    if (!a || a.kind !== "polygon") return null;
    const pts = a.data.points;
    for (let i = 0; i < pts.length; i++) {
      const [x1, y1] = this.toScr(pts[i][0], pts[i][1]);
      const [x2, y2] = this.toScr(...pts[(i + 1) % pts.length]);
      if (this._distToSegment(sx, sy, x1, y1, x2, y2) < 7) return i + 1;
    }
    return null;
  }

  _distToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    let t = lenSq ? ((px - x1) * dx + (py - y1) * dy) / lenSq : 0;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  }

  _resizeBBox(d, h, ix, iy) {
    const x2 = d.x + d.w, y2 = d.y + d.h;
    if (h.includes("w")) { d.w = x2 - ix; d.x = ix; }
    if (h.includes("e")) { d.w = ix - d.x; }
    if (h.includes("n")) { d.h = y2 - iy; d.y = iy; }
    if (h.includes("s")) { d.h = iy - d.y; }
  }

  _normalizeSel() {
    const a = this.anns[this.sel];
    if (!a) return;
    if (a.kind === "bbox") {
      const d = a.data;
      if (d.w < 0) { d.x += d.w; d.w = -d.w; }
      if (d.h < 0) { d.y += d.h; d.h = -d.h; }
      d.x = this._clampX(d.x); d.y = this._clampY(d.y);
      d.w = Math.max(2, Math.min(d.w, this.img.width - d.x));
      d.h = Math.max(2, Math.min(d.h, this.img.height - d.y));
    }
  }

  _clampX(x) { return Math.min(Math.max(x, 0), this.img.width); }
  _clampY(y) { return Math.min(Math.max(y, 0), this.img.height); }

  select(i) { this.sel = i; this.opts.onSelect && this.opts.onSelect(i); }
  _dirty() { this.opts.onChange && this.opts.onChange(); }

  /* ── rendering ─────────────────────────────────────────────────── */
  classOf(a) { return this.opts.classes().find(c => c.id === a.class_id); }

  draw() {
    const ctx = this.ctx, v = this.view;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, this.cv.width, this.cv.height);
    if (!this.img) return;

    ctx.save();
    ctx.translate(v.tx, v.ty); ctx.scale(v.s, v.s);
    ctx.imageSmoothingEnabled = v.s < 3;
    ctx.drawImage(this.img, 0, 0);
    ctx.restore();

    // annotations (drawn in screen space for crisp 1px lines)
    // while actively drawing something new, hide labels (they clutter the
    // image) except on the annotation currently under the pointer
    const hideTags = this.tool === "sam" || this.tool === "polygon" || this.tool === "bbox";
    const hoverIdx = hideTags ? this._hitAnn(this.mouse.ix, this.mouse.iy) : -1;
    // number same-named labels — "cat (1)", "cat (2)" — only when there's more than one
    const classCounts = {};
    this.anns.forEach(a => { classCounts[a.class_id] = (classCounts[a.class_id] || 0) + 1; });
    const classSeen = {};
    this.anns.forEach((a, i) => {
      let idx = null;
      if (classCounts[a.class_id] > 1) idx = classSeen[a.class_id] = (classSeen[a.class_id] || 0) + 1;
      this._drawAnn(a, i === this.sel, idx, hideTags && i !== hoverIdx);
    });
    if (this.draft) this._drawDraft();
    if (this.poly) this._drawPolyDraft();
    this._drawSam();
    if ((this.tool === "bbox" || this.tool === "polygon" ||
         (this.tool === "sam" && !this.samPending.length)) && !this.mouse.down) this._drawCrosshair();
  }

  _drawAnn(a, selected, idx, hideTag) {
    const ctx = this.ctx, cls = this.classOf(a);
    const color = cls ? cls.color : "#999";
    ctx.lineWidth = selected ? 2.5 : 1.6;
    ctx.strokeStyle = color;
    ctx.fillStyle = color + "26";

    if (a.kind === "bbox") {
      const [x, y] = this.toScr(a.data.x, a.data.y);
      const w = a.data.w * this.view.s, h = a.data.h * this.view.s;
      ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
      if (!hideTag) this._tag(x, y, cls, a, idx);
      if (selected) for (const [, hx, hy] of this._handles(a.data)) this._knob(...this.toScr(hx, hy), color);
    } else if (a.kind === "polygon") {
      const pts = a.data.points.map(p => this.toScr(p[0], p[1]));
      ctx.beginPath();
      pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.closePath(); ctx.fill(); ctx.stroke();
      if (!hideTag) this._tag(pts[0][0], pts[0][1], cls, a, idx);
      if (selected) pts.forEach(([x, y]) => this._knob(x, y, color));
    }
  }

  _tag(x, y, cls, a, idx) {
    if (!cls || this.view.s * this.img.width < 120) return;
    const ctx = this.ctx;
    const label = cls.name + (idx ? ` (${idx})` : "") +
      (a.source === "model" ? ` ·${Math.round(a.confidence * 100)}%` : "");
    ctx.font = "10px 'IBM Plex Mono', monospace";
    const w = ctx.measureText(label).width + 10;
    ctx.fillStyle = cls.color;
    ctx.fillRect(x, y - 16, w, 15);
    ctx.fillStyle = "#10131a";
    ctx.fillText(label, x + 5, y - 5);
  }

  _knob(x, y, color) {
    const ctx = this.ctx;
    ctx.fillStyle = "#0C0F13"; ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill(); ctx.stroke();
  }

  _drawDraft() {
    const ctx = this.ctx, d = this.draft;
    const [x0, y0] = this.toScr(d.x0, d.y0), [x1, y1] = this.toScr(d.x1, d.y1);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "#FFD23F"; ctx.lineWidth = 1.5;
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.setLineDash([]);
  }

  _drawPolyDraft() {
    const ctx = this.ctx;
    const pts = this.poly.map(p => this.toScr(p[0], p[1]));
    ctx.strokeStyle = "#FFD23F"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.lineTo(this.mouse.x, this.mouse.y);
    ctx.stroke();
    pts.forEach(([x, y], i) => this._knob(x, y, i === 0 ? "#22D3C7" : "#FFD23F"));
  }

  _drawSam() {
    const ctx = this.ctx;
    // pending masks — teal, dashed, breathing fill
    for (const r of this.samPending) {
      const pts = r.polygon.map(p => this.toScr(p[0], p[1]));
      ctx.beginPath();
      pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.closePath();
      ctx.fillStyle = "rgba(34,211,199,.22)"; ctx.fill();
      ctx.setLineDash([6, 4]); ctx.strokeStyle = "#22D3C7"; ctx.lineWidth = 2; ctx.stroke();
      ctx.setLineDash([]);
      if (r.label || r.score < 1) {
        const [tx, ty] = pts[0];
        ctx.font = "10px 'IBM Plex Mono', monospace";
        const t = `${r.label || "mask"} ${Math.round(r.score * 100)}%`;
        ctx.fillStyle = "#22D3C7"; ctx.fillRect(tx, ty - 15, ctx.measureText(t).width + 8, 14);
        ctx.fillStyle = "#06231d"; ctx.fillText(t, tx + 4, ty - 4);
      }
    }
    // prompt box
    if (this.samBox) {
      const b = this.samBox;
      const [x0, y0] = this.toScr(b.x0, b.y0), [x1, y1] = this.toScr(b.x1, b.y1);
      this.ctx.setLineDash([5, 4]); this.ctx.strokeStyle = "#22D3C7"; this.ctx.lineWidth = 1.5;
      this.ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
      this.ctx.setLineDash([]);
    }
    // prompt points — ＋ positive (teal), − negative (red)
    for (const [px, py, lab] of this.samPts) {
      const [x, y] = this.toScr(px, py);
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 7);
      ctx.fillStyle = lab ? "#22D3C7" : "#F0555C"; ctx.fill();
      ctx.strokeStyle = "#0C0F13"; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.strokeStyle = "#0C0F13"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x - 3, y); ctx.lineTo(x + 3, y);
      if (lab) { ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3); }
      ctx.stroke();
    }
  }

  _drawCrosshair() {
    const ctx = this.ctx, r = this.cv.getBoundingClientRect();
    ctx.strokeStyle = "rgba(255,210,63,.75)"; ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(this.mouse.x, 0); ctx.lineTo(this.mouse.x, r.height);
    ctx.moveTo(0, this.mouse.y); ctx.lineTo(r.width, this.mouse.y);
    ctx.stroke(); ctx.setLineDash([]);
  }
}
