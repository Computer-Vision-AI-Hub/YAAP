"""Dataset-version health statistics, computed from the exported folder
(labels/*.txt + data.yaml, or class folders for classify) so it also works
for versions generated before this existed."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_LABELS_PER_CLASS = 30
SMALL_AREA, LARGE_AREA = 0.01, 0.10     # fraction of image area


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _box(vals: list[float]) -> tuple[float, float] | None:
    """(w, h) normalized — from a 4-value box or a polygon's bounding box."""
    if len(vals) == 4:
        return vals[2], vals[3]
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        return max(xs) - min(xs), max(ys) - min(ys)
    return None


def compute(vdir: Path, task: str) -> dict:
    classify = task == "classify"
    names: list[str] = []
    if not classify:
        try:
            raw = yaml.safe_load((vdir / "data.yaml").read_text()).get("names", {})
            names = [raw[k] for k in sorted(raw)] if isinstance(raw, dict) else list(raw)
        except Exception:
            names = []

    split_images: dict[str, int] = {}
    split_empty: dict[str, int] = {}
    per_class: dict[str, Counter] = defaultdict(Counter)        # class -> split -> labels
    per_image_counts: list[int] = []
    sizes = Counter()
    aspects = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)             # md5 -> [split, ...]

    for split in SPLITS:
        sdir = vdir / split
        if split == "val" and not sdir.is_dir():
            sdir = vdir / "valid"                 # COCO-format versions use Roboflow's folder name
        if not sdir.is_dir():
            continue
        if classify:
            n = 0
            for cdir in sorted(d for d in sdir.iterdir() if d.is_dir()):
                files = [f for f in cdir.iterdir() if f.suffix.lower() in IMG_EXT]
                n += len(files)
                per_class[cdir.name][split] += len(files)
                for f in files:
                    hashes[_md5(f)].append(split)
            split_images[split] = n
            split_empty[split] = per_class.get("_unlabeled", Counter())[split]
            continue
        images = [f for f in (sdir / "images").glob("*") if f.suffix.lower() in IMG_EXT] \
            if (sdir / "images").is_dir() else []
        split_images[split] = len(images)
        empty = 0
        for img in images:
            hashes[_md5(img)].append(split)
            lbl = sdir / "labels" / f"{img.stem}.txt"
            lines = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()] if lbl.exists() else []
            if not lines:
                empty += 1
            per_image_counts.append(len(lines))
            for f in lines:
                try:
                    ci = int(f[0])
                    wh = _box([float(v) for v in f[1:]])
                except ValueError:
                    continue
                cname = names[ci] if ci < len(names) else f"class{ci}"
                per_class[cname][split] += 1
                if wh:
                    w, h = wh
                    area = w * h
                    sizes["small" if area < SMALL_AREA else "large" if area >= LARGE_AREA else "medium"] += 1
                    r = (w / h) if h else 1.0
                    aspects["tall" if r < 0.75 else "wide" if r > 1.33 else "square"] += 1
        split_empty[split] = empty

    for n in names:                                  # classes with zero labels still listed
        per_class.setdefault(n, Counter())
    classes = [{"name": n, "train": c["train"], "val": c["val"], "test": c["test"],
                "total": sum(c.values())} for n, c in per_class.items()
               if not (classify and n == "_unlabeled")]
    classes.sort(key=lambda c: -c["total"])

    total_images = sum(split_images.values())
    totals = [c["total"] for c in classes]
    nonzero = [t for t in totals if t]
    imbalance = round(max(nonzero) / min(nonzero), 1) if nonzero else None

    dup_within = sum(len(v) - 1 for v in hashes.values() if len(v) > 1)
    dup_cross = sum(1 for v in hashes.values() if len(set(v)) > 1)

    hist = Counter(min(c, 10) for c in per_image_counts)
    warnings: list[dict] = []
    score = 100

    def warn(level: str, msg: str, cost: int):
        nonlocal score
        warnings.append({"level": level, "msg": msg})
        score -= cost

    if total_images == 0:
        warn("error", "The version has no images.", 100)
    else:
        for s in SPLITS:
            if split_images.get(s, 0) == 0 and s != "test":
                warn("error" if s == "train" else "warn", f"The {s} split is empty.", 25 if s == "train" else 15)
        for c in classes:
            if c["total"] == 0:
                warn("error", f"Class “{c['name']}” has no labels.", 15)
                continue
            if c["total"] < MIN_LABELS_PER_CLASS:
                warn("warn", f"Class “{c['name']}” has only {c['total']} label(s) — aim for {MIN_LABELS_PER_CLASS}+.", 8)
            for s in ("val", "test"):
                if split_images.get(s, 0) and not c[s]:
                    warn("warn", f"Class “{c['name']}” has no labels in the {s} split.", 6)
        if imbalance and imbalance >= 10:
            warn("warn", f"Class imbalance is {imbalance}:1 (largest vs smallest class).", 10)
        empties = sum(split_empty.values())
        if not classify and empties / total_images > 0.5:
            warn("warn", f"{round(100 * empties / total_images)}% of images have no labels.", 10)
        if dup_cross:
            warn("error", f"{dup_cross} identical image(s) appear in more than one split (leakage).", 15)
        elif dup_within:
            warn("info", f"{dup_within} duplicate image file(s) within the same split.", 3)
        if sizes and sizes["small"] / sum(sizes.values()) > 0.6:
            warn("info", "Most objects are very small (<1% of the image) — consider a larger image size.", 3)

    return {
        "images": {s: split_images.get(s, 0) for s in SPLITS}, "empty_images": {s: split_empty.get(s, 0) for s in SPLITS},
        "total_images": total_images, "classes": classes, "imbalance_ratio": imbalance,
        "labels_total": sum(totals),
        "boxes_per_image": {"mean": round(sum(per_image_counts) / len(per_image_counts), 2) if per_image_counts else 0,
                            "max": max(per_image_counts, default=0),
                            "hist": [hist.get(i, 0) for i in range(11)]},
        "box_size": dict(sizes), "aspect": dict(aspects),
        "duplicates": {"within_split": dup_within, "cross_split": dup_cross},
        "warnings": warnings, "score": max(0, min(100, score)),
    }
