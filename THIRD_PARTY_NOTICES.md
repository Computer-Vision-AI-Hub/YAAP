# Third-Party Licenses

YAAP's own source code is MIT-licensed (see [LICENSE](LICENSE)). That license
covers the code in this repository — it does **not** change the license
terms of the third-party packages YAAP depends on. Some of those carry
obligations of their own, summarized below so you know what you're agreeing
to before you deploy or build a business on top of this project.

## The one that actually matters: Ultralytics

YAAP's detection/segmentation backbone (YOLO, RT-DETR, and the SAM/SAM3
wrapper) is provided by the [`ultralytics`](https://github.com/ultralytics/ultralytics)
package, licensed **AGPL-3.0**.

Ultralytics' own published policy states:

> "Any usage of Ultralytics models, including in an R&D setup within a
> company, requires an Ultralytics Enterprise License unless the entire
> project is open-sourced under the AGPL-3.0 license. This applies
> regardless of whether the usage is commercial or not."
> — [ultralytics.com/license](https://www.ultralytics.com/license)

**What this means in practice:**

- **Running YAAP yourself, for your own research or internal use** — fine.
  This is the normal case AGPL is designed for.
- **Distributing a modified copy of YAAP, or running it as a network
  service for others** (including a paid dataset-labeling or model-training
  service) — under Ultralytics' stated policy, this requires either your
  entire offering to also be open-sourced under AGPL-3.0, or an
  [Ultralytics Enterprise License](https://www.ultralytics.com/license).

If you plan to use YAAP commercially beyond your own internal use, budget
for an Enterprise License or get your own legal advice on your specific
setup — this is not something YAAP's own MIT license can override.

## Meta SAM 3

SAM/SAM3 model weights (downloaded separately, not bundled in this repo) are
distributed by Meta under a custom
["SAM License"](https://github.com/facebookresearch/sam3/blob/main/LICENSE).
Key points:

- Commercial use is explicitly permitted, royalty-free.
- If you redistribute the weights themselves, you must include a copy of
  the license text.
- Prohibited fields of use: military/warfare, nuclear applications,
  espionage, weapons development.
- Attribution is required if you publish research using SAM.

## RF-DETR (optional, disabled by default)

The optional [`rfdetr`](https://github.com/roboflow/rf-detr) package
(commented out in `requirements.txt`) is **Apache-2.0** — permissive, no
commercial restriction. Note this refers only to the base/large models YAAP
integrates, not Roboflow's separately-licensed "RF-DETR+" product line.

## Everything else

All other dependencies (FastAPI, SQLAlchemy, Pydantic, PyTorch, torchvision,
Pillow, OpenCV, NumPy, Albumentations, PyYAML, uvicorn, python-multipart,
aiofiles, timm) use standard permissive licenses (MIT / BSD-3-Clause /
Apache-2.0 / HPND) with no copyleft or commercial restrictions.

---

*This document is a technical summary for convenience, not legal advice.
Consult a licensing attorney before entering commercial agreements built on
this platform.*
