"""Pydantic request/response schemas."""
import math
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    task_type: str = Field("detect", pattern="^(detect|segment|classify)$")


class ClassCreate(BaseModel):
    name: str
    color: Optional[str] = None


class ClassUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None


class AnnotationIn(BaseModel):
    class_id: int
    kind: str = Field(pattern="^(bbox|polygon|classification)$")
    data: dict[str, Any] = {}
    source: str = "manual"
    confidence: float = 1.0
    origin: str = Field("", max_length=32)

    @model_validator(mode="after")
    def _geometry(self):
        """Refuse shapes no exporter can write (a bbox without x/y/w/h broke the whole project's export)."""
        d = self.data or {}
        if self.kind == "bbox":
            try:
                x, y, w, h = (float(d[k]) for k in ("x", "y", "w", "h"))
            except (KeyError, TypeError, ValueError):
                raise ValueError("a bbox needs numeric x, y, w and h")
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError("a bbox needs finite x, y and w, h > 0")
            self.data = {"x": x, "y": y, "w": w, "h": h}
        elif self.kind == "polygon":
            pts = d.get("points")
            try:
                pts = [[float(q[0]), float(q[1])] for q in pts]
            except (TypeError, ValueError, IndexError):
                raise ValueError("polygon points must be [x, y] pairs")
            if len(pts) < 3 or not all(math.isfinite(v) for q in pts for v in q):
                raise ValueError("a polygon needs at least 3 finite points")
            self.data = {"points": pts}
        else:
            self.data = {}
        return self


class AnnotationsReplace(BaseModel):
    annotations: list[AnnotationIn]
    # the ids the client last loaded/saved: 409 if the image now holds others (a job wrote predictions,
    # or another tab saved), so a stale editor can't silently delete them. None = no check.
    expect_ids: Optional[list[int]] = None


class ImagePatch(BaseModel):
    status: Optional[str] = None
    split: Optional[str] = None
    expect_annotations: Optional[list[int]] = None   # 409 unless these are still the image's annotations


class InferenceRequest(BaseModel):
    model_name: str                 # alias ("yolo11n.pt"), weight id ("weight:3") or "rfdetr-base"
    image_ids: list[int] = []       # empty = all images in project
    conf: float = 0.4
    iou: float = 0.5
    device: str = "auto"            # auto | cpu | cuda:0 ...
    replace_existing: bool = False
    create_missing_classes: bool = False   # off: predictions of classes the project doesn't have are dropped
    class_filter: list[str] = []    # keep only these model class names, empty = all
    # one box per object: any-class overlap at/above this IoU = the same object (1 = off) — jobs.drop_duplicates
    max_overlap: float = Field(0.7, gt=0, le=1)


class TestModelRequest(BaseModel):
    version_id: int
    model_name: str
    conf: float = 0.4
    iou: float = 0.5
    device: str = "auto"
    limit: int = 20
    offset: int = 0


class AugmentPreviewRequest(BaseModel):
    image_id: int
    ops: dict[str, Any] = {}
    count: int = 4


class VersionCreate(BaseModel):
    name: str = ""
    format: str = Field("yolo", pattern="^(yolo|coco)$")   # coco: detect/segment only
    splits: dict[str, float] = {"train": 0.7, "val": 0.2, "test": 0.1}
    preprocess: dict[str, Any] = {"resize": 640, "mode": "stretch", "grayscale": False}
    augment: dict[str, Any] = {"multiplier": 1, "ops": {}}
    include_unannotated: bool = False
    approved_only: bool = True


class MergeRequest(BaseModel):
    source_project_id: int
    default_split: str = ""


class TrainRequest(BaseModel):
    version_id: int
    model_arch: str = "yolov8n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "auto"
    patience: int = 10
    extra: dict[str, Any] = {}
    mode: str = Field("local", pattern="^(local|ssh)$")


class SshSettings(BaseModel):
    host: str = ""
    user: str = ""
    port: int = 22
    identity_file: str = ""
    python: str = "~/yaap-env/bin/python"
    workdir: str = "~/yaap_jobs"
    cleanup: bool = False


class BootstrapRequest(BaseModel):
    model_arch: str = ""                       # empty = yolo11n for the project's task
    epochs: int = Field(0, ge=0, le=500)       # 0 = auto: more passes for fewer images
    imgsz: int = Field(0, ge=0, le=2048)       # 0 = the task's default: 640, or 224 for classification
    conf: float = Field(0.4, gt=0, lt=1)
    iou: float = Field(0.5, gt=0, lt=1)
    device: str = "auto"
    refresh_unreviewed: bool = True            # also re-label images holding only unreviewed predictions
    relabel_rejected: bool = True              # … and rejected ones whose boxes all came from a model
    relabel_other_tools: bool = False          # … even when those predictions came from SAM / Run a model
    max_overlap: float = Field(0.7, gt=0, le=1)   # one box per object, see InferenceRequest
