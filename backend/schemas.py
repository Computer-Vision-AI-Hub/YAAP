"""Pydantic request/response schemas."""
from typing import Any, Optional

from pydantic import BaseModel, Field


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


class AnnotationsReplace(BaseModel):
    annotations: list[AnnotationIn]


class ImagePatch(BaseModel):
    status: Optional[str] = None
    split: Optional[str] = None


class InferenceRequest(BaseModel):
    model_name: str                 # alias ("yolo11n.pt"), weight id ("weight:3") or "rfdetr-base"
    image_ids: list[int] = []       # empty = all images in project
    conf: float = 0.4
    iou: float = 0.5
    device: str = "auto"            # auto | cpu | cuda:0 ...
    replace_existing: bool = False
    create_missing_classes: bool = True
    class_filter: list[str] = []    # keep only these model class names, empty = all


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
