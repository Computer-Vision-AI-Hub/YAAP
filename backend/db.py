"""Database layer — SQLite + SQLAlchemy 2.0."""
import datetime as dt
import json

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                        Text, create_engine)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from .config import DB_PATH

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def now():
    return dt.datetime.utcnow()


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    task_type = Column(String, default="detect")  # detect | segment | classify
    created_at = Column(DateTime, default=now)

    images = relationship("ImageAsset", back_populates="project", cascade="all, delete-orphan")
    classes = relationship("LabelClass", back_populates="project", cascade="all, delete-orphan",
                           order_by="LabelClass.order_idx")
    versions = relationship("DatasetVersion", back_populates="project", cascade="all, delete-orphan")


class LabelClass(Base):
    __tablename__ = "label_classes"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    color = Column(String, default="#FFC53D")
    order_idx = Column(Integer, default=0)

    project = relationship("Project", back_populates="classes")


class ImageAsset(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    filename = Column(String, nullable=False)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    status = Column(String, default="unannotated")  # unannotated | annotated | review
    split = Column(String, default="")              # optional manual split pin
    created_at = Column(DateTime, default=now)

    project = relationship("Project", back_populates="images")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")


class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("label_classes.id"), nullable=False)
    kind = Column(String, nullable=False)   # bbox | polygon | classification
    data = Column(Text, default="{}")       # bbox: {x,y,w,h} px  polygon: {points:[[x,y],..]}
    source = Column(String, default="manual")  # manual | model
    confidence = Column(Float, default=1.0)

    image = relationship("ImageAsset", back_populates="annotations")
    label_class = relationship("LabelClass")

    @property
    def data_obj(self):
        return json.loads(self.data or "{}")


class DatasetVersion(Base):
    __tablename__ = "versions"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    config = Column(Text, default="{}")
    stats = Column(Text, default="{}")
    status = Column(String, default="ready")   # building | ready | failed
    created_at = Column(DateTime, default=now)

    project = relationship("Project", back_populates="versions")


class ModelWeight(Base):
    __tablename__ = "model_weights"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)  # null = global
    name = Column(String, nullable=False)
    arch = Column(String, default="")
    task = Column(String, default="detect")
    path = Column(String, nullable=False)
    source = Column(String, default="uploaded")  # uploaded | trained
    created_at = Column(DateTime, default=now)


class TrainJob(Base):
    __tablename__ = "train_jobs"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    version_id = Column(Integer, ForeignKey("versions.id"), nullable=False)
    model_arch = Column(String, nullable=False)
    params = Column(Text, default="{}")
    status = Column(String, default="queued")  # queued | running | done | failed | stopped
    pid = Column(Integer, default=0)
    log_path = Column(String, default="")
    weights_path = Column(String, default="")
    created_at = Column(DateTime, default=now)


class AutolabelJob(Base):
    __tablename__ = "autolabel_jobs"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    model_name = Column(String, nullable=False)
    status = Column(String, default="running")  # running | done | failed
    total = Column(Integer, default=0)
    processed = Column(Integer, default=0)
    images_labeled = Column(Integer, default=0)
    annotations_added = Column(Integer, default=0)
    skipped = Column(Text, default="[]")            # JSON list of filenames
    labeled_image_ids = Column(Text, default="[]")  # JSON list — images this job actually touched
    log = Column(Text, default="")                  # newline-joined progress lines
    error = Column(Text, default="")
    created_at = Column(DateTime, default=now)


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
