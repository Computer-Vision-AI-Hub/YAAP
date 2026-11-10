"""Database layer — SQLite + SQLAlchemy 2.0."""
import datetime as dt
import json

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                        Text, create_engine, event, inspect, text)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from .config import DB_PATH

# timeout: how long a write waits for another writer's lock (sqlite3's default of 5 s was too
# short with background jobs, the editor and Review writing at once → "database is locked").
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False, "timeout": 30})


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):
    """WAL: readers (job polling, image lists) never block a writer's commit and vice versa."""
    cur = dbapi_conn.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=30000")
    finally:
        cur.close()

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
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    color = Column(String, default="#FFC53D")
    order_idx = Column(Integer, default=0)

    project = relationship("Project", back_populates="classes")



def next_class_index(db, project_id: int) -> int:
    """order_idx for a new class: after the highest one. (len(classes) repeated an index once a
    class had been deleted — two classes then shared one YOLO export index.)"""
    from sqlalchemy import func
    top = db.query(func.max(LabelClass.order_idx)).filter(LabelClass.project_id == project_id).scalar()
    return 0 if top is None else top + 1


class ImageAsset(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    # unannotated | annotated | auto-annotated | review | needs_reannotation | empty | approved
    # — meanings and transitions in DOCUMENTATION.md §6.3
    status = Column(String, default="unannotated")
    split = Column(String, default="")              # optional manual split pin
    created_at = Column(DateTime, default=now)

    project = relationship("Project", back_populates="images")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")
    score = relationship("ImageScore", uselist=False, cascade="all, delete-orphan")
    proposals = relationship("Proposal", cascade="all, delete-orphan")


class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False, index=True)
    class_id = Column(Integer, ForeignKey("label_classes.id"), nullable=False)
    kind = Column(String, nullable=False)   # bbox | polygon | classification
    data = Column(Text, default="{}")       # bbox: {x,y,w,h} px  polygon: {points:[[x,y],..]}
    source = Column(String, default="manual")  # manual | model
    confidence = Column(Float, default=1.0)
    # which tool made a model annotation: sam | autolabel | bootstrap ("" = a person, or made before this
    # was recorded). Bootstrap only replaces its own earlier guesses unless asked to (relabel_other_tools).
    origin = Column(String, default="")

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


class Proposal(Base):
    """A class-less object outline from SAM's generic search — waits for a human to
    give it a class in the editor (then it becomes a normal Annotation)."""
    __tablename__ = "proposals"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False, index=True)
    data = Column(Text, default="{}")        # {"polygon": [[x,y]..], "bbox": {x,y,w,h}} in pixels
    score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=now)


class ImageScore(Base):
    """Active-learning uncertainty (0 sure … 1 torn) from the last scoring run."""
    __tablename__ = "image_scores"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False, unique=True)
    uncertainty = Column(Float, default=0.0)
    model_name = Column(String, default="")
    created_at = Column(DateTime, default=now)


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
    labeled_image_ids = Column(Text, default="[]")  # JSON list — images this job touched
    log = Column(Text, default="")                  # newline-joined progress lines
    error = Column(Text, default="")
    created_at = Column(DateTime, default=now)


class BootstrapRun(Base):
    """One Bootstrap round: a quick model trained on the human-labeled images, then
    run over the rest (services/bootstrap.py). Training is a TrainJob, the labeling
    pass an AutolabelJob — this row ties them together and carries the progress."""
    __tablename__ = "bootstrap_runs"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    round = Column(Integer, default=1)
    status = Column(String, default="training")    # training | labeling | done | failed | stopped
    config = Column(Text, default="{}")             # the BootstrapRequest body
    model_arch = Column(String, default="")
    epochs = Column(Integer, default=0)
    epoch = Column(Integer, default=0)              # last finished epoch
    train_images = Column(Integer, default=0)       # labeled images trained on (background not counted)
    target_images = Column(Integer, default=0)      # images handed to the labeling pass
    train_job_id = Column(Integer, default=0)
    autolabel_job_id = Column(Integer, default=0)
    weight_id = Column(Integer, default=0)
    message = Column(Text, default="")
    error = Column(Text, default="")
    created_at = Column(DateTime, default=now)
    finished_at = Column(DateTime, nullable=True)


# ── schema upgrades ──────────────────────────────────────────────────
# New tables and new columns need no code: init_db() creates the tables and adds every
# column a model has gained (SQLite's ALTER TABLE ... ADD COLUMN). Anything that can't
# be said that way — a data fix, a rename, a new constraint — goes into MIGRATIONS:
# append a function taking a Connection; each runs once per database, in order, and
# PRAGMA user_version records how many have run. Never edit or reorder old entries.

def _m1_reset_empty_predictions(conn):
    """Before the fix to 'replace existing', an auto-label run that found nothing could
    leave an image 'auto-annotated' with no annotations left at all."""
    conn.execute(text("UPDATE images SET status = 'unannotated' WHERE status = 'auto-annotated' "
                      "AND NOT EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = images.id)"))


MIGRATIONS = [_m1_reset_empty_predictions]


def _sql_default(col) -> str | None:
    """A column's plain default as an SQL literal, so rows that exist before the column
    get it too. Computed defaults (e.g. now()) can't be expressed: those rows get NULL."""
    d = col.default
    if d is None or not d.is_scalar:
        return None
    v = d.arg
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return None


def _add_missing_columns(conn):
    insp = inspect(conn)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in have:
                continue
            if col.primary_key or col.unique:
                raise RuntimeError(f"{table.name}.{col.name}: SQLite can't add a primary-key or unique "
                                   "column to an existing table — write a migration in db.MIGRATIONS.")
            ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col.type.compile(dialect=conn.dialect)}'
            default = _sql_default(col)
            if default is not None:
                ddl += f" DEFAULT {default}"
            conn.execute(text(ddl))


def init_db(bind=None):
    """Create missing tables, add missing columns and indexes, then run pending MIGRATIONS.
    Safe to call on every start (the server and each training worker do)."""
    with (bind or engine).begin() as conn:
        fresh = not inspect(conn).get_table_names()
        Base.metadata.create_all(conn)
        _add_missing_columns(conn)
        # create_all() only builds indexes for brand-new tables
        for table in Base.metadata.sorted_tables:
            for idx in table.indexes:
                idx.create(conn, checkfirst=True)
        done = conn.execute(text("PRAGMA user_version")).scalar()
        if not fresh:                    # a brand-new database is already in the final shape
            for step in MIGRATIONS[done:]:
                step(conn)
        if done < len(MIGRATIONS):       # (a database from a newer YAAP keeps its higher number)
            conn.execute(text(f"PRAGMA user_version = {len(MIGRATIONS)}"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
