"""Points the whole app at a throwaway data dir before anything else imports
it, so tests never touch the real data/ directory."""
import os
import shutil
import tempfile

_TMP_DATA_DIR = tempfile.mkdtemp(prefix="yaap-test-data-")
os.environ["YAAP_DATA_DIR"] = _TMP_DATA_DIR

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.config import PROJECTS_DIR  # noqa: E402
from backend.db import Base, engine  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_db():
    """Every test gets an empty database AND an empty projects/ dir — IDs
    restart from 1 each time (DROP+CREATE resets SQLite's rowid sequence),
    so leftover files from a previous test's project 1 would otherwise
    collide with a new test's project 1."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    shutil.rmtree(PROJECTS_DIR, ignore_errors=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture
def client():
    return TestClient(app)


def make_png_bytes(w=64, h=48, color=(200, 60, 60)):
    """A tiny real PNG — save_upload() opens uploads with PIL, so arbitrary
    bytes won't do."""
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP_DATA_DIR, ignore_errors=True)
