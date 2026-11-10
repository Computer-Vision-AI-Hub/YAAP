"""Schema upgrades (backend/db.py init_db): an existing database gains new tables and
columns on startup, and MIGRATIONS run exactly once each. Every test uses its own
SQLite file, never the test suite's database."""
from sqlalchemy import create_engine, inspect, text

from backend import db as dbmod


def _engine(tmp_path, name="yaap.db"):
    return create_engine(f"sqlite:///{tmp_path / name}")


def _version(eng):
    with eng.connect() as c:
        return c.execute(text("PRAGMA user_version")).scalar()


def test_an_old_database_gains_new_tables_and_columns(tmp_path):
    eng = _engine(tmp_path)
    with eng.begin() as c:              # a database from before most columns existed
        c.execute(text("CREATE TABLE projects (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL)"))
        c.execute(text("INSERT INTO projects (name) VALUES ('old')"))
    dbmod.init_db(eng)
    insp = inspect(eng)
    assert {"description", "task_type", "created_at"} <= {c["name"] for c in insp.get_columns("projects")}
    assert insp.has_table("bootstrap_runs") and insp.has_table("images")
    with eng.connect() as c:
        row = c.execute(text("SELECT name, description, task_type, created_at FROM projects")).one()
    assert tuple(row) == ("old", "", "detect", None)     # plain defaults fill old rows; now() can't
    assert _version(eng) == len(dbmod.MIGRATIONS)


def test_a_new_database_needs_no_migrations(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(dbmod, "MIGRATIONS", [lambda conn: ran.append(1)])
    eng = _engine(tmp_path)
    dbmod.init_db(eng)
    assert ran == [] and _version(eng) == 1


def test_pending_migrations_run_once_in_order(tmp_path, monkeypatch):
    ran = []
    eng = _engine(tmp_path)
    monkeypatch.setattr(dbmod, "MIGRATIONS", [lambda c: ran.append("a")])
    dbmod.init_db(eng)                                    # fresh: "a" is already reflected
    monkeypatch.setattr(dbmod, "MIGRATIONS", [lambda c: ran.append("a"), lambda c: ran.append("b"),
                                              lambda c: ran.append("c")])
    dbmod.init_db(eng)
    dbmod.init_db(eng)                                    # second start: nothing left to run
    assert ran == ["b", "c"] and _version(eng) == 3


def test_a_failing_migration_changes_nothing(tmp_path, monkeypatch):
    import pytest
    eng = _engine(tmp_path)
    monkeypatch.setattr(dbmod, "MIGRATIONS", [])
    dbmod.init_db(eng)

    def broken(conn):
        conn.execute(text("UPDATE projects SET name = 'x'"))
        raise RuntimeError("boom")
    monkeypatch.setattr(dbmod, "MIGRATIONS", [broken])
    with pytest.raises(RuntimeError):
        dbmod.init_db(eng)
    assert _version(eng) == 0                             # will be retried on the next start


def test_first_migration_resets_images_left_without_predictions(tmp_path):
    eng = _engine(tmp_path)
    dbmod.init_db(eng)
    with eng.begin() as c:
        c.execute(text("INSERT INTO projects (id, name) VALUES (1, 'p')"))
        c.execute(text("INSERT INTO images (id, project_id, filename, status) VALUES "
                       "(1, 1, 'a.png', 'auto-annotated'), (2, 1, 'b.png', 'auto-annotated')"))
        c.execute(text("INSERT INTO label_classes (id, project_id, name) VALUES (1, 1, 'cat')"))
        c.execute(text("INSERT INTO annotations (image_id, class_id, kind, data) VALUES (2, 1, 'bbox', '{}')"))
        c.execute(text("PRAGMA user_version = 0"))      # as if it came from before the migration
    dbmod.init_db(eng)
    with eng.connect() as c:
        rows = dict(c.execute(text("SELECT id, status FROM images")).all())
    assert rows == {1: "unannotated", 2: "auto-annotated"}
