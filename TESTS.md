# YAAP tests — what they check and how they work

YAAP has two test suites:

| Suite | Where | Size | Runs in | Status (2026-11-08) |
|---|---|---|---|---|
| **Backend** (API + services) | `tests/test_*.py` | 172 tests in 15 files | pytest | all pass |
| **Frontend** (UI views) | `tests/frontend/*.test.html` | 139 checks in 6 files | headless Chrome | all pass |

Neither suite needs a GPU, model weights, an SSH host or your real `data/` folder.

---

## 1. Running them

### Backend

**In Docker, against the code as it is right now** (no rebuild needed — the current
`backend/`, `tests/` and `frontend/` folders are mounted over the ones baked into the image):

```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp \
  -v $PWD/backend:/app/backend:ro -v $PWD/tests:/app/tests:ro -v $PWD/frontend:/app/frontend:ro \
  -w /app --entrypoint python yaap:gpu -m pytest tests -q -p no:cacheprovider
```

(`-p no:cacheprovider` because `tests/` is mounted read-only, so pytest can't write its cache.)

**Inside the running app container:** `docker compose exec -u yaap yaap pytest -q` (as the app user — a root `exec` leaves root-owned files in `data/`) — but note this tests the code
the image was **built** with, not your latest edits, until you `docker compose up --build`.

**On the host** (a virtualenv with `requirements.txt` installed). On this machine ROS puts its own pytest
plugins on `PYTHONPATH`, which break test collection, so run:

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

Useful variations (work with any of the above):

| Command | Does |
|---|---|
| `pytest tests/test_bootstrap.py` | one file |
| `pytest tests/test_bootstrap.py::test_stop_kills_the_worker_and_marks_everything_stopped` | one test |
| `pytest -k bootstrap` | every test whose name contains "bootstrap" |
| `pytest -x` | stop at the first failure |
| `pytest -vv` | print every test name + full assertion diffs |

### Frontend

On the host (needs Chrome or Chromium; set `CHROME=/path/to/chrome` if it isn't on `PATH`):

```bash
python3 tests/frontend/run.py            # all files
python3 tests/frontend/run.py review     # only files whose name contains "review"
```

Output is one line per file (`✓ review.test.html: 23 passed, 0 failed`) plus every failing check by name.

---

## 2. How the backend tests work (read this first)

Everything below happens automatically — it is set up in `tests/conftest.py`.

1. **Throwaway data folder.** Before the app is imported, `conftest.py` points `YAAP_DATA_DIR` at a new temp
   folder. The database, uploaded images, versions, training runs — all go there, and it is deleted when pytest
   finishes. Your real `data/` is never opened.
2. **Clean slate for every test.** An automatic fixture (`_clean_db`) drops and recreates all tables and empties
   the projects folder before each test. So every test starts with an empty database and ids start at 1 again.
3. **`client` = the real app, in-process.** The `client` fixture is FastAPI's `TestClient`: `client.post(...)` /
   `client.get(...)` send real HTTP requests to the real routes, without starting a server.
4. **Background jobs finish before the call returns.** `TestClient` runs FastAPI `BackgroundTasks` right after the
   response. That's why a test can `POST /autolabel` and immediately read the job as `"done"`.
5. **No models: the one expensive call is replaced.** Tests use pytest's `monkeypatch` to swap the call that
   would load a model — `inference.predict`, `inference.uncertainty`, `sam.predict_phrases`, `sam.predict_all`,
   `sam._load_sam`, `sam.warmup`, `trainer.launch` — for a small fake that returns fixed results. Everything
   around that call is real: the database writes, status changes, class mapping, job logs, error handling.
   The swap is undone automatically after the test.
6. **Real images.** `make_png_bytes(w, h, color)` (in `conftest.py`) creates a tiny real PNG, because uploads are
   opened with Pillow and fake bytes would be rejected.
7. **Seeding the database directly.** When the API way would need real training or a real model, tests insert rows
   themselves through `SessionLocal()` — e.g. a `TrainJob` that "is running", or a `DatasetVersion` folder built
   by hand.
8. **Fake worker processes (Bootstrap only).** `FakeLaunch` in `test_bootstrap.py` starts
   `sh -c "sleep 60; true" backend.services.train_worker --job <id>` — a harmless process whose command line looks
   like the real training worker. The app's process checks accept it, so Stop can really send it `SIGTERM` and the
   test can check that it died.

A typical test is **arrange → act → assert**:

```python
def test_autolabel_never_adds_classes_unless_asked(client, monkeypatch):
    monkeypatch.setattr(models_router.inference, "predict", dog)       # fake model: always finds a "dog"
    pid, iid = _project_with_image(client)                             # arrange: project with class "cat" + 1 image
    jid = client.post(f"/api/projects/{pid}/autolabel",               # act: run auto-label
                      json={"model_name": "yolo11n.pt"}).json()["id"]
    assert client.get(f"/api/projects/{pid}/autolabel/{jid}").json()["annotations_added"] == 0   # assert
```

---

## 3. Backend tests, file by file

### `test_projects.py` — projects and classes (14)

| Test | Checks |
|---|---|
| `test_empty_status_counts_toward_annotated_count` | An image marked **empty** has no annotation rows but still counts in the project's `annotated_count` (0 → 1). |
| `test_create_and_list_project` | Creating a project returns its name, task type and zero image/class counts; the list endpoint shows it. |
| `test_create_project_defaults_untitled_name` | A blank name becomes `"Untitled"`. |
| `test_get_missing_project_404` | An unknown project id gives 404. |
| `test_delete_project` | Delete returns `{"ok": true}` and the project is gone afterwards (404). |
| `test_add_class_and_reject_duplicate` | Adding a class works; adding the same name again gives 409. |
| `test_add_class_rejects_empty_name` | A blank class name gives 400. |
| `test_class_gets_next_palette_color_when_unset` | Classes created without a color get different palette colors and `order_idx` 0, 1. |
| `test_update_and_delete_class` | Rename a class (PATCH), delete it, class count back to 0. |
| `test_labeled_count_means_checked_by_a_person` | Of 7 images in every status: *annotated*, *approved*, *empty* count as labeled (3); *auto-annotated* and *review* as predicted (2); rejected and unannotated as neither. |
| `test_deleting_a_project_removes_everything_it_owns` | Deleting a project removes its training run folder and log, its trained and uploaded weight files, and its training-job, weight, auto-label-job and Bootstrap rows — while a second project's identical set stays untouched. |
| `test_deleting_a_project_stops_its_running_training` | A running training job of the deleted project is stopped (its worker process gets `SIGTERM`). |
| `test_deleting_a_project_keeps_a_weight_file_another_project_uses` | A weight file that another project's weight row still points to is not deleted. |
| `test_project_list_counts_and_constant_query_count` | The project list reports image / annotated (1 annotated + 1 empty = 2) / class / version counts correctly, **and** uses the same number of SQL queries after 10 more images are added — a guard against slow "one query per image" code. It counts queries with a SQLAlchemy `before_cursor_execute` listener. |

### `test_images_annotations.py` — upload, list, delete, annotations (8)

| Test | Checks |
|---|---|
| `test_upload_creates_image_with_dimensions` | Uploading a 64×48 PNG stores width/height, status `unannotated`, nothing skipped. |
| `test_upload_skips_disallowed_extension` | A `.txt` upload is listed in `skipped`; nothing is created. |
| `test_upload_dedupes_filename_collisions` | Uploading `a.png` twice stores two different filenames. |
| `test_list_images_filters_by_status` | `GET /images?status=approved` returns only approved images. |
| `test_delete_image` | After delete, the image file endpoint gives 404. |
| `test_replace_annotations_sets_status_annotated` | `PUT /annotations` saves the box exactly and flips the image to `annotated`. |
| `test_replace_annotations_with_empty_list_reverts_to_unannotated` | Saving an empty list puts the image back to `unannotated`. |
| `test_replace_annotations_rejects_unknown_class` | A box of a class that isn't in the project (deleted meanwhile) is dropped and reported (`dropped_unknown_class`); the rest is saved. |

### `test_import.py` — Roboflow-style YOLO zip import (3)

How: each test builds a small zip in memory (`data.yaml`, `train/` and `valid/` folders, one label line) and
uploads it.

| Test | Checks |
|---|---|
| `test_roboflow_valid_folder_maps_to_val_split` | Roboflow's `valid/` folder becomes YAAP's `val` split; imported images are `annotated`. |
| `test_polygons_become_boxes_in_detect_project` | A polygon label imported into a **detect** project becomes its bounding box (pixel values checked). |
| `test_boxes_become_rectangles_in_segment_project` | A box label imported into a **segment** project becomes a 4-point polygon. |

### `test_exporter.py` — dataset versions (6)

| Test | Checks |
|---|---|
| `test_yolo_export_builds_data_yaml_and_labels` | Version becomes `ready`; `data.yaml` exists; label lines are `class cx cy w h` with the right normalized numbers. |
| `test_coco_export_is_additive_not_exclusive` | A **COCO** version still contains the YOLO layout (so it can be trained in YAAP) **plus** `_annotations.coco.json` and flat images per split. |
| `test_coco_json_content_matches_annotations` | The COCO json has 1 image, 1 annotation, category `{"id": 1, "name": "cat"}` and bbox `[10, 10, 30, 20]` (resize set to 100 so the box isn't rescaled). |
| `test_coco_format_rejected_for_classify_projects` | Classify + COCO: the version row is created, but the build fails with a "classification" error. |
| `test_segment_task_writes_polygon_labels` | Segment labels are `class` + 4 point pairs. |
| `test_no_approved_images_raises` | No approved images → the version fails with "no approved images". |

### `test_training.py` — training job files (5)

How: starting real training needs torch, so jobs are inserted directly into the database with a fake `best.pt`.

| Test | Checks |
|---|---|
| `test_job_plot_404_for_unknown_name` | Asking for a plot name that doesn't exist gives 404. |
| `test_job_plot_404_before_it_exists` | Asking for `results` before training wrote it gives 404. |
| `test_job_plot_serves_file_once_it_exists` | `results.png` and `confusion_matrix.png` are served byte-for-byte once they exist. |
| `test_delete_job_wipes_run_dir_and_registered_weight` | Deleting a job removes the whole `runs/job_<id>/` folder **and** its registered model weight. |
| `test_delete_job_without_weights_still_removes_run_dir` | A failed job without `best.pt` still deletes cleanly. |

### `test_model_testing.py` — the "Test model" page backend (7)

How: `inference.predict` is faked to return one "cat" box; versions are built with every image in the test split.

| Test | Checks |
|---|---|
| `test_test_model_returns_predictions_for_the_test_split` | Every test-split image comes back with its prediction mapped to the project's class id, source `model` and a color. |
| `test_test_model_limit_and_offset_page_through_the_split` | `limit` / `offset` return pages that don't overlap. |
| `test_test_model_rejects_version_with_empty_test_split` | A version with no test images gives 400. |
| `test_test_model_rejects_unresolvable_stored_weight` | `weight:99999` gives 400. |
| `test_test_model_rejects_unknown_version` | An unknown version id gives 404. |
| `test_test_model_skips_images_on_inference_error` | If the model fails on every image: 200, nothing shown, both images listed in `skipped`. |
| `test_arbitrary_model_paths_are_refused_everywhere` | **Security:** `/etc/evil.pt`, `../../data/models/x.pt` and `weight:abc` are refused by Test model, Auto-label and Train (400) and by SAM (422). `.pt` files are pickles, so loading a path chosen by the browser would run code. |

### `test_autolabel.py` — "Run a model" auto-label (8)

How: `inference.predict` is faked to return one "cat" box; the project has a class "cat".

| Test | Checks |
|---|---|
| `test_autolabel_runs_as_job_and_leaves_status_auto_annotated` | The job finishes (processed = total = 1, 1 annotation, a log line) and the image becomes `auto-annotated` — not `review`, so it still passes the approval step. |
| `test_clearing_annotations_resets_status_to_unannotated` | Erasing an auto-labeled image's annotations (the 🗑 button) puts it back to `unannotated`. |
| `test_autolabel_preview_lists_only_this_jobs_images` | The preview endpoint returns exactly this job's images with their model annotations. |
| `test_approving_preview_image_moves_it_to_review` | "Approve → To review" (PATCH status `review`) works. |
| `test_autolabel_with_no_images_returns_400` | A project with no images gives 400. |
| `test_autolabel_ignores_image_ids_from_another_project` | Passing another project's image id gives 400, and that image is not touched. |
| `test_inference_error_is_logged_not_fatal` | A model error on an image doesn't crash the job: job `done`, image in `skipped`, "inference failed" in the log. |
| `test_autolabel_never_adds_classes_unless_asked` | A "dog" prediction in a cat-only project is dropped by default; with `create_missing_classes` the class is added. |

### `test_sam.py` — SAM click prompts (3; the 3rd: 5 clicks on 2 images encode only 3 times, never with another image's features)

How: the SAM model loader is replaced by a fake that records exactly what SAM would have received.

| Test | Checks |
|---|---|
| `test_point_prompt_nests_points_and_labels_as_one_object` | 3 clicks are sent as **one** object (`[[[x,y],[x,y],[x,y]]]`, labels `[[1, 1, 0]]`), not 3 separate objects — the bug this guards against made one mask per click. |
| `test_single_point_prompt_still_nested_consistently` | A single click uses the same nested shape. |

### `test_health_and_active.py` — health stats, find-all, active learning, SAM memory (17)

**Version health**

| Test | Checks |
|---|---|
| `test_health_counts_classes_splits_and_warns` | On a hand-made version folder (4 train + 1 val images, classes cat/dog): images per split, empty images, labels per class and split, warnings ("only …", "no labels in the val split"), a score below 100, and the result cached in the version's stats. |
| `test_health_flags_cross_split_duplicates` | The same image in train **and** val counts as a cross-split duplicate and raises a "leakage" error. Calls `health.compute()` directly. |
| `test_health_rejects_unready_version` | A version that is still building gives 400. |

**Data used by Review**

| Test | Checks |
|---|---|
| `test_image_list_exposes_min_model_confidence` | `min_conf` = the lowest **model** confidence on the image (0.35); manual annotations are ignored. |
| `test_min_conf_is_null_for_manual_only_images` | An image with only manual annotations has `min_conf: null`. |

**SAM "Find all"** (SAM is faked)

| Test | Checks |
|---|---|
| `test_exemplar_prompt_returns_all_matches` | A box "exemplar" prompt in the editor returns every match; the box and min score reach `sam.predict_all`. |
| `test_find_all_job_saves_model_annotations_for_review` | Dataset-wide find-all by class: the already-annotated image is skipped, the other gets 2 model annotations, status `auto-annotated`, `min_conf` 0.6. |
| `test_find_all_refused_for_classification` | Classification projects give 400. |
| `test_find_all_several_classes_in_one_job` | Two ticked classes run in one job (phrases sent once, in order); an object found by both phrases is kept once — the higher score wins. |
| `test_find_all_needs_at_least_one_target` | An empty request gives 400. |
| `test_generic_find_all_stores_proposals_not_annotations` | Generic mode saves class-less **proposals**, not annotations: overlapping finds are merged (higher score kept), objects already annotated are skipped, and the GET / DELETE proposals endpoints work. |
| `test_generic_mode_needs_no_classes` | Generic mode works in a project with no classes, using the default phrases. |

**Active learning**

| Test | Checks |
|---|---|
| `test_uncertainty_is_one_minus_mean_confidence` | The uncertainty formula: nothing found (≥ 0.25) → 0.5; lower confidences → more uncertain; a busy image with one doubtful object among 49 sure ones is less uncertain than an image where every detection is doubtful. |
| `test_scoring_job_ranks_images` | The ranking job stores each image's uncertainty (0.2 / 0.9); the state endpoint reports the count and model; deleting an image deletes its score. |

**SAM GPU memory**

| Test | Checks |
|---|---|
| `test_find_all_retries_smaller_on_oom_then_reports_clearly` | Find-all runs SAM 3 at 1008 px; on "CUDA out of memory" it retries once at 644 px, then fails with a clear "GPU out of memory" message. |
| `test_find_all_oom_retry_can_succeed` | Out of memory at full size, success at the smaller size → results are returned. |
| `test_warmup_endpoint_reports_sam_errors` | `POST /api/sam/warmup` returns the warm-up result, and a SAM error becomes a 422. |

### `test_ssh_training.py` — SSH training mode (7)

No real SSH connection is ever made.

| Test | Checks |
|---|---|
| `test_settings_roundtrip` | Default work dir is `~/yaap_jobs`; saved host/user/port come back from GET. |
| `test_settings_reject_option_injection` | A host like `-oProxyCommand=evil` is refused (400) — it would be read as an ssh option. |
| `test_test_endpoint_reports_missing_host` | "Test connection" without a host returns `ok: false` and an error that mentions the host. |
| `test_ssh_command_is_batch_mode` | The built ssh command has `BatchMode=yes` (never asks for a password), `-p` for a non-22 port, and `user@host` as target. |
| `test_remove_remote_refuses_non_job_dir` | Deleting a remote folder is refused unless it is a `job_<n>` folder. |
| `test_job_serialization_shows_ssh_host` | An SSH job in the Train list shows `mode: "ssh"` and host `nx@jetson.local`. |
| `test_remote_script_rewrites_data_yaml_path` | The remote script rewrites `path:` in `data.yaml` to the folder the yaml is in on the remote machine. |

### `test_system.py` — idle shutdown and UI caching (5)

| Test | Checks |
|---|---|
| `test_leaving_writes_marker_when_idle` | `POST /api/system/leaving` with nothing running writes `data/.leave_at` (`watch-idle.sh` then stops the stack). |
| `test_here_clears_marker` | `POST /api/system/here` removes the marker. |
| `test_leaving_refuses_while_job_running` | A running training job → `ok: false`, no marker yet (the shutdown waits for the job — see `test_jobs.py`). |
| `test_leaving_allowed_once_job_finishes` | A finished job doesn't block it. |
| `test_ui_files_are_revalidated_after_updates` | The page and JS files are sent with `Cache-Control: no-cache` and answer 304 to a matching ETag — so browsers never keep stale JS after an update. |

### `test_migrations.py` — database upgrades (5)

How: each test builds its own small SQLite file (`tmp_path`) and calls `init_db()` on it — sometimes after creating
an "old" table by hand, or after swapping `db.MIGRATIONS` for recording fakes.

| Test | Checks |
|---|---|
| `test_an_old_database_gains_new_tables_and_columns` | A `projects` table with only `id` and `name` (and one row) gains the missing columns and all other tables; the old row gets the plain defaults (`""`, `"detect"`) and `NULL` for `created_at`; the schema version is recorded. |
| `test_a_new_database_needs_no_migrations` | A brand-new database runs no migrations and is marked up to date. |
| `test_pending_migrations_run_once_in_order` | Added migrations run once, in order, and never again on the next start. |
| `test_a_failing_migration_changes_nothing` | A migration that fails rolls back with everything else; the version isn't bumped, so it retries next start. |
| `test_first_migration_resets_images_left_without_predictions` | The first real migration resets *auto-annotated* images with no annotations to *unannotated* and leaves the others alone. |

### `test_jobs.py` — background jobs while you keep working (20)

How: the faked model edits the image **during** inference (saves a label, approves it), which recreates the
moment where a job and a human touch the same image. Idle-shutdown tests insert a "running" job row directly.

| Test | Checks |
|---|---|
| `test_replace_existing_keeps_an_approved_image_when_nothing_is_found` | "Replace existing" + the model finds nothing → the approved image keeps its boxes and stays `approved` (the log says "existing annotations kept"). |
| `test_replace_existing_ignores_predictions_of_unknown_classes` | A prediction of a class the project doesn't have is no replacement either — the boxes stay. |
| `test_replace_existing_still_replaces_when_something_is_found` | With a real prediction the old boxes are replaced and the image goes to `auto-annotated`. |
| `test_autolabel_leaves_an_image_labeled_by_hand_meanwhile` | You save a label while auto-label runs on that image → your label stays, nothing added, image `annotated`. |
| `test_autolabel_leaves_an_image_approved_meanwhile` | You approve the image while auto-label (with replace) runs on it → it stays `approved` with its boxes. |
| `test_find_all_leaves_an_image_labeled_by_hand_meanwhile` | Same for SAM "Find all by concept". |
| `test_closing_the_tab_during_a_job_defers_the_shutdown_until_it_ends` | Closing the tab during a job → no shutdown yet (`.leave_pending`); when the job ends → shutdown armed (`.leave_at`). |
| `test_a_real_job_arms_the_deferred_shutdown_when_it_finishes` | A real auto-label job arms a pending shutdown when it finishes. |
| `test_coming_back_cancels_a_deferred_shutdown` | Opening YAAP again cancels it. |
| `test_drop_duplicates_keeps_one_box_per_object` | The one-box-per-object rule itself: another class overlapping by IoU ≈ 0.82 is dropped, the same class at ≈ 0.54 is dropped, another class at ≈ 0.54 is kept (a different, overlapping object), a far box is kept. |
| `test_drop_duplicates_lets_existing_boxes_win_and_can_be_turned_off` | A box already on the image beats even a 0.99 prediction; `max_overlap = 1` keeps everything; whole-image labels keep one. |
| `test_autolabel_keeps_one_box_when_the_model_is_torn_between_classes` | Run a model: cat and dog predicted on the same shape → only the higher-scoring cat is saved; the log reports the skipped duplicate. |
| `test_autolabel_does_not_box_an_object_that_already_has_a_box` | A shape someone boxed (as dog) gets no second box (cat) — only the new object elsewhere is added. |
| `test_autolabel_with_nothing_new_leaves_the_image_alone` | Every detection is a duplicate → nothing is written and the image keeps its status; the log says "nothing new". |
| `test_find_all_does_not_box_an_object_already_labeled_with_another_class` | SAM Find all skips a match on an object already labeled with another class. |
| `test_outlines_are_compared_by_area_not_by_box` | Two triangles splitting one square (same box, no shared area) are both kept; a second outline of the same triangle (another class) is dropped. |
| `test_model_outlines_are_simplified_for_editing` | A 360-point circle outline becomes 8–40 points, all still on the circle. |
| `test_autolabel_in_a_segment_project_keeps_one_outline_per_object` | Run a model in a segment project saves polygons, keeps both triangles of a split square and drops the duplicate. |
| `test_find_all_adds_a_new_class_to_annotated_images_and_keeps_the_rest` | A class added after import: Find all (scope all images) searches only that class, adds its outline to an approved image and leaves the earlier annotation identical; the image goes back to To review. |
| `test_jobs_cut_off_by_a_restart_are_marked_failed` | At startup, jobs the old server was running are marked `failed` ("Interrupted…") instead of "running" forever; a Bootstrap round's labeling job is left to Bootstrap. |

### `test_bootstrap.py` — Bootstrap (29)

How: `FakeLaunch` stands in for the training worker (see §2, point 8); the labeling pass is called directly
(`bootstrap.label_phase(...)`) with `inference.predict` faked. Helpers: `_project(client, labeled=5, unlabeled=3)`
makes images with/without a hand label; `_set(...)` changes an image's status or adds model boxes directly in the
database.

**Starting a round**

| Test | Checks |
|---|---|
| `test_auto_epochs_more_passes_for_fewer_images` | Automatic epochs: 5 or 15 images → 300, 30 → 150, 500 → 50. |
| `test_auto_batch_shrinks_with_image_size` | Batch 8 (16 from 64 images) at 640 px, halved per doubling of imgsz: 1280 → 4 / 8, 1920 → 2. |
| `test_state_counts_what_trains_and_what_gets_labeled` | The card's numbers: annotated images = labeled (5), empty = background (1), unlabeled (2), unreviewed auto-annotated (1); a rejected image counts as neither. Default model `yolo11n.pt`, next round 1. |
| `test_start_needs_enough_labeled_images` | 3 labeled images → 400 "at least 5", no worker started. |
| `test_start_needs_something_left_to_label` | Nothing left to label → 400. |
| `test_start_refuses_a_model_of_another_task` | `yolo11n-seg.pt` in a detect project → 400. |
| `test_start_builds_the_training_set_and_launches_the_worker` | The whole start: run `training`, round 1, image counts; training job settings (300 epochs, `val: False`, weight name "Bootstrap r1 · yolo11n …"); `data.yaml` names; 6 training images (5 labeled + 1 background); the background label file is empty and a box label line is exact; the Train list shows "bootstrap round 1"; a second start → 409; idle shutdown refused while the round labels. |
| `test_segment_set_keeps_polygons` | Segment training labels are polygons. |
| `test_classify_set_is_one_folder_per_class_with_val` | Classify training set: `train/` and `val/`, one folder per class (`a/b` becomes `a_b`), 5 images in each. |
| `test_classify_needs_two_classes` | Classify with only one class → 400. |
| `test_training_set_is_a_copy_and_is_removed_after_training` | Training images are **copies**, not links to your originals (ultralytics may re-save JPEGs in place), and the copy is deleted when labeling starts. |

**The labeling pass**

| Test | Checks |
|---|---|
| `test_labeling_pass_fills_targets_and_replaces_unreviewed_guesses` | Unlabeled image → labeled; an image the user labels by hand during the round → left alone; an old unreviewed guess → replaced (2 boxes → 1); a "dog" prediction (no such class) → dropped; an image where nothing is found → keeps its earlier guess. Afterwards: run `done`, 2 images to review, next round 2, "+1 labeled since the last round". |
| `test_unreviewed_guesses_are_kept_when_refresh_is_off` | With "re-label unreviewed" off, auto-annotated images are not touched. |
| `test_the_next_round_relabels_rejected_images` | A rejected image holding only model boxes is a target (the card counts it): the next round replaces its boxes and sends it back to Review. A rejected image with a box a person drew is left alone. |
| `test_rejected_images_wait_for_a_person_when_asked` | With "re-label rejected" off, rejected images keep their boxes and status. |
| `test_a_round_keeps_one_box_per_object` | Two predictions of one object (same class, slightly shifted) → one box. |
| `test_a_segment_round_saves_outlines` | A segment round uses `yolo11n-seg.pt` and saves both outlines of two objects that share one box. |
| `test_state_warns_about_classes_with_few_examples` | The card's state lists classes with fewer than 5 labeled objects. |
| `test_a_hand_label_saved_while_its_image_is_predicted_wins` | The user saves a label (made with SAM, so stored as source `model`) **while the model is predicting that same image** → the user's label survives and the image stays `annotated`. |
| `test_one_unreadable_image_does_not_end_the_round` | An unexpected error on one image → that image is skipped, the rest are labeled, the round ends `done`. |
| `test_a_round_where_every_image_fails_is_reported_failed` | If every image fails, the round ends `failed` with the first error in the message. |
| `test_classes_renamed_mid_round_still_map` | A class renamed while the round runs still receives its predictions. |
| `test_classify_rounds_respect_min_confidence` | Classify: a top class below *Min confidence* leaves the image unlabeled; above it, the image is labeled. |

**Review, stop, crashes**

| Test | Checks |
|---|---|
| `test_accept_refuses_predictions_the_reviewer_did_not_see` | Accept sends the annotation ids that were on screen: if they changed meanwhile → 409; with the current ids → approved. |
| `test_stop_kills_the_worker_and_marks_everything_stopped` | Stop on the card: the worker process gets `SIGTERM`, the round and its training job become `stopped`, and a late error from the worker can't turn it into `failed`. |
| `test_stopping_or_deleting_the_train_job_stops_the_round` | Stop on the Train page during training, and ✕ (delete) during labeling, both kill the worker and mark the round `stopped`. |
| `test_a_worker_that_died_silently_fails_its_run` | A worker that exits without reporting → the round is marked `failed` ("stopped unexpectedly") and a new round can start. |
| `test_a_round_being_launched_is_not_mistaken_for_a_dead_one` | While the training set is still being built (worker not started yet) the round stays `training`, and a second start is refused. |
| `test_a_recycled_pid_is_not_taken_for_the_worker` | After a container restart a stored process id can belong to something else — here the test process itself. It is not trusted as the worker, so the round is marked `failed`. |
| `test_leaving_is_not_blocked_by_a_round_whose_worker_died` | A round whose worker died doesn't block the idle shutdown forever. |

### `test_bugfixes.py` — regressions from the VisDrone end-to-end test (35)

One test per bug report of the 2026-11-08 test run (`~/yaap-visdrone-test/2026-11-08/report/bugs/`).

| Test | Checks (bug) |
|---|---|
| `test_find_all_predictor_runs_at_sam3_native_size` | The find-all predictor is created with `imgsz=1008` (sam-1). |
| `test_generic_find_all_defaults_to_the_class_names` | Generic find-all without nouns searches for the project's class names (sam-2). |
| `test_find_all_survives_a_locked_database_on_one_image` | A "database is locked" on one image skips that image, the job finishes the rest; find-all annotations carry `origin: "sam"` (bootstrap-6). |
| `test_pretrained_alias_resolves_to_the_models_dir` | `yolo11n.pt` resolves to `data/models/yolo11n.pt` (platform-1 / bootstrap-1). |
| `test_training_worker_gets_a_writable_config_dir` | A read-only `…/Ultralytics` config folder → the worker gets a private writable one, fonts copied (bootstrap-5). |
| `test_ranking_scores_unreviewed_predictions_too` | Ranking covers unannotated + auto-annotated images, not hand-labeled ones (bootstrap-3). |
| `test_bootstrap_does_not_replace_other_tools_predictions_by_default` / `test_bootstrap_card_counts_other_tools_predictions` | A round only replaces its own (or pre-origin) predictions unless `relabel_other_tools`; the card counts the others (bootstrap-7). |
| `test_class_rename_to_existing_name_is_refused` / `test_new_class_after_a_delete_gets_a_fresh_index` | Duplicate class names → 409; a new class after a delete gets max index + 1 (platform-9/10). |
| `test_annotation_put_refuses_broken_geometry` (5 cases) | bbox without x/y/w/h, w ≤ 0 or h ≤ 0, a 2-point polygon, points that aren't pairs → 422 (platform-3). |
| `test_annotation_put_with_stale_ids_is_refused` | A PUT whose `expect_ids` don't match the image any more → 409, nothing deleted (annotator-6/13). |
| `test_unapproving_an_empty_image_returns_it_to_unannotated` | `annotated` on an image with no annotations becomes `unannotated` (annotator-7). |
| `test_version_keeps_an_image_with_one_broken_annotation` | A broken stored box is left out and listed, the image is still exported; `data.yaml` has no `path`; `split_ids` recorded (platform-2/14/16). |
| `test_raw_exports_dont_share_a_folder_and_same_stems_keep_their_labels` | Two raw exports use separate folders (removed afterwards); `same.png` + `same.jpg` keep separate label files (platform-4/5). |
| `test_import_matches_labels_by_folder_not_just_name` | `train/images/dup.png` gets `train/labels/dup.txt`, `valid/…` its own (platform-6). |
| `test_import_skips_malformed_label_lines` | A bad label line is skipped and reported, all 3 images imported (platform-7). |
| `test_corrupt_upload_leaves_no_file` | A corrupt upload writes nothing; the next valid one keeps its name (platform-8). |
| `test_exif_rotated_upload_is_stored_upright` | An Orientation=6 JPEG is stored rotated, without the tag (platform-13). |
| `test_tiff_upload_is_stored_as_png` | A 16-bit TIFF is stored as `scan.png` (platform-15). |
| `test_test_model_uses_the_versions_own_test_split` | After the project changes, Test model still shows exactly the version's test images (platform-16). |
| `test_closing_one_of_two_tabs_does_not_arm_the_shutdown` | `/leaving?tab=A` while tab B is open arms nothing; the last tab does (platform-11). |
| `test_restart_drops_a_shutdown_deferred_by_a_dead_job` | `fail_interrupted_jobs()` removes `.leave_pending` (platform-12). |
| `test_classify_images_hold_one_class_tag_and_no_boxes` | Classify: two tags or a box → 422; detect: a class tag → 422 (classification test). |
| `test_bootstrap_uses_224_px_for_classification_by_default` | Classify Bootstrap trains at 224 px; the card warns below 20 labeled images per class. |
| `test_clear_splits_forgets_imported_pins` | `POST …/images/clear-splits` forgets imported train/val/test pins (segmentation test). |
| `test_classify_raw_export_round_trip_keeps_the_class_order` | A classify raw export has `classes.txt`; re-importing it keeps the class order (round-trip test). |
| `test_version_reports_labels_too_small_after_the_resize` | A 3×3 box resized ×0.1 is left out and counted in `tiny_dropped`. |
| `test_listing_images_uses_a_fixed_number_of_queries` | Listing 30 images takes ≤ 8 queries, not 3 per image (load test: 5 000 images 6 s → 0.2 s). |
| `test_coco_versions_use_roboflow_split_folders` | COCO versions: `train/ valid/ test/` (`data.yaml` val → `valid/images`), YOLO ones keep `val/`; health still counts val. |

---

## 4. How the frontend tests work

1. `tests/frontend/run.py` starts a small web server on a random port that serves the repo and maps `/static/` to
   `frontend/` — the same URLs the real app uses. It opens every `*.test.html` in headless Chrome.
2. Each test page loads `harness.js`. `t.boot({ routes })` builds the page from the **real** `frontend/index.html`
   and loads the **real** `api.js`, `editor.js` and `app.js` — but replaces `window.fetch` with a mock API: a list
   of `[method, urlRegex, handler]` routes. No backend runs.
3. The test drives the UI like a user: `t.go("#/p/1/review")` navigates, then it clicks buttons, presses keys,
   reads the DOM, and calls `t.check("name", condition)` for each expectation.
4. When the test function ends, the results are written into the page title (`RESULT {...json...}`); `run.py`
   reads them from the dumped page and prints passed/failed checks.
5. Any API call a test didn't mock is answered with `{}` and listed as **unmocked requests** — a sign that a
   test's mock list is behind the app (currently none).

### `annotate.test.html` — the editor (35 checks)

| Scenario | Checks |
|---|---|
| Keyboard listeners don't pile up | one editor key handler after three visits to Annotate · no handler after leaving · `S.editor` cleared on leave · arrow keys do nothing outside Annotate |
| Starting tool | detect starts on the box tool (highlighted) · segment starts on polygon (highlighted) |
| Drawing a box | the drawn box waits for a class · it stays visible while you pick · the picker lists every class · the picked class lands on the annotation · the picker hides · no pending outline afterwards |
| Autosave | a delayed autosave still writes to the image you edited, even if you moved on · no error toast |
| Leaving | Exit lands on project Home · Next on the last image also goes to project Home |
| Most uncertain first | the toggle is hidden until images are ranked · shown once they are · ticking it opens the most uncertain image · and keeps the ranking order |
| Undo / redo | buttons start disabled · a new box can be undone and redone with the buttons · a click that changes nothing adds no step · a drag is one step · `Ctrl+Z` / `Ctrl+Shift+Z` undo and redo it · `Delete` then `Ctrl+Z` restores the box · `Ctrl+Y` redoes · `Ctrl+Z` while drawing a polygon removes its last point · undone changes are autosaved · each image has its own history, kept when you come back |

### `review.test.html` — Review / QC statuses (26 checks — incl. a fast double `R` in Quick Review rejects only the image on screen)

| Scenario | Checks |
|---|---|
| Which status goes in which queue | rejected is **not** in To review but **is** in Needs re-annotation · annotated, review, auto-annotated, empty **are** in To review · approved and unannotated are **not** (8 checks) · each of the 7 statuses is never in both queues (7 checks) |
| Review grid | To review count is 2 · Needs re-annotation count is 1 · ✕ on a rejected image clears it back to annotated |
| Quick Review | Reject sets `needs_reannotation` |
| Clicking a tile | a To review tile opens Quick Review at that image · an Approved tile opens Quick Review · a Needs re-annotation tile still opens the editor |
| Leaving Quick Review | ⏏ Exit returns to project Home · so does `Esc` |

### `autolabel.test.html` — Find all without classes, Run a model on rejected images (14 checks)

| Scenario | Checks |
|---|---|
| A project with no classes | Find all opens in Generic mode · Generic can run · switching to By class blocks it with a hint · switching back re-enables it · the request is sent with `mode: "generic"` |
| After the run | with no classes the button says "Add classes →" · it opens Classes · Classes shows no "Label them" while there are still no classes |
| Classes added | Classes offers "Label them →" (1 image waiting) · it opens the editor on just the outlined images · with classes, Find all opens in By class |
| Run a model → Rejected images | choosing that scope ticks "Replace existing" · only the rejected images are sent · with the duplicate-overlap setting |

### `test-model.test.html` — the Test model page (19 checks)

| Scenario | Checks |
|---|---|
| Pickers and page | only `ready` versions offered · trained/uploaded weights offered · weight upload present · "Powered by ComputerVisionAIHub" credit and its logo |
| Running | request sends version, model and limit 20 · 20 cards rendered · "Load more" shown |
| Show boxes / labels toggles | hiding boxes · labels stay on their own · hiding labels · boxes come back |
| Full-screen card | opens full screen · title is the filename · keeps the grid's hidden labels · keeps boxes shown |
| Paging | Load more asks for offset 20 · appends the rest (25) · hides at the end |

### `train-viewer.test.html` — Train page log and plots (14 checks)

| Scenario | Checks |
|---|---|
| Inline log | shows · asks for 200 lines |
| Full-screen log | opens · asks for 5000 lines · has the latest line · fills the screen |
| Viewer closing | Close closes and empties it · clicking the dark area closes it · a late "close" event doesn't wipe a viewer that was reopened · navigating away closes it |
| Plots | results.png opens full screen · clicking toggles actual size · plots reload once when the job finishes · a finished job's plots don't reload on every poll |

### `editor-fixes.test.html` — editor regressions from the VisDrone test (31 checks)

Run against the code before the fixes, 20 of the first 24 checks fail. Covers: digit / `0` / `Enter` label a box that waits
for its class and drawing the next box keeps it · an edit made during an in-flight save reaches the server and the UI
says "saved" · a failed save keeps you on the image, shows an error, and saves once the server is back; a viewed-only
image is never written · a job's prediction added while the image is open survives the next save (merge on 409) ·
the smallest box under the cursor is picked, Alt-click cycles · a drag is one change, selecting only moves the
highlight · boxes are clipped at the left/top edge; a 3 px object can be boxed at 16× · a resize keeps the zoom ·
double-click closes a polygon with 3 points · a 2nd SAM click during a request is sent after it · the class picker for
a box at the top edge is fully visible · a late SAM answer isn't shown on the next image · double-click on Approve
sends one request; un-approving an empty image → unannotated · late Review data doesn't paint over Classes · Return
to Home goes to the project Home · a classification project's editor opens (it used to crash on the missing ↶
button), a click and a key tag the image with one class, Ctrl+Z restores the previous one. The mock server can answer errors (`{ __status: 409 }`) and serves a 100×100 image.

---

## 5. What the tests do **not** cover

- **Real models.** No test runs ultralytics training, YOLO inference or SAM 3; those were checked by hand and with
  live runs (steps in `FEATURES_USAGE.md`).
- **Real SSH.** Only settings, validation and the shape of the ssh command are tested.
- **Newer UI.** There are no frontend tests yet for the Bootstrap card, the SAM "find all" walkthrough, the
  find-all-by-concept card, the health panel or the SSH tab ("Most uncertain first" now has some, after its
  wiring was found missing). The backend behind them is tested; the screens are
  covered by the manual checks in `FEATURES_USAGE.md`.

---

## 6. Which file to look at for a feature

| Feature | Backend test file | Frontend test file |
|---|---|---|
| Projects, classes | `test_projects.py` | — |
| Upload, annotations, statuses | `test_images_annotations.py` | `annotate.test.html` |
| Import zip | `test_import.py` | — |
| Versions / export | `test_exporter.py` | — |
| Version health | `test_health_and_active.py` | — |
| Run a model (auto-label) | `test_autolabel.py` | — |
| SAM clicks / box | `test_sam.py` | — |
| SAM find all (editor `E`, dataset, generic) | `test_health_and_active.py` | `autolabel.test.html` |
| Active learning ranking | `test_health_and_active.py` | — |
| Bootstrap | `test_bootstrap.py` | — |
| Review / QC | `test_autolabel.py`, `test_bootstrap.py` (guarded Accept) | `review.test.html` |
| Training jobs, plots | `test_training.py` | `train-viewer.test.html` |
| SSH training | `test_ssh_training.py` | — |
| Test model | `test_model_testing.py` | `test-model.test.html` |
| Idle shutdown, UI caching | `test_system.py`, `test_jobs.py`, `test_bootstrap.py` | — |
| Project deletion, labeled counts | `test_projects.py` | — |
| Database upgrades | `test_migrations.py` | — |
| Jobs vs. a user editing at the same time | `test_jobs.py`, `test_bootstrap.py` | — |
| One box per object (duplicate overlap) | `test_jobs.py`, `test_bootstrap.py` | `autolabel.test.html` |
| Re-labeling rejected images | `test_bootstrap.py` | `autolabel.test.html` |
| Segmentation (outlines) | `test_jobs.py`, `test_bootstrap.py`, `test_exporter.py` | — |

---

## 7. Adding a test

**Backend** — add a `test_...` function to the matching file (or a new `tests/test_<feature>.py`). Ask for the
`client` fixture, and `monkeypatch` if a model is involved:

```python
from conftest import make_png_bytes


def test_autolabel_finding_nothing_labels_nothing(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", lambda *a: [])    # fake model: finds nothing
    pid = client.post("/api/projects", json={"name": "P", "task_type": "detect"}).json()["id"]
    files = [("files", ("a.png", make_png_bytes(100, 100), "image/png"))]
    client.post(f"/api/projects/{pid}/images", files=files)
    jid = client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"}).json()["id"]
    job = client.get(f"/api/projects/{pid}/autolabel/{jid}").json()
    assert job["status"] == "done" and job["images_labeled"] == 0
```

Patch the function the code actually calls. The auto-label route calls `inference.predict(...)` through the
`inference` module, so patching that module's attribute is enough. If code imported a function with
`from x import y`, patch `y` in the module that did the import.

**Frontend** — copy an existing `.test.html`, change the mocked `routes`, navigate with `t.go(...)`, and add
`t.check(...)` lines. Run it alone with `python3 tests/frontend/run.py <part-of-the-filename>`.
