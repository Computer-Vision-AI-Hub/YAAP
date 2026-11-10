# New features — usage and how to test

After pulling these changes: rebuild (`docker compose up --build`) and hard-refresh the browser (Ctrl+Shift+R).
Automated backend tests: `pytest tests -q` (models are mocked; the UI and real SAM 3 / SSH need the manual checks below).

---

## 1. SSH training (Train → 🔌 SSH)

Train on a remote GPU machine over SSH; log, Stop and results stay in YAAP.

**Setup** — key-based auth only (no passwords). The same steps are in the SSH tab under *First-time setup*.

*1. On the remote (GPU) computer* — install and start the SSH server, create the Python environment YAAP runs, and
note its IP address:

```bash
sudo apt update
sudo apt install -y openssh-server rsync python3-venv
sudo systemctl enable --now ssh
# (only if the firewall is on:  sudo ufw allow ssh)

python3 -m venv ~/yaap-env
~/yaap-env/bin/pip install -U pip
~/yaap-env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
~/yaap-env/bin/pip install ultralytics
~/yaap-env/bin/python -c "import torch; print(torch.cuda.is_available())"   # must print True (needs the NVIDIA driver: nvidia-smi)
hostname -I                                                                 # the IP to use as Host
```

On a Jetson, install PyTorch from NVIDIA's JetPack wheels instead of the `pip install torch` line.

*2. On your computer* (the one running YAAP) — create an SSH key if you don't have one, **without a passphrase**
(YAAP runs ssh in the background and can't type it), and copy it to the remote once:

```bash
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519   # creates it only if missing
ssh-copy-id -i ~/.ssh/id_ed25519.pub janko@192.168.42.115                     # asks the remote password one last time
ssh -o BatchMode=yes janko@192.168.42.115 true && echo OK                       # must print OK without any prompt
```

(If your existing key has a passphrase, make a separate one for YAAP: `ssh-keygen -t ed25519 -N "" -f ~/.ssh/yaap_ed25519`,
copy that one, and put `~/.ssh/yaap_ed25519` in *Identity file*.)

*3. In YAAP* — Train → SSH: *Host* = the IP, *User* = the remote user, *Identity file* empty (uses `~/.ssh/id_ed25519`),
*Remote python* `~/yaap-env/bin/python` → **Test connection**. The container reads your `~/.ssh` read-only
(`docker-compose.yml` mounts it) and keeps the remote's host key in `data/ssh_known_hosts` — no "are you sure" question.

**If Test connection fails**
- *Permission denied (publickey)* — the key wasn't copied (step 2), it has a passphrase, or the container can't read it:
  the container runs as user id 1000, so your `id -u` must be 1000 or set `UID`/`GID` in `docker-compose.yml` and rebuild.
- *Connection refused / timed out* — the SSH server isn't running on the remote (`sudo systemctl status ssh`), wrong IP,
  or a firewall.
- *CUDA not available* — the remote torch is a CPU build or the NVIDIA driver is missing.

The remote needs `rsync` for copying (YAAP falls back to `scp` without it).

**Try it**
1. Train → SSH tab. Fill host, user, port, (identity file), remote python, remote work dir.
2. **Test connection** → shows torch / ultralytics versions and the GPU. "CUDA not available" means the remote torch has no CUDA.
3. Pick a version, model, epochs/imgsz/batch/patience/device, optionally open *Optional hyperparameters*, then **Start training on SSH host**.
4. The job appears under Jobs with a 🔌 host badge. **log** streams the remote output live. **stop** kills the remote process too.
5. On success the run folder is downloaded to `data/runs/job_<id>/` and `best.pt` is registered as a model (use it in Auto-label).
6. Delete the job → you are asked whether to also remove the remote folder.

Quick smoke test: `yolo11n.pt`, 1 epoch, imgsz 320, batch 4.
Remote files live in `<work dir>/job_<id>/` (`train.log`, `pid`, `exitcode`, `runs/`).

---

## 2. Import / upload confirmation

Upload images or import a dataset zip → a green **✓ Uploaded / Imported N image(s)** toast stays ~8 s. The Import button shows "Importing…" while working.

## 3. Annotate tool: empty state

Open the annotator from a QC section (Review page) that has no images → "No images to annotate." with a **Return to Home** button.

---

## 4. Version health statistics (Versions → ♥ Health)

1. Generate a version, wait until it is `ready`, click **♥ Health**.
2. Check: score /100, warnings (empty split, class with few labels, class missing in val/test, imbalance, leakage), images per split, labels per class (train/val/test), objects-per-image histogram, object size/shape, duplicates.
3. Results are cached in the version; works for old versions too (computed from the exported files).

Good test: a version with a class that has only a few labels, or the same image in train and val → expect warnings.

---

## 5. Review mode (Review → To review)

Back to the simple flow: one image at a time, **Accept** (`A`) or **Reject** (`R`). The per-box keyboard mode and low-confidence ordering were removed.

---

## 6. "Find all like this" and "Find all by concept" (SAM 3)

Needs SAM 3 weights (`sam3.pt`) and a recent ultralytics.
When you pick the SAM tool, SAM 3 is warmed up in the background (~20 s once; the hint bar says "warming up SAM…"). Both SAM copies stay loaded — one on the GPU, the other parked in RAM — so switching between clicks/box and `E` takes ~1–3 s and the GPU holds only ~2 GB.

**Click and box prompts:** only the first click on an image waits for SAM to read it (~2 s on a phone photo); further
clicks and boxes on the same image answer in ~0.2 s. A box around the part gives a much better outline than one
click in its middle (UP_board: mean outline IoU .79 vs .47) — drag a box, then refine with clicks. After a mask
appears, a class key (`1`–`9`, `0`) labels and accepts it.

**In the editor (one image):** SAM tool (`S`) → drag a box around one example → the **E** button (find all like this, shortcut `E`). Adjust *min score*.
YAAP then walks through the matches one by one: the current box is highlighted in yellow and a panel asks for its class.
- Pick a class (click or `1`–`9`) → it moves to the next box. Matches you've labeled turn into that class's colour.
- **🗑 Erase box** (`Del` / `X`) removes a wrong match. `←` / `→` (or Skip) move between boxes without labeling.
- **✓ Done** (`Enter`) adds the labeled boxes and leaves out the unlabeled ones. `Esc` / ✕ discards everything (asks first if you already labeled some).

**Whole dataset:** Auto-label → *Find all by concept — SAM 3*.
1. Tick the classes to find (each has an editable phrase, default = the class name, e.g. "USB port"), set min score and scope. One run handles all ticked classes; if two phrases hit the same object, the higher score wins.
2. Run. Matches are saved as model predictions and shown in the preview grid → check them in Review (low confidence first).
3. To add more classes later, run again with scope **All images** (the default skips images that already have annotations). Duplicates of the same class are skipped.

**Generic mode (no classes yet):** in the same card switch to *Generic — outline everything, label later* (a project
without classes opens in this mode, since *By class* needs classes). SAM outlines every object using generic nouns — the
project's class names when it has classes, otherwise `component, part, object`. Edit them to name the kind of thing in your
images: SAM 3 does not treat vehicles as an "object" (on VisDrone `component, part, object` found 0 of 681 vehicles,
`car, van, truck, bus` 78 %). Nothing is labeled automatically: the objects are saved as class-less proposals.
1. When the run finishes click **Label the found objects →** (or open the editor on any image showing the **⌖ Label found objects** button).
   No classes yet? The button says **Add classes →**: add them on the Classes page, which then offers **Label them →**.
2. The editor walks through each outlined object (highlighted, class panel bottom-right): pick a class (`1`–`9`, `0` = 10th), erase wrong ones (`Del`), **Done** (`Enter`). It then jumps to the next image that still has proposals.
3. `Esc` pauses: proposals stay until you finish an image (unlabeled ones are dropped at that point).

Note: a box example only works inside its own image; across images SAM 3 uses text phrases.
SAM 3 sees each image at its native 1008 px (it used to be ultralytics' default 640, which missed small objects); on a
GPU out-of-memory error it retries once at 644 px.

---

## 7. Active learning (Auto-label → Active learning)

1. Pick a model in the Auto-label model dropdown (a trained one is best) → **Rank unlabeled images**. This ranks every
   image nobody has checked yet: unlabeled ones and those holding only unreviewed predictions (after a Bootstrap round or
   a find-all that is most of them).
2. When done, the page shows how many images are ranked and by which model.
3. Open the annotator: tick **Most uncertain first**. Those images are walked from most to least uncertain. Re-rank after retraining.

Uncertainty: 1 − the mean confidence of the image's detections (those ≥ 0.25); an image with no detections scores 0.5;
classification uses entropy. (The old rule — the single most ambiguous detection decides — gave almost every busy image
≈ 1.0; on VisDrone it didn't correlate with how wrong the model was, the mean does: Spearman −0.64 vs per-image F1.)

---

## 8. Bootstrap (Auto-label → Bootstrap)

Label a few images by hand; YAAP trains a small model on them and pre-labels the rest. Repeat in rounds.

**What it trains on:** images a human has checked — *annotated* (saved in the editor), *approved* (accepted in Review)
and *empty* (marked "no objects", used as background). Unreviewed model output and rejected images are never used.
**What it labels:** images with no annotations, plus (option, on by default) images that only hold unreviewed predictions
(*auto-annotated*, from an earlier round, SAM or *Run a model*) — those get the new model's predictions instead; if the
new model finds nothing there, the old prediction stays. Also (option, on by default) **rejected** images
(*Needs re-annotation*) whose boxes all came from a model: the new model's predictions replace the rejected ones and
the image goes back to Review — so the round after a review gives every rejected image a fresh try. Hand labels and
review decisions are never overwritten, even if you keep labeling or reviewing while a round runs (each image is
re-checked at the moment it is written); a rejected image with boxes a person drew stays for you to fix.
Your original image files are never modified — the training set is a temporary copy, deleted after training.

1. Label at least 5 images (10+ recommended) in the editor — every class you care about should appear a few times.
2. Auto-label → **⚡ Bootstrap**. Step 1/2 trains `yolo11n` (or `-seg` / `-cls`, matching the project) — ~1 min on a GPU
   for 10 images; the progress bar shows the epoch. Step 2/2 labels the images; the log lists each one.
3. **Review round N → (count)** opens Quick Review with only this round's images still waiting: Accept (`A`) what's right,
   Reject (`R`) what's wrong. Fix rejected ones in the editor later (Review → Needs re-annotation). If a round replaced an
   image's predictions while you were looking at it, Accept refuses and shows the new ones.
4. Back on Auto-label the card shows "+N labeled since the last round" and "N rejected" → **⚡ Run round N+1**: it trains
   on everything you accepted and re-labels what you rejected.

Options: base model, epochs (auto: 300 for ≤15 images, 150 for 30, down to 50 for 90+), image size (640; use 1280 for small
objects — batch is lowered automatically for larger sizes), min confidence (0.4), duplicate overlap (0.7, see section 11),
re-label unreviewed, re-label rejected, and **also replace predictions made by SAM or Run a model** (off: a round only
replaces its own earlier guesses — on VisDrone, round 1 replacing SAM's boxes lowered truck F1 from .64 to .43).
Each round is also a normal training job (Train page, labeled "bootstrap round N") and registers its weights as
"Bootstrap rN · yolo11n · K imgs" — usable in *Run a model*, Test model, or as an active-learning ranker.
**Stop** (on the card, or stop/✕ on the Train page) ends the whole round; images labeled so far keep their predictions.
Classification rounds use *Min confidence* too: an image whose top class scores lower stays unlabeled. They train at
224 px (the *Image size* default for classification; 640 for boxes/outlines) and need far more labels than detection:
on VisDrone vehicle crops (car/van/truck/bus), 8 labeled images per class gave ~35 % accuracy, 24 per class ~55 %, 40 per
class ~65 % — the card warns below 20 per class. Such a small model is also over-confident (wrong answers at ~90 %), so
*Min confidence* filters little and the uncertainty ranking can't yet tell right from wrong; label more first.
In the editor a classification image has one class tag: click a class or press its key (`1`–`9`, `0`); `Ctrl+Z` undoes.

Which tool for which data: on the UP_board photos (10 large electronics parts) Bootstrap from 17–19 photos reached box
F1 .86–.99 (300 epochs; 88 epochs gave .80–.94), from 10 photos only .58 — label 15–20 photos from different angles.
SAM 3 *Find all by class* found almost none of those parts by name (F1 ≤ .10) but works well for everyday objects
(vehicles on VisDrone: F1 .76); its *Generic* outlines still found 80 % of the parts, which speeds up labeling the seed.

Why YOLO and not a few-shot "proposer": on RaspberryPI01 with 10 labeled photos a 1-minute YOLO11n fine-tune reached
box F1 0.74 (0.75 with 20), vs 0.55 for a DINOv2 + SAM 3 proposer and 0.33 for SAM 3 text prompts alone.

Quick test: a detect project with ~10 annotated and ~30+ unlabeled images → Bootstrap → watch the epochs → Review round 1.

---

## 9. Jobs and you, at the same time

- **Auto-label / Find all never overwrite your work in progress.** If you save, approve or reject an image while a
  job is processing it, the job leaves that image alone (its log says "changed by hand meanwhile, left alone").
- **"Replace existing annotations" only replaces.** If the model finds nothing on an image (or only classes the
  project doesn't have), the image keeps its annotations and status — an approved image is never emptied.
- **Closing the tab no longer kills a running job.** The idle shutdown waits until auto-label, Find all, ranking,
  training or Bootstrap finishes, then stops the container ~60 s later. Opening YAAP again cancels it.
  Jobs cut off by a server restart now show as failed ("Interrupted…") instead of "running" forever.

Quick test: start Find all on many images, close the tab, check `docker ps` — the container stays up until the job
ends, then stops about a minute later.

---

## 10. Housekeeping

- **Deleting a project deletes everything it owns.** Besides its images, labels and versions: its training jobs
  (a running one is stopped first), their `data/runs/job_<id>/` folders and logs, the weights trained or uploaded for
  it, its auto-label jobs and Bootstrap rounds. Folders on an SSH host are not touched.
- **"labeled" means checked by a person.** The header and the project cards count *annotated*, *approved* and
  *empty* images as labeled, and show model output nobody has reviewed yet separately as **predicted**
  (e.g. `177 imgs · 10 labeled · 167 predicted`). Rejected images count as neither.
- **Database upgrades are automatic.** On start YAAP adds new tables and columns to an existing `data/yaap.db` and runs
  any pending one-off fixes, so updates never need a fresh database. Back up `data/` before big updates anyway.

Quick test: delete a project that has a training job — its `data/runs/job_<id>/` folder and its ★ weights are gone.

---

## 11. One box per object (duplicate overlap)

Every model-based labeling — *Run a model*, *Bootstrap*, *Find all by concept* (both modes) and *Find all like this* in
the editor — now keeps **one box per object**:

- two boxes of the **same class** overlapping by IoU ≥ 0.5 are the same object (the higher score is kept);
- two boxes of **any class** overlapping by IoU ≥ **Duplicate overlap** (default 0.7) are the same object — e.g. a
  model unsure between *Triangle* and *Rectangle* outputs both on one shape;
- a box that is **already on the image** always wins: a model never adds a second box on an object someone (or an
  earlier run) already boxed. If every detection on an image is such a duplicate, the image is left as it was.

Set *Duplicate overlap* in the Bootstrap options, the *Run a model* card and the *Find all* card; lower it to merge more
aggressively, 1 keeps everything. Job logs say how many duplicates were skipped ("…, 2 duplicate(s) skipped").
In the editor, *Find all like this* leaves out matches on objects that already have a box ("· 3 already labeled").

**Run a model on rejected images:** the *Scope* list has *Rejected images (needs re-annotation)*. It ticks *Replace
existing annotations* for you, so the new boxes replace the rejected ones; the images go back to Review.

Quick test: on the Shapes data a Bootstrap round skipped 15 duplicate boxes in 11 images and left none; after rejecting
3 images in Review, the next round trained on the accepted ones and gave the 3 rejected images new boxes.

---

## 12. Segmentation projects

Every auto-label feature works the same way in a segmentation project — the labels are outlines (polygons) instead
of boxes: Bootstrap trains `yolo11n-seg`, *Run a model* uses `-seg` models, *Find all by concept* / *Generic* /
*Find all like this* (SAM 3) save outlines, Review and the rejected-image loop are unchanged.

- **Editable outlines:** model outlines are simplified like SAM's (≈ 10 points instead of hundreds).
- **One outline per object** compares outlines **by area**, not by box: two shapes whose boxes overlap (a triangle
  lying on a hexagon) are both kept; two outlines of the same shape are merged.
- **Few examples warning:** the Bootstrap card warns when a class has fewer than 5 labeled objects
  ("Few examples of: Circle (4)").

Tested on a segmentation copy of Shapes (outlines made by SAM 3, 10 hand-labeled images):

| | Result |
|---|---|
| Bootstrap round 1 (yolo11n-seg, 10 images, 29 objects) | 72/79 images; 98 % of outlines land on a real shape and fit it well, but only 41 % have the right class — Circle had 4 examples, Triangle 6 |
| Reject 3 / accept 2 → round 2 | the 3 rejected images got new outlines and went back to Review |
| Find all by class (SAM 3, no training) | 79/79 images · 99 % precision, 100 % recall |
| Generic (no classes) | 263 outlines for 266 objects |
| Every path | 0 duplicate outlines left (37 + 25 skipped in the two rounds) |

So for segmentation: start with *Find all by concept* when the class names describe the objects, and give Bootstrap
at least ~5 labeled objects of every class (the warning says which ones are short).

---

## 13. Undo / redo in the editor, Exit in Review

- **Undo** — the **↶** button under the tools, or `Ctrl+Z` (`Cmd+Z` on a Mac). **Redo** — **↷**, `Ctrl+Shift+Z` or
  `Ctrl+Y`. Every change can be taken back: a drawn box or polygon, a move or resize (one drag = one step), a vertex
  added, moved or removed, a delete, a class change, SAM results accepted, the "find all" walkthrough, *Mark empty*.
  While you are still drawing a polygon, `Ctrl+Z` removes its last point. Undone changes are saved like any edit.
- Each image keeps its own history (up to 100 steps) while the editor is open — go to the next image and back, and
  you can still undo there. The history is cleared when you leave the editor.
- **Quick Review → ⏏ Exit** (or `Esc`) returns to the project Home.

---

## 14. A new class on an already-annotated dataset

Import or upload an annotated dataset, add the new class on the Classes page, then Auto-label → *Find all by concept*:
tick **only the new class** (*All / none* clears the rest), edit its phrase if needed, set *Scope* to **All images**, run.
SAM 3 adds the new class (boxes or outlines, matching the project) and never changes or deletes what is already there;
a new box that would sit on an already-labeled object is skipped. Images that got new labels move to *To review*
(the new labels are unchecked predictions); the others keep their status.

Tested on the segmentation copy of Shapes: 89 approved images without hexagons → "Hexagon" added → 75 of 75
hexagons found, 1 wrong outline, earlier annotations unchanged on all 89 images, 41 images back in To review.

*Run a model* is not the tool for this: it predicts every class the model knows (already-boxed objects are skipped,
but near-misses are not), and Bootstrap only labels images without human labels.

---

## 15. Fixes from the VisDrone end-to-end test (2026-11-08)

What changed for you (details and the test numbers: `~/yaap-visdrone-test/2026-11-08/report/REPORT.md`):

- **Editor never loses an edit.** An edit made while a save is running is saved right after it; if the server doesn't take
  a save, ←/→ / Exit stay on the image (the save retries every 5 s); closing the tab with unsaved edits asks first.
- **Changes made elsewhere are merged.** If a job (find-all, Run a model, Bootstrap) or another tab changes the image you
  have open, your next save merges: your edits stay, their new boxes are added. Boxes of a class deleted elsewhere are
  dropped with a message instead of blocking every save.
- **Dense images:** a click picks the smallest box under the cursor (Alt-click cycles through a stack); dragging a box no
  longer rebuilds the annotation list on every mouse move (4 fps → smooth with 900 boxes); inside a tiny box a drag moves it.
- **Class keys:** `1`–`9` and `0` (10th class) label a box that waits for its class; `Enter` labels it with the active
  class; drawing the next box labels the waiting one with the active class instead of dropping it.
- **Drawing:** boxes are clipped at the left/top edge too; at high zoom 1–4 px objects can be boxed; a window resize keeps
  the zoom; double-click closes a polygon without a duplicate point; the class picker stays inside the canvas.
- **SAM:** an answer that arrives after you moved to the next image is dropped; clicks made while SAM is busy are sent once
  it answers; find-all runs at SAM 3's native 1008 px.
- **Review:** Approve/Reject buttons ignore double clicks and un-approving an image without labels makes it *unannotated*
  again (it used to count as labeled and train Bootstrap as background); a fast second `R` in Quick Review no longer
  rejects the next image unseen.
- **Uploads / import / export:** corrupt files leave nothing behind; rotated phone photos are stored upright; TIFF is stored
  as PNG; an import gives `train/x.jpg` the labels of `train/x.txt` (not `valid/x.txt`) and skips a malformed label line
  with a message; class names stay unique and new classes never reuse an export index; versions list images that could not
  be exported (⚠) instead of dropping them silently; downloaded `data.yaml` files work on any machine; two raw exports at
  once both work; Test model always uses the version's own test split.
- **Training offline:** pretrained weights in `data/models` are used instead of re-downloaded; the AMP-check weights
  (`yolo26n.pt`) and the plot font are preloaded; a root-owned `data/models/Ultralytics` no longer crashes training. Use
  `docker compose exec -u yaap yaap …` for commands inside the container.
- **Idle shutdown:** closing one of several YAAP tabs no longer stops the app; a deferred shutdown doesn't survive a restart.
- **Database:** WAL mode and a 30 s lock wait — a busy database no longer fails a whole find-all job.
- **Classification and segmentation (2026-11-09):** the classification editor opens again (it crashed), class keys tag
  the image; a project only accepts its own kind of label; Bootstrap has an *Image size* field (224 for classification);
  SAM clicks reuse the image's encoding (2.7 s → 0.2 s per click); the Generate page shows images that keep an imported
  split, with *Re-split everything*.
