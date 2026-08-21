# Real-Time Helmet Detection

A Streamlit app that plays a video frame-by-frame through a two-model YOLO pipeline
and shows, live as the video plays, whether each tracked person is wearing a
helmet or not — every tracked person always gets a concrete `HELMET`/`NO_HELMET`
verdict; see "No UNKNOWN status" below for how that's reconciled with not just
guessing on hard cases.

## Models

Two independent models are used because no single available model reliably
provides both stable full-body person boxes *and* real helmet/no-helmet classes.

| Role | Model file | Source | Classes |
|---|---|---|---|
| Person detection + tracking | `models/yolo26x.pt` | Ultralytics (auto-downloaded, COCO-pretrained) | 80 COCO classes; only `person` is used |
| Helmet / no-helmet call | `models/safetyHelmet_yolov8_best.pt` | [sharathhhhh/safetyHelmet-detection-yolov8](https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8) → `best.pt` (SHA-256 `06297f6c…`) | `With helmet`, `Without helmet` |

The helmet weights are `best.pt` pulled straight from that Hugging Face repo,
kept under a filename that echoes its origin so the startup log and the
sidebar both make plain which weights are deciding helmet vs. no-helmet. To
re-fetch it from scratch:

```bash
curl -L -o models/safetyHelmet_yolov8_best.pt \
  https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt
```

Both are pretrained, publicly available checkpoints — no custom training was
done. **YOLO26x is COCO-pretrained and has no helmet-related class at all**, so
it contributes only person boxes and identity; every helmet/no-helmet call
comes from the second model.

### Choosing the person detector

`yolo26x` is the largest tier of the newest YOLO family (tiers are n/s/m/l/x —
there is no "xl"). It is *not* automatically the most accurate choice for this
task, which is worth knowing before swapping tiers around. Measured on this
video (8 segments spread across the hour, person threshold 0.40):

| Person model | detections/frame | mean confidence | crowded FPS |
|---|---|---|---|
| `yolo26m.pt` | 1.57 | — | 7.6 |
| `yolo11x.pt` | **1.75** | 0.67 | 5.4 |
| `yolo26x.pt` | 1.57 | **0.73** | 5.8 |

YOLO26x is the better-*calibrated* detector (noticeably higher confidence on
the people it finds, and it produced more stable track IDs) but also the more
*conservative* one — ~8% fewer person detections than YOLO11x at every
threshold tested. For helmet compliance that shortfall matters directly: a
rider who is never detected never gets a helmet verdict at all.

So the shipped pairing is **`yolo26x` at person confidence 0.30** rather than
0.40, which recovers the recall (2.98 vs YOLO11x@0.40's 2.88 detections/frame,
and 5.2 vs 5.5 avg people on a crowded stretch) while keeping the better
calibration. If you switch the person model back to `yolo11x.pt` or
`yolo26m.pt` — both still in `models/` — put the threshold back to 0.40, which
is the better pairing for those. `yolo26m` is ~30% faster if throughput
matters more than recall.

Note the helmet accuracy figures further down are unaffected by this choice:
the helmet model classifies a crop, so the person detector changes *which*
people get judged, not how well each one is judged.

**The helmet model was swapped once already.** The original choice,
`keremberke/yolov8m-hard-hat-detection`, is trained on *construction-site hard
hats* — a rigid, brimless dome shape. This project's video shows *motorcycle
riders*, whose full-face/half-face helmets look quite different (integrated
visor, rounder profile, often showing part of the face). That domain mismatch
was measured directly against this video: the hard-hat model was
under-confident on real helmets (most raw predictions landed below any
reasonable threshold) and, once cropped to isolate a single head, sometimes
read an open-visor rider's visible face as "no hardhat." Swapping to a model
trained on rider/CCTV footage (`sharathhhhh/safetyHelmet-detection-yolov8`)
fixed both failure modes on the same test cases — see "How this was tuned"
below. If you're deploying this for an actual construction-PPE scenario
instead of traffic footage, the original hard-hat model (still present at
`models/hardhat_yolov8m.pt`) is likely the better-matched choice — point
`MODEL_PATH` at it.

At every app startup, both models' `model.names` are logged and validated —
the app refuses to start with a clear error if the "person" class or a
helmet/no-helmet class pair (matched generically against names containing
"helmet"/"hardhat", with a "no"/"without" marker distinguishing the negative
class) aren't present, rather than silently mislabeling detections. Class IDs
are never hardcoded; they're looked up by name at load time (see
`detection/models.py`).

## Why two models need a crop + association step

The person model outputs full body boxes; the helmet model classifies a
single crop. Each tracked person's box is cropped out of the frame and run
through the helmet model individually (`detection/pipeline.py`,
`detection/crops.py`) rather than running the helmet model once on the full
frame — isolating one person at a time removes the burden of localizing a
small, often-distant head inside a 1280×720 frame, leaving only
classification, which the model is far more confident at.

1. The crop covers most of the person's box plus a bit of margin below and
   around it (see "How this was tuned") — not just a tight head-only region,
   since this model was trained on typical CCTV framing (head, shoulders,
   part of the bike).
2. Within that crop, the highest-confidence `With helmet`/`Without helmet`
   detection is used, **after** checking it actually lands on this person's
   own (horizontally-centered) head region rather than a neighbor's — needed
   because multiple people sharing one motorcycle is common in this footage,
   and the crop's margin can otherwise sweep in an adjacent rider's head.
3. A frame with no usable detection (occlusion, distance, a head outside the
   frame, a plain model miss, or a person box too small/too close to the
   frame edge to trust) is recorded internally as an *abstention* — it is
   not treated as evidence for either HELMET or NO_HELMET. See "No UNKNOWN
   status" for why this matters even though the app never displays it.
4. Per-track status is the plurality of only the frames that had real
   evidence (abstentions don't count as votes), smoothed over the last 15
   frames so one bad frame can't flip the displayed status. A track with no
   real evidence yet reports `NO_HELMET` by default (see below) rather than
   an arbitrary guess.

## No UNKNOWN status

Every tracked person is always shown as `HELMET` or `NO_HELMET` — there is no
third "can't tell" label in the UI. This was a deliberate choice made after
testing: a per-frame abstention concept is still used internally (a person
too small, too near the frame edge, or with no matching detection this frame
doesn't count as evidence either way), but the *displayed* status always
resolves to a concrete verdict:

- **Voting ignores abstentions.** A track's status is the plurality of only
  its real (non-abstention) observations in the last 15 frames — an
  occluded or low-confidence frame doesn't get to outvote several good
  ones, it just gets skipped.
- **A track with zero real evidence so far defaults to `NO_HELMET`**, not a
  coin flip — flagging a still-unclassified person for a human to check is
  the more conservative failure mode for a compliance tool than silently
  assuming they're compliant. This only shows briefly, for the handful of
  frames right as a person is first tracked, before real evidence takes
  over (see `detection/association.py`'s `DEFAULT_STATUS`).

This means the false-negative/false-positive tradeoff is different from a
"HELMET / NO_HELMET / UNKNOWN" design: hard cases that would otherwise be
UNKNOWN now get resolved with whatever fragmentary evidence exists (or the
conservative default), which can occasionally be wrong on a genuinely
ambiguous case rather than honestly saying so. That tradeoff was requested
explicitly rather than a default — if you want the honest "can't tell"
signal back for a different deployment, `build_person_statuses` and
`StatusSmoother` in `detection/association.py` are where to reintroduce it.

Manual pre-upscaling of small crops (via `cv2.resize`) was tried and measured
to *hurt* accuracy — the added blur erases exactly the sharp edges/texture
the model uses to recognize a helmet, and it was seen to flip a correct
"with helmet" call to "without" on multiple crops from this video. Crops are
now passed through close to their native resolution and left to
`ultralytics`' own internal letterbox-resize (`CROP_MAX_UPSCALE = 1.0` in
`detection/config.py` disables the extra manual step; the knob is kept for a
future model that might actually benefit from it).

## How this was tuned (and measured)

Early tuning used two known ground-truth tracks from this video (one rider
verifiably in a full-face helmet across ~40 frames, one verifiably
bare-headed) to sweep the crop geometry — `PERSON_CROP_HEIGHT_FRACTION=1.1`
and `CROP_PADDING_FRACTION=0.5` won; both a tight head-only crop and a much
larger one scored worse.

Two tracks is too thin to tune a classifier on, so accuracy work since then
runs against a **35-crop hand-labelled evaluation set**: crops were sampled
across the full hour via the exact same code path the app uses, then each
subject's own head was zoomed and labelled by eye as helmet / no-helmet, with
15 genuinely ambiguous crops (motion-blurred, turbans/caps, non-head objects)
excluded rather than guessed at. Every config below was scored on that set, so
these are measurements rather than impressions:

| Config | Accuracy (of those classified) | Helmet recall | Coverage |
|---|---|---|---|
| Original (`imgsz` default 640, conf 0.25, no TTA) | 0.79 | 0.64 | 0.83 |
| `imgsz=800` | 0.89 | 0.79 | 0.80 |
| `imgsz=800` + horizontal-flip TTA | 0.93 | 0.93 | 0.83 |
| **`imgsz=800` + flip TTA + conf 0.12 (shipped)** | **0.94** | **0.93** | **0.89** |

What each change does, and why:

- **`HELMET_INFERENCE_IMGSZ = 800`** — the crops are small, and letting
  ultralytics letterbox them to its 640 default was throwing away
  resolution the classifier needed. 960+ makes it worse again, as padding
  starts to dominate the real content.
- **`HELMET_TTA_HFLIP = True`** — this model turns out to be noticeably
  direction-sensitive: the same head can be confidently classified facing
  one way and missed mirrored. Running both views and keeping the more
  confident one lifted helmet recall from 0.79 to 0.93 — the single biggest
  win. Both views go through in **one batched inference call**, so it stays
  one model invocation per frame. Multi-*scale* ensembling was also tried and
  measured **worse** (a small scale contributes confident wrong detections
  that win the fusion), so only the flip is used.
- **`helmet_confidence = 0.12`** (was 0.25) — with the flip ensemble already
  suppressing weak spurious detections, a lower threshold buys coverage
  (0.83 → 0.89 of people getting a verdict) and no-helmet recall
  (0.67 → 0.76) at no cost to accuracy.
- **Confidence-weighted temporal voting** — the per-track window now weights
  each frame's vote by detector confidence instead of counting frames
  equally, so one clear 0.9 reading isn't outvoted by a handful of marginal
  0.15 ones. This pairs with the lower confidence threshold above.

Things that were tried and measured as *not* helping, and so were not
shipped: multi-scale ensembling (above), `augment=True` (ultralytics' own TTA
— slightly worse than the plain flip), ensembling the rider model with the
hard-hat model, and tightening the subject-validation geometry
(`head_band` / `width_frac` sweeps moved accuracy by 0.00 — confirming the
remaining errors are the classifier's, not the association logic's).

All of this is tuned to *this* model on *this* footage. A different model or a
much closer/farther camera angle would need re-running the same sweep rather
than assuming these defaults transfer.

### Re-measuring accuracy

The labelled set and the scorer are committed, so any future change can be
checked instead of eyeballed:

```bash
python eval/eval_accuracy.py            # score the current config
python eval/eval_accuracy.py --sweep    # reproduce the table above
```

`eval/crops/` holds the 50 sampled crops, `eval/labels.json` the labels
(including the 15 deliberately-excluded ambiguous ones), and
`eval/crops_meta.json` each crop's source frame and person box so the
subject-validation geometry can be reproduced exactly. The scorer imports the
real `detection/` code, so it measures the shipped path rather than a
reimplementation of it.

Worth knowing about the metrics: **coverage** is the fraction of people who
get any helmet verdict from the model at all, and **accuracy** is measured
only over those. The two trade against each other, which is why both are
reported — a config that abstains on everything hard would post excellent
accuracy and be useless. (People with no model verdict still get a displayed
status via the default described in "No UNKNOWN status".)

## Pipeline (per frame, streamed — never batched)

```
read one frame
  -> person model (class-filtered to "person") -> supervision.ByteTrack (persistent IDs)
  -> per tracked person: crop -> helmet model (With helmet / Without helmet)
  -> on-own-head check -> per-track raw status
  -> majority-vote smoothing
  -> aggregate counts + compliance
  -> annotate frame (box, ID, status, confidence, optional helmet box, summary overlay)
  -> push to Streamlit image placeholder + live metrics
```

Both models and the person tracker are constructed once and reused — never
recreated inside the frame loop. Models are cached with `st.cache_resource`;
the tracker/smoother live in `st.session_state` and are only rebuilt on
**Reset**, so **Stop** followed by **Start** resumes tracking instead of
losing identities.

## Getting the model weights

`models/` and `video/` are gitignored — the weights total ~320 MB (two of them
individually exceed GitHub's 100 MB file limit) and the sample video is
~956 MB. Both are reproducible, so after cloning fetch them once:

```bash
mkdir -p models video

# Helmet / no-helmet model (6 MB) - the only model that decides helmet status
curl -L -o models/safetyHelmet_yolov8_best.pt \
  https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt

# Person detector (113 MB). Ultralytics publishes these; this pulls it via the
# same URL ultralytics itself uses, straight into models/.
curl -L -o models/yolo26x.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt
```

Then drop any `.mp4`/`.avi`/`.mov`/`.mkv` into `video/` — the app
auto-discovers whatever is there. If a weight file is missing the app fails
fast with an explicit "Model weights not found" error rather than a traceback.

Optional alternates referenced in this README (`yolo11x.pt`, `yolo26m.pt` for
the person/speed trade, `hardhat_yolov8m.pt` for construction-PPE footage) can
be fetched the same way, swapping the filename.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The video auto-discovered from `video/` (any `.mp4`/`.avi`/`.mov`/`.mkv`) is
used by default; override the path, model paths, and thresholds from the
sidebar. Click **Start**/**Stop**/**Reset** to control playback — Reset
rewinds the video and clears tracking IDs, counts, and the unique-people set.

### Playback controls

- **Seek**: drag the "Seek to" slider to any point in the video and click
  **Jump** (only while stopped) to preview that frame and have the next
  **Start** resume from there. Jumping resets the tracker/smoother, since
  track IDs from before a jump don't mean anything after it.
- **Speed**: the "Playback speed" sidebar dropdown (0.5x/1x/2x/4x). Since
  the two-model pipeline is already the bottleneck below the source video's
  frame rate (see Throughput below), speeds ≥1x are achieved by skipping
  frames rather than sleeping less — a "2x" run scores every other frame
  instead of racing to process every one, matching how a real video player's
  fast-forward works. 0.5x instead stretches the per-frame delay, since that
  direction doesn't need skipping.

## Configuration

All configurable via the sidebar (see `detection/config.py` for defaults):

- `MODEL_PATH` (person and helmet, independently)
- `VIDEO_PATH` (auto-discovered, overridable)
- `PERSON_CONFIDENCE` (default 0.30 — paired with `yolo26x`; use 0.40 if you
  switch the person model to `yolo11x`/`yolo26m`, see above)
- `HELMET_CONFIDENCE` (default 0.12 — see the accuracy table above)
- `IOU_THRESHOLD` — on-own-head containment threshold (default 0.35)
- Playback speed (0.5x/1x/2x/4x)

Crop geometry (`PERSON_CROP_HEIGHT_FRACTION`, `CROP_PADDING_FRACTION`,
`HEAD_CORE_WIDTH_FRACTION`), the helmet-model inference settings
(`HELMET_INFERENCE_IMGSZ`, `HELMET_TTA_HFLIP`) and the edge/size abstention
gates (`MIN_PERSON_BOX_AREA`, `EDGE_MARGIN_FRACTION`) are in
`detection/config.py` rather than the sidebar, since they're tuned to the
current model pairing rather than something to adjust per-run.

## Metrics

Per-frame: People / With Helmet / Without Helmet, plus **Unique People
Seen** (distinct track IDs across the whole run) and **Processing FPS**.
Compliance = `helmet / people`, shown as `N/A` only when there are zero
tracked people this frame.

## Known limitations

- **Default test video is generic traffic footage**, not a curated
  helmet-compliance clip — it's a 1-hour, 1280×720 roadside recording with
  mostly small/distant people on motorcycles and in cars, which is
  genuinely hard for any small helmet-classification model.
- **Throughput**: measured on an Apple M4 (MPS) with the shipped
  YOLO26x + flip-TTA config: **~12 FPS on sparse frames, ~6 FPS on crowded
  ones** (7 people), against a 30 FPS source. The accuracy work above cost
  roughly half the original throughput (the x-tier person model ~30%, the
  larger helmet inference size and flip TTA the rest). This is a deliberate
  accuracy-over-speed trade; to go back the other way, point the person model
  at `yolo26m.pt` (~30% faster) and/or set `HELMET_TTA_HFLIP = False`. Note
  the app paces to the source frame rate and will simply fall behind rather
  than drop frames at 1x — use 2x/4x playback to cover the video faster by
  scoring fewer frames.
- **Association is a heuristic, not learned** — tight clusters of people
  (e.g. three riders on one motorcycle) can still confuse the on-own-head
  check when heads are only a few pixels apart; the horizontal core-region
  trim reduces but doesn't eliminate this.
- **No UNKNOWN status is shown (by request)** — every person always gets a
  concrete HELMET/NO_HELMET verdict (see "No UNKNOWN status" above). On a
  genuinely ambiguous frame (tiny, occluded, edge-of-frame, or a track with
  no real evidence yet) this means the app commits to its best guess or a
  conservative default rather than admitting it can't tell, which can read
  as a confident wrong answer on hard cases rather than an honest one.
- **The helmet model isn't perfect** — like any ~6MB single-stage detector,
  it will still occasionally misclassify, particularly at the edges of the
  confidence threshold. Vote smoothing absorbs one-off flickers but can't
  correct a call the model gets consistently wrong across many frames.
- `supervision`'s `ByteTrack` is marked deprecated as of `supervision==0.30`
  (removal planned for `0.31`) with no replacement class shipped yet in this
  version — still the correct API to use today; revisit if upgrading
  `supervision` past 0.30.
