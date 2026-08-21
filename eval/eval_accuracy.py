"""Score helmet-classification accuracy against the hand-labelled crop set.

Run from the project root:

    python eval/eval_accuracy.py                  # score the shipped config
    python eval/eval_accuracy.py --sweep          # re-run the tuning sweep

The crops in eval/crops/ were sampled across the full input video through the
same code path the app uses (detection/crops.py), then each subject's own head
was zoomed and labelled by eye. Genuinely ambiguous crops (motion blur,
turbans/caps, non-head objects) are listed in labels.json under
EXCLUDED_AMBIGUOUS and are not scored - guessing on those would measure the
labeller's coin-flips rather than the model.

This exists so accuracy changes are measured rather than eyeballed: a change
that looks better on one frame very often scores worse over 35.
"""

import argparse
import json
import os
import sys

import cv2
import supervision as sv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.association import HELMET, NO_HELMET, is_on_own_head  # noqa: E402
from detection.config import (  # noqa: E402
    HEAD_CORE_WIDTH_FRACTION,
    HELMET_INFERENCE_IMGSZ,
    HELMET_TTA_HFLIP,
    PERSON_CROP_HEIGHT_FRACTION,
    PipelineConfig,
)
from detection.crops import compute_head_crop_box  # noqa: E402
from detection.models import load_yolo_model, validate_helmet_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_W, SOURCE_H = 1280, 720  # resolution the crops were sampled at


def load_set():
    meta = {r["idx"]: r for r in json.load(open(f"{HERE}/crops_meta.json"))}
    raw = json.load(open(f"{HERE}/labels.json"))
    truth = {i: HELMET for i in raw["HELMET"]}
    truth.update({i: NO_HELMET for i in raw["NO_HELMET"]})
    return meta, truth


def core_box_for(rec):
    """Subject-validation box in crop-local coords - mirrors pipeline.py."""
    x1, y1, x2, y2 = rec["bbox"]
    cb = compute_head_crop_box((x1, y1, x2, y2), SOURCE_W, SOURCE_H)
    ox, oy = int(round(cb[0])), int(round(cb[1]))
    bw, bh = x2 - x1, y2 - y1
    cx1 = x1 + bw * (1 - HEAD_CORE_WIDTH_FRACTION) / 2
    cx2 = x2 - bw * (1 - HEAD_CORE_WIDTH_FRACTION) / 2
    return (cx1 - ox, y1 - oy, cx2 - ox, y1 + bh * PERSON_CROP_HEIGHT_FRACTION - oy)


def label_of(names, class_id):
    n = str(names[class_id]).strip().lower()
    return NO_HELMET if ("no" in n or "without" in n) else HELMET


def classify(model, img, conf, imgsz, hflip, core, iou_thr):
    """Best on-own-head detection for one crop, max-fused across TTA views."""
    batch = [img] + ([cv2.flip(img, 1)] if hflip else [])
    outputs = model.model(batch, conf=conf, imgsz=imgsz, device=model.device, verbose=False)
    best = None
    for view, output in enumerate(outputs):
        dets = sv.Detections.from_ultralytics(output)
        for class_id, box, confidence in zip(dets.class_id, dets.xyxy, dets.confidence):
            b = tuple(float(v) for v in box)
            if view == 1:
                b = (img.shape[1] - b[2], b[1], img.shape[1] - b[0], b[3])
            if not is_on_own_head(b, core, iou_thr):
                continue
            if best is None or confidence > best[1]:
                best = (label_of(model.names, int(class_id)), float(confidence))
    return best


def evaluate(model, meta, truth, conf, imgsz, hflip, iou_thr):
    covered = correct = 0
    per_class = {HELMET: [0, 0], NO_HELMET: [0, 0]}
    for idx, gt in sorted(truth.items()):
        rec = meta[idx]
        img = cv2.imread(f"{HERE}/crops/{rec['file']}")
        best = classify(model, img, conf, imgsz, hflip, core_box_for(rec), iou_thr)
        per_class[gt][1] += 1
        if best is None:
            continue
        covered += 1
        if best[0] == gt:
            correct += 1
            per_class[gt][0] += 1
    n = len(truth)
    return {
        "coverage": covered / n,
        "accuracy": correct / covered if covered else 0.0,
        "helmet_recall": per_class[HELMET][0] / per_class[HELMET][1],
        "nohelmet_recall": per_class[NO_HELMET][0] / per_class[NO_HELMET][1],
    }


def show(tag, r):
    print(f"{tag:<40} coverage={r['coverage']:.2f}  accuracy={r['accuracy']:.3f}  "
          f"helmet_recall={r['helmet_recall']:.2f}  nohelmet_recall={r['nohelmet_recall']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="re-run the tuning sweep")
    args = ap.parse_args()

    cfg = PipelineConfig()
    meta, truth = load_set()
    model = load_yolo_model(cfg.helmet_model_path)
    validate_helmet_model(model.names)

    print(f"helmet model: {os.path.basename(cfg.helmet_model_path)}  classes={model.names}")
    print(f"labelled crops: {len(truth)} "
          f"(helmet={sum(1 for v in truth.values() if v == HELMET)}, "
          f"no-helmet={sum(1 for v in truth.values() if v == NO_HELMET)})\n")

    show("SHIPPED CONFIG", evaluate(model, meta, truth, cfg.helmet_confidence,
                                    HELMET_INFERENCE_IMGSZ, HELMET_TTA_HFLIP,
                                    cfg.iou_threshold))
    if not args.sweep:
        return

    print("\n-- what each shipped choice is worth --")
    show("640 (ultralytics default), conf .25, no TTA",
         evaluate(model, meta, truth, 0.25, 640, False, cfg.iou_threshold))
    show("imgsz 800, conf .25, no TTA",
         evaluate(model, meta, truth, 0.25, 800, False, cfg.iou_threshold))
    show("imgsz 800, conf .25, +flip TTA",
         evaluate(model, meta, truth, 0.25, 800, True, cfg.iou_threshold))
    show("imgsz 800, conf .12, +flip TTA (shipped)",
         evaluate(model, meta, truth, 0.12, 800, True, cfg.iou_threshold))

    print("\n-- imgsz sweep (conf .12, +flip TTA) --")
    for s in (416, 512, 640, 800, 928):
        show(f"imgsz={s}", evaluate(model, meta, truth, 0.12, s, True, cfg.iou_threshold))

    print("\n-- confidence sweep (imgsz 800, +flip TTA) --")
    for c in (0.08, 0.12, 0.18, 0.25, 0.35):
        show(f"conf={c}", evaluate(model, meta, truth, c, 800, True, cfg.iou_threshold))


if __name__ == "__main__":
    main()
