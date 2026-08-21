"""Configurable parameters for the helmet detection pipeline."""

import glob
import os
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# YOLO26x - the largest tier of the newest YOLO family (n/s/m/l/x; there is no
# "xl"). COCO-pretrained, so it contributes ONLY person boxes and identity - it
# has no helmet-related class, and the helmet call always comes from the second
# model below.
#
# Measured on this video against yolo11x: yolo26x is the better-calibrated but
# more *conservative* detector - higher mean confidence (0.73 vs 0.67) yet ~8%
# FEWER person detections at any given threshold. Since a rider who is never
# detected never gets a helmet verdict at all, that shortfall matters, so the
# person threshold below is lowered to 0.30 to compensate; at 0.30 yolo26x
# recovers yolo11x@0.40's recall (2.98 vs 2.88 detections/frame) while keeping
# the better calibration. yolo11x.pt and yolo26m.pt are both still in models/;
# yolo26m is ~2x faster if throughput matters more than recall.
DEFAULT_PERSON_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo26x.pt")
# best.pt from the Hugging Face repo sharathhhhh/safetyHelmet-detection-yolov8,
# saved under a filename that echoes its origin so it's obvious in the sidebar
# and in the startup log which weights are actually doing the helmet call:
#   curl -L -o models/safetyHelmet_yolov8_best.pt \
#     https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt
# (SHA-256 06297f6c2d27bd157297866e526710e8ffc06b5c04da28ab77db949e805c141c)
#
# This is the ONLY model that decides helmet vs. no-helmet - the person model
# above is COCO-pretrained and has no helmet class at all. It's trained on
# rider CCTV/surveillance footage (classes "With helmet"/"Without helmet"),
# which matches this project's traffic video far better than a construction
# hard-hat dataset: a full-face motorcycle helmet doesn't look much like a
# rigid, brimmed construction hard hat, and that domain mismatch was measured
# to cause both misses (under-confident on real helmets) and false positives
# (misreading an open-visor helmet's visible face as bare).
DEFAULT_HELMET_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "safetyHelmet_yolov8_best.pt")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "video")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Fraction of a person's bounding-box height, measured from the top, that
# gets cropped out and fed to the helmet model. >1.0 (the full body box plus
# some room below it) rather than a tight head-only crop: this model was
# trained on typical CCTV framing showing a rider's head, shoulders, and
# some of the bike/body below - empirically (swept against known ground
# truth crops from this video) both a tight head crop and the bare person
# box under-perform a somewhat more generous region.
PERSON_CROP_HEIGHT_FRACTION = 1.1

# Extra margin (fraction of the crop region's own size) added around it
# before cropping, so the helmet model sees more context. Also swept
# empirically - both smaller and much larger margins measured worse.
CROP_PADDING_FRACTION = 0.5

# A person's actual head is narrower than their full shoulder-width bounding
# box and roughly centered in it. The "core" region used to validate a
# detection belongs to THIS person (not a neighbor caught by crop padding -
# common on shared motorcycles/crowds) is trimmed to this central fraction
# of the box width.
HEAD_CORE_WIDTH_FRACTION = 0.7

# Manual pre-upscaling (via cv2.resize) was measured to bias predictions on
# two different helmet models - it adds blur that erases exactly the sharp
# edges/texture the model relies on. ultralytics already letterbox-resizes
# whatever crop it's given internally, so CROP_MAX_UPSCALE=1.0 means "don't
# resize ourselves at all" - just crop, and let the model's own preprocessing
# handle scale. CROP_TARGET_SIZE is kept only as the (now inactive) knob for
# that upscaling, should a future model actually benefit from it.
CROP_TARGET_SIZE = 160
CROP_MAX_UPSCALE = 1.0

# Inference resolution ultralytics letterboxes each head crop to before running
# the helmet model. Swept against a 35-crop hand-labelled set from this video:
# the 640 default scored 0.79 accuracy / 0.64 helmet-recall, while 800 scored
# 0.89 / 0.79. Larger still (960+) drops off sharply as the letterbox padding
# starts to dominate the (small) real content.
HELMET_INFERENCE_IMGSZ = 800

# Also run each crop mirrored and keep whichever view is more confident
# (test-time augmentation). On the same labelled set this lifted accuracy from
# 0.89 -> 0.935 and helmet-recall from 0.79 -> 0.93: this model is noticeably
# direction-sensitive, so a rider facing one way can be classified confidently
# while the mirrored view of the same head is not. Multi-SCALE ensembling was
# also tried and measured *worse* (a small scale contributes confident wrong
# detections that win the max-fusion), so only the flip is used.
HELMET_TTA_HFLIP = True

# A person box is too small (far away / low-res head) to trust a helmet
# classification if its area is below this many pixels.
MIN_PERSON_BOX_AREA = 2000

# A person box within this fraction of the frame width/height from any edge
# is treated as "head possibly outside the frame" -> forced UNKNOWN, per the
# same reasoning as MIN_PERSON_BOX_AREA: a couple of pixels of margin isn't
# enough to rule out a clipped head.
EDGE_MARGIN_FRACTION = 0.02

# Number of past frames' raw status kept per track for majority-vote smoothing.
STATUS_SMOOTHING_WINDOW = 15


def discover_video_path() -> str | None:
    """Return the first supported video file found in VIDEO_DIR, if any."""
    for ext in VIDEO_EXTENSIONS:
        matches = sorted(glob.glob(os.path.join(VIDEO_DIR, f"*{ext}")))
        if matches:
            return matches[0]
    return None


@dataclass
class PipelineConfig:
    person_model_path: str = DEFAULT_PERSON_MODEL_PATH
    helmet_model_path: str = DEFAULT_HELMET_MODEL_PATH
    video_path: str | None = None
    # 0.30 rather than 0.40 to offset yolo26x being a more conservative
    # detector than yolo11x (see DEFAULT_PERSON_MODEL_PATH above) - this is
    # what keeps person recall from regressing with the newer model. If you
    # switch the person model back to yolo11x/yolo26m, 0.40 is the better
    # pairing there.
    person_confidence: float = 0.30
    # 0.12 rather than 0.25: with the imgsz/TTA settings above, lowering the
    # threshold raised coverage (0.83 -> 0.89 of people getting any verdict)
    # and no-helmet recall (0.67 -> 0.76) on the labelled set with no loss of
    # accuracy among the covered, because the flip-ensemble already filters
    # out most of the weak spurious detections a low threshold lets through.
    helmet_confidence: float = 0.12
    iou_threshold: float = 0.35

    def __post_init__(self):
        if self.video_path is None:
            self.video_path = discover_video_path()
