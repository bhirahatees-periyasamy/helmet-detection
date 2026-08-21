"""Per-person head-region cropping for the helmet model.

Running the helmet model on the full frame forces it to both localize and
classify a tiny (often <20px) head region inside a 1280x720 image - a much
harder task than what it was trained on (mostly head/shoulder-scale crops).
Cropping each tracked person's head region and upscaling it removes the
localization burden entirely (only classification remains) and matches the
model's training scale far more closely, which substantially raises
confidence for both classes - especially the visually-subtler NO-Hardhat one.
"""

import cv2

from detection.config import (
    CROP_MAX_UPSCALE,
    CROP_PADDING_FRACTION,
    CROP_TARGET_SIZE,
    PERSON_CROP_HEIGHT_FRACTION,
)


def compute_head_crop_box(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    head_h = (y2 - y1) * PERSON_CROP_HEIGHT_FRACTION
    pad_w = (x2 - x1) * CROP_PADDING_FRACTION
    pad_h = head_h * CROP_PADDING_FRACTION
    cx1 = max(0.0, x1 - pad_w)
    cy1 = max(0.0, y1 - pad_h)
    cx2 = min(float(frame_width), x2 + pad_w)
    cy2 = min(float(frame_height), y1 + head_h + pad_h)
    return cx1, cy1, cx2, cy2


def extract_upscaled_crop(frame, crop_box, min_side_px=4):
    """Returns (crop_image, (offset_x, offset_y), scale) or None if too small."""
    x1, y1, x2, y2 = (int(round(v)) for v in crop_box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < min_side_px or y2 - y1 < min_side_px:
        return None

    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    scale = min(CROP_MAX_UPSCALE, CROP_TARGET_SIZE / max(h, w))
    if scale > 1.0:
        crop = cv2.resize(
            crop, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        scale = 1.0
    return crop, (x1, y1), scale


def crop_to_frame_box(box, offset, scale):
    ox, oy = offset
    x1, y1, x2, y2 = box
    return (x1 / scale + ox, y1 / scale + oy, x2 / scale + ox, y2 / scale + oy)
