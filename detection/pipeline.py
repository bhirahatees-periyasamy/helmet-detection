"""Per-frame processing: detect -> track -> crop heads -> classify -> smooth -> aggregate.

process_frame() is called once per video frame from the Streamlit loop in
app.py. It never constructs a model or tracker itself - those are created
once and passed in via a PipelineResources instance.
"""

from dataclasses import dataclass

import cv2
import supervision as sv

from detection.association import HELMET, StatusSmoother, build_person_statuses, is_on_own_head
from detection.config import (
    HEAD_CORE_WIDTH_FRACTION,
    HELMET_INFERENCE_IMGSZ,
    HELMET_TTA_HFLIP,
    PERSON_CROP_HEIGHT_FRACTION,
    PipelineConfig,
    STATUS_SMOOTHING_WINDOW,
)
from detection.crops import compute_head_crop_box, crop_to_frame_box, extract_upscaled_crop
from detection.models import LoadedModel
from detection.tracker import new_tracker


@dataclass
class PipelineResources:
    person_model: LoadedModel
    helmet_model: LoadedModel
    person_class_id: int
    helmet_class_id: int
    no_helmet_class_id: int
    tracker: sv.ByteTrack
    smoother: StatusSmoother

    @classmethod
    def build(cls, person_model, helmet_model, person_class_id, helmet_class_id,
              no_helmet_class_id, source_fps=None):
        return cls(
            person_model=person_model,
            helmet_model=helmet_model,
            person_class_id=person_class_id,
            helmet_class_id=helmet_class_id,
            no_helmet_class_id=no_helmet_class_id,
            tracker=new_tracker(source_fps),
            smoother=StatusSmoother(STATUS_SMOOTHING_WINDOW),
        )

    def reset(self, source_fps=None):
        self.tracker = new_tracker(source_fps)
        self.smoother.reset()


@dataclass
class FrameResult:
    person_statuses: list
    person_detections: sv.Detections
    counts: dict
    compliance: float | None


def _classify_head_crops(frame, person_tracks, resources, config):
    """Runs the helmet model once per tracked person's head crop (batched)
    and returns {track_id: (class_id, bbox_in_frame_coords, confidence)},
    keeping only the best on-head detection per person.
    """
    frame_height, frame_width = frame.shape[:2]

    crops, crop_meta, core_boxes = [], [], []
    for track_id, bbox in person_tracks:
        crop_box = compute_head_crop_box(bbox, frame_width, frame_height)
        extracted = extract_upscaled_crop(frame, crop_box)
        if extracted is None:
            continue
        crop_img, offset, scale = extracted
        crops.append(crop_img)
        crop_meta.append((track_id, offset, scale))
        # Unpadded, horizontally-trimmed head region, in the crop's own local
        # (pre-upscale) coords - used to reject detections that belong to a
        # neighboring person swept in by the crop's padding/full box width
        # (common with multiple riders sharing one motorcycle).
        x1, y1, x2, y2 = bbox
        head_h = (y2 - y1) * PERSON_CROP_HEIGHT_FRACTION
        box_w = x2 - x1
        core_x1 = x1 + box_w * (1 - HEAD_CORE_WIDTH_FRACTION) / 2
        core_x2 = x2 - box_w * (1 - HEAD_CORE_WIDTH_FRACTION) / 2
        core_boxes.append(
            ((core_x1 - offset[0]) * scale, (y1 - offset[1]) * scale,
             (core_x2 - offset[0]) * scale, (y1 + head_h - offset[1]) * scale)
        )

    helmet_by_track = {}
    if not crops:
        return helmet_by_track

    # Original crops first, then (optionally) the same crops mirrored, all in a
    # single batched inference call - one model invocation per frame regardless
    # of how many people are tracked or whether TTA is on.
    batch = list(crops)
    if HELMET_TTA_HFLIP:
        batch += [cv2.flip(crop, 1) for crop in crops]

    outputs = resources.helmet_model.model(
        batch,
        conf=config.helmet_confidence,
        imgsz=HELMET_INFERENCE_IMGSZ,
        device=resources.helmet_model.device,
        verbose=False,
    )

    n = len(crops)
    for i, ((track_id, offset, scale), core_box) in enumerate(zip(crop_meta, core_boxes)):
        crop_width = crops[i].shape[1]
        # Gather this crop's detections from the original view and, if enabled,
        # from its mirrored view with boxes un-mirrored back into crop coords.
        candidates = []
        for view, output in ((0, outputs[i]), (1, outputs[i + n] if HELMET_TTA_HFLIP else None)):
            if output is None:
                continue
            dets = sv.Detections.from_ultralytics(output)
            for class_id, box, confidence in zip(dets.class_id, dets.xyxy, dets.confidence):
                b = tuple(float(v) for v in box)
                if view == 1:
                    b = (crop_width - b[2], b[1], crop_width - b[0], b[3])
                candidates.append((int(class_id), b, float(confidence)))

        # Keep the single most confident detection that lands on this person's
        # own head - i.e. max-fuse across the two views. Mean-fusion was
        # measured worse: a head only one view recognises still carries real
        # information, and averaging it against the other view's silence
        # discards that.
        best = None
        for class_id, box, confidence in candidates:
            if not is_on_own_head(box, core_box, config.iou_threshold):
                continue
            if best is None or confidence > best[2]:
                best = (class_id, box, confidence)
        if best is not None:
            class_id, box, confidence = best
            helmet_by_track[track_id] = (class_id, crop_to_frame_box(box, offset, scale), confidence)

    return helmet_by_track


def process_frame(frame, resources: PipelineResources, config: PipelineConfig) -> FrameResult:
    frame_height, frame_width = frame.shape[:2]

    person_output = resources.person_model.model(
        frame,
        classes=[resources.person_class_id],
        conf=config.person_confidence,
        device=resources.person_model.device,
        verbose=False,
    )[0]
    person_detections = sv.Detections.from_ultralytics(person_output)
    person_detections = resources.tracker.update_with_detections(person_detections)

    person_tracks = [
        (int(tracker_id), tuple(float(v) for v in box))
        for tracker_id, box in zip(person_detections.tracker_id, person_detections.xyxy)
        if tracker_id is not None
    ]

    helmet_by_track = _classify_head_crops(frame, person_tracks, resources, config)

    person_statuses = build_person_statuses(
        person_tracks, helmet_by_track, resources.helmet_class_id, frame_width, frame_height,
    )
    person_statuses = resources.smoother.update(person_statuses)

    counts = {"people": len(person_statuses), "helmet": 0, "no_helmet": 0}
    for ps in person_statuses:
        if ps.status == HELMET:
            counts["helmet"] += 1
        else:
            counts["no_helmet"] += 1

    compliance = (counts["helmet"] / counts["people"]) if counts["people"] > 0 else None

    return FrameResult(
        person_statuses=person_statuses,
        person_detections=person_detections,
        counts=counts,
        compliance=compliance,
    )
