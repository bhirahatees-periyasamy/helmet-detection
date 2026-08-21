"""Per-person helmet status resolution.

Each tracked person's head region is cropped and run through the helmet
model individually (see detection/crops.py), so association no longer has
to guess which of several frame-wide detections belongs to which person.
What's left is: pick the best candidate detection for a person's crop,
sanity-check it actually lands on their own head (not just padding/context
picked up by a neighboring person), and fold in the "can't tell" cases.

A single frame with no usable detection (occlusion, distance, a head partly
outside the frame, a plain model miss) is recorded internally as UNKNOWN -
it does NOT count as evidence for either HELMET or NO_HELMET when voting.
UNKNOWN never reaches the UI, though: StatusSmoother always resolves each
track to a concrete HELMET/NO_HELMET verdict, using only the frames that
actually had real evidence and falling back to DEFAULT_STATUS on a track
that has none yet. Per-frame abstention is still worth keeping even though
the verdict is always concrete - it stops a single bad frame from being
weighed as real evidence and drowning out a track's more reliable frames.
"""

import collections
from dataclasses import dataclass, field

from detection.config import EDGE_MARGIN_FRACTION, MIN_PERSON_BOX_AREA

HELMET = "HELMET"
NO_HELMET = "NO_HELMET"
UNKNOWN = "UNKNOWN"  # internal per-frame "no evidence" marker - never shown to the user

# Conservative choice for a track with no real evidence yet: flag it for a
# human to check rather than silently assuming a helmet is present.
DEFAULT_STATUS = NO_HELMET


@dataclass
class PersonStatus:
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    raw_status: str
    confidence: float | None
    helmet_bbox: tuple | None = None
    status: str = field(default=UNKNOWN)  # filled in after smoothing


def _touches_edge(bbox, frame_width, frame_height) -> bool:
    x1, y1, x2, y2 = bbox
    margin_x = frame_width * EDGE_MARGIN_FRACTION
    margin_y = frame_height * EDGE_MARGIN_FRACTION
    return x1 <= margin_x or y1 <= margin_y or x2 >= frame_width - margin_x or y2 >= frame_height - margin_y


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center_inside(box, container) -> bool:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    x1, y1, x2, y2 = container
    return x1 <= cx <= x2 and y1 <= cy <= y2


def is_on_own_head(detection_bbox, head_core_bbox, iou_threshold) -> bool:
    """Reject a detection that mostly falls in the crop's padding margin
    rather than this person's own (unpadded) head region - a sign it
    actually belongs to a neighboring person caught in the crop's context."""
    return _center_inside(detection_bbox, head_core_bbox) or _iou(detection_bbox, head_core_bbox) >= iou_threshold


def build_person_statuses(
    person_tracks,
    helmet_by_track: dict,
    helmet_class_id: int,
    frame_width: int,
    frame_height: int,
    min_person_box_area: int = MIN_PERSON_BOX_AREA,
) -> list[PersonStatus]:
    """
    person_tracks: list of (track_id, bbox) for currently tracked people.
    helmet_by_track: {track_id: (class_id, bbox_in_frame_coords, confidence)},
        already isolated to that person's own head crop.
    """
    statuses = []
    for track_id, bbox in person_tracks:
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        trustworthy = area >= min_person_box_area and not _touches_edge(bbox, frame_width, frame_height)

        match = helmet_by_track.get(track_id)
        if not trustworthy or match is None:
            status, confidence, helmet_bbox = UNKNOWN, None, None
        else:
            class_id, det_bbox, confidence = match
            status = HELMET if class_id == helmet_class_id else NO_HELMET
            helmet_bbox = det_bbox

        statuses.append(
            PersonStatus(track_id=track_id, bbox=bbox, raw_status=status, confidence=confidence, helmet_bbox=helmet_bbox)
        )
    return statuses


class StatusSmoother:
    """Per-track confidence-weighted vote over a rolling window of raw statuses.

    Prevents a single missed detection or spurious classification from
    flickering a person's displayed status frame-to-frame:

    - Only frames with real evidence (HELMET/NO_HELMET) vote; per-frame
      abstentions (UNKNOWN) are skipped entirely rather than diluting the
      votes that actually mean something.
    - Votes are weighted by the detector's confidence, not counted equally,
      so one clear 0.9 reading isn't outvoted by a handful of marginal 0.15
      ones. Marginal frames are exactly where this model is least reliable,
      and the confidence threshold is deliberately low (see
      config.helmet_confidence) to keep coverage up - weighting is what
      stops that extra coverage from costing accuracy.
    - A track with no real evidence at all yet reports DEFAULT_STATUS, so
      every displayed verdict is always a concrete HELMET/NO_HELMET.
    """

    def __init__(self, window: int):
        self._window = window
        self._history: dict[int, collections.deque] = {}

    def update(self, person_statuses: list[PersonStatus]) -> list[PersonStatus]:
        for ps in person_statuses:
            history = self._history.setdefault(
                ps.track_id, collections.deque(maxlen=self._window)
            )
            history.append((ps.raw_status, ps.confidence))
            weights: dict[str, float] = {}
            for status, confidence in history:
                if status == UNKNOWN:
                    continue
                weights[status] = weights.get(status, 0.0) + (confidence or 0.0)
            ps.status = max(weights, key=weights.get) if weights else DEFAULT_STATUS
        return person_statuses

    def reset(self):
        self._history.clear()
