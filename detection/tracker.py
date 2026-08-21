"""ByteTrack wrapper for stable person tracking IDs across frames."""

import warnings

import supervision as sv


def new_tracker(source_fps: float | None = None) -> sv.ByteTrack:
    """Create a fresh ByteTrack instance.

    Called once at Start and again on Reset - never inside the frame loop,
    so tracker state (and therefore track IDs) persists frame-to-frame.

    supervision==0.30.0 marks ByteTrack deprecated (removal in 0.31.0) but
    ships no replacement class yet, so it's still the correct API to use.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return sv.ByteTrack(frame_rate=int(round(source_fps)) if source_fps else 30)
