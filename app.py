"""Real-Time Helmet Detection - Streamlit app.

Pipeline per frame (never batches the whole video):
    read frame -> person model + tracker -> helmet model
    -> head-region association -> status smoothing -> annotate -> display
"""

import collections
import logging
import os
import shutil
import tempfile
import time

import cv2
import streamlit as st

from detection.annotate import annotate_frame
from detection.config import VIDEO_EXTENSIONS, PipelineConfig, discover_video_path
from detection.models import ModelLoadError, load_yolo_model, validate_helmet_model, validate_person_model
from detection.pipeline import PipelineResources, process_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("helmet_detection")

FPS_WINDOW = 30
DISPLAY_WIDTH = 800
SPEED_OPTIONS = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def persist_upload(uploaded) -> str:
    """Spool an uploaded video to a temp file and return its path.

    cv2.VideoCapture needs a real path on disk, and Streamlit re-runs this
    script on every interaction, so the upload is written exactly once per
    distinct file (keyed by name+size) and reused afterwards. Copied in
    chunks rather than read into memory whole, since these clips run to
    hundreds of MB.
    """
    key = (uploaded.name, uploaded.size)
    existing = st.session_state.get("upload_path")
    if st.session_state.get("upload_key") == key and existing and os.path.isfile(existing):
        return existing

    if existing and os.path.isfile(existing):
        try:
            os.remove(existing)  # a new upload supersedes the previous temp file
        except OSError:
            pass

    suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
    fd, path = tempfile.mkstemp(prefix="helmet_upload_", suffix=suffix)
    try:
        uploaded.seek(0)
        with os.fdopen(fd, "wb") as dest:
            shutil.copyfileobj(uploaded, dest, length=4 * 1024 * 1024)
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise

    st.session_state.upload_key = key
    st.session_state.upload_path = path
    logger.info("Stored uploaded video %s (%.1f MB) at %s", uploaded.name, uploaded.size / 1048576, path)
    return path


@st.cache_data(show_spinner=False)
def probe_video(path: str):
    """One-off read of fps/duration for the seek slider - independent of the
    persistent capture used for playback, so probing doesn't disturb it."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return {"fps": fps, "frame_count": frame_count, "duration": frame_count / fps if fps else 0.0}


@st.cache_resource(show_spinner="Loading person detection model...")
def _load_person_model(path: str):
    loaded = load_yolo_model(path)
    class_id = validate_person_model(loaded.names)
    return loaded, class_id


@st.cache_resource(show_spinner="Loading helmet detection model...")
def _load_helmet_model(path: str):
    loaded = load_yolo_model(path)
    helmet_id, no_helmet_id = validate_helmet_model(loaded.names)
    return loaded, helmet_id, no_helmet_id


def init_state():
    defaults = {
        "running": False,
        "cap": None,
        "resources": None,
        "unique_ids": set(),
        "frame_index": 0,
        "video_ended": False,
        "last_frame_rgb": None,
        "frame_times": collections.deque(maxlen=FPS_WINDOW),
        "fps_estimate": 0.0,
        "current_counts": {"people": 0, "helmet": 0, "no_helmet": 0},
        "current_compliance": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def open_capture(video_path: str):
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    st.session_state.cap = cap
    return cap


def start_processing(config: PipelineConfig, person_bundle, helmet_bundle):
    person_model, person_class_id = person_bundle
    helmet_model, helmet_class_id, no_helmet_class_id = helmet_bundle

    if st.session_state.cap is None:
        cap = open_capture(config.video_path)
    else:
        cap = st.session_state.cap
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if st.session_state.resources is None:
        st.session_state.resources = PipelineResources.build(
            person_model, helmet_model, person_class_id, helmet_class_id,
            no_helmet_class_id, source_fps,
        )

    st.session_state.video_ended = False
    st.session_state.running = True


def seek_to(config: PipelineConfig, seconds: float, source_fps: float):
    if st.session_state.cap is None:
        open_capture(config.video_path)
    cap = st.session_state.cap
    target_frame = int(seconds * source_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if ret:
        st.session_state.last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)  # undo the 1-frame peek above
    st.session_state.frame_index = target_frame
    # Tracking identity doesn't survive a jump - stale track IDs from before
    # the seek would otherwise get silently reused for unrelated people.
    if st.session_state.resources is not None:
        st.session_state.resources.reset(source_fps)
    st.session_state.video_ended = False


def reset_all(config: PipelineConfig):
    st.session_state.running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    st.session_state.cap = cv2.VideoCapture(config.video_path)
    source_fps = st.session_state.cap.get(cv2.CAP_PROP_FPS) or 30.0
    if st.session_state.resources is not None:
        st.session_state.resources.reset(source_fps)
    st.session_state.unique_ids = set()
    st.session_state.frame_index = 0
    st.session_state.video_ended = False
    st.session_state.last_frame_rgb = None
    st.session_state.frame_times.clear()
    st.session_state.fps_estimate = 0.0
    st.session_state.current_counts = {"people": 0, "helmet": 0, "no_helmet": 0}
    st.session_state.current_compliance = None


st.set_page_config(page_title="Real-Time Helmet Detection", layout="wide")
init_state()

st.title("Real-Time Helmet Detection")

_defaults = PipelineConfig()

with st.sidebar:
    st.header("Video source")
    default_video = discover_video_path()
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=[ext.lstrip(".") for ext in VIDEO_EXTENSIONS],
        disabled=st.session_state.running,
        help="Takes precedence over the path below. Needed on a fresh clone, "
             "since video/ is gitignored.",
    )
    video_path = st.text_input(
        "…or path to a video file",
        value=default_video or "",
        disabled=st.session_state.running,
        placeholder="e.g. video/input_video.mp4",
    )

    st.header("Configuration")
    person_model_path = st.text_input(
        "Person model path", value=_defaults.person_model_path, disabled=st.session_state.running
    )
    helmet_model_path = st.text_input(
        "Helmet model path", value=_defaults.helmet_model_path, disabled=st.session_state.running
    )
    person_confidence = st.slider(
        "Person confidence", 0.0, 1.0, _defaults.person_confidence, 0.05, disabled=st.session_state.running
    )
    helmet_confidence = st.slider(
        "Helmet confidence", 0.0, 1.0, _defaults.helmet_confidence, 0.05, disabled=st.session_state.running
    )
    iou_threshold = st.slider(
        "Association IoU threshold", 0.0, 1.0, _defaults.iou_threshold, 0.05, disabled=st.session_state.running
    )
    speed_label = st.selectbox(
        "Playback speed", list(SPEED_OPTIONS.keys()), index=1, disabled=st.session_state.running
    )
    speed_multiplier = SPEED_OPTIONS[speed_label]

# An upload wins over the typed path; the typed path (which defaults to
# whatever was auto-discovered in video/) is the fallback.
resolved_video = None
if uploaded_video is not None:
    try:
        resolved_video = persist_upload(uploaded_video)
    except OSError as exc:
        st.error(f"Could not save the uploaded video: {exc}")
        st.stop()
elif video_path:
    resolved_video = video_path

config = PipelineConfig(
    person_model_path=person_model_path,
    helmet_model_path=helmet_model_path,
    video_path=resolved_video,
    person_confidence=person_confidence,
    helmet_confidence=helmet_confidence,
    iou_threshold=iou_threshold,
)

# Switching video mid-session invalidates the open capture and all tracking
# state - track IDs and counts from a different clip are meaningless here.
if st.session_state.get("active_video") != config.video_path:
    st.session_state.active_video = config.video_path
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.running = False
    st.session_state.resources = None
    st.session_state.unique_ids = set()
    st.session_state.frame_index = 0
    st.session_state.video_ended = False
    st.session_state.last_frame_rgb = None
    st.session_state.frame_times.clear()
    st.session_state.fps_estimate = 0.0
    st.session_state.current_counts = {"people": 0, "helmet": 0, "no_helmet": 0}
    st.session_state.current_compliance = None

if not config.video_path:
    st.info(
        "**No video loaded yet.** Upload one with **Upload a video** in the "
        "sidebar, or drop a `.mp4` / `.avi` / `.mov` / `.mkv` file into the "
        "project's `video/` folder and reload — it will be picked up "
        "automatically.\n\n"
        "`video/` is gitignored, so a freshly cloned copy of this project "
        "starts out with no video."
    )
    st.stop()

if not os.path.isfile(config.video_path):
    st.error(
        f"No file at `{config.video_path}`. Fix the path in the sidebar, or "
        "upload a video instead."
    )
    st.stop()

video_info = probe_video(config.video_path)
if video_info is None:
    st.error(
        f"`{os.path.basename(config.video_path)}` could not be opened or decoded. "
        "It may be corrupt, or use a codec this OpenCV build doesn't support - "
        "try re-encoding it to H.264 MP4."
    )
    st.stop()

try:
    person_bundle = _load_person_model(config.person_model_path)
    helmet_bundle = _load_helmet_model(config.helmet_model_path)
except ModelLoadError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Loaded models"):
    st.markdown(
        f"**Helmet / no-helmet call** — `{os.path.basename(config.helmet_model_path)}`  \n"
        "This is the only model that decides helmet vs. no-helmet."
    )
    st.write(helmet_bundle[0].names)
    st.markdown(
        f"**Person detection + tracking** — `{os.path.basename(config.person_model_path)}`  \n"
        "COCO-pretrained: supplies person boxes and track IDs only, and has no "
        "helmet-related class of its own."
    )
    st.write(person_bundle[0].names)

col1, col2, col3 = st.columns(3)
start_clicked = col1.button("Start", width="stretch", disabled=st.session_state.running)
stop_clicked = col2.button("Stop", width="stretch", disabled=not st.session_state.running)
reset_clicked = col3.button("Reset", width="stretch")

if start_clicked:
    try:
        start_processing(config, person_bundle, helmet_bundle)
    except RuntimeError as exc:
        st.error(str(exc))
if stop_clicked:
    st.session_state.running = False
if reset_clicked:
    reset_all(config)

seek_col, jump_col = st.columns([5, 1])
current_time = min(st.session_state.frame_index / video_info["fps"], video_info["duration"])
seek_seconds = seek_col.slider(
    f"Seek to ({format_time(current_time)} / {format_time(video_info['duration'])})",
    0.0, max(1.0, video_info["duration"]), value=float(current_time),
    step=1.0, disabled=st.session_state.running,
)
jump_clicked = jump_col.button("Jump", width="stretch", disabled=st.session_state.running)

if jump_clicked:
    try:
        seek_to(config, seek_seconds, video_info["fps"])
    except RuntimeError as exc:
        st.error(str(exc))

video_col, stats_col = st.columns([2, 1])
video_ph = video_col.empty()
time_ph = video_col.empty()
stats_col.subheader("Live Metrics")
metric_people = stats_col.empty()
metric_helmet = stats_col.empty()
metric_no_helmet = stats_col.empty()
metric_compliance = stats_col.empty()
metric_unique = stats_col.empty()
metric_fps = stats_col.empty()


def render_metrics():
    counts = st.session_state.current_counts
    compliance = st.session_state.current_compliance
    metric_people.metric("People (this frame)", counts["people"])
    metric_helmet.metric("With Helmet", counts["helmet"])
    metric_no_helmet.metric("Without Helmet", counts["no_helmet"])
    metric_compliance.metric(
        "Helmet Compliance", f"{compliance * 100:.0f}%" if compliance is not None else "N/A"
    )
    metric_unique.metric("Unique People Seen", len(st.session_state.unique_ids))
    metric_fps.metric("Processing FPS", f"{st.session_state.fps_estimate:.1f}")


def render_time():
    position = min(st.session_state.frame_index / video_info["fps"], video_info["duration"])
    time_ph.caption(f"{format_time(position)} / {format_time(video_info['duration'])} - {speed_label}")


if st.session_state.last_frame_rgb is not None:
    video_ph.image(st.session_state.last_frame_rgb, channels="RGB", width=DISPLAY_WIDTH)
else:
    video_ph.info("Click Start to begin processing.")

render_time()
render_metrics()

if st.session_state.video_ended:
    st.success(
        f"Video finished after {st.session_state.frame_index} frames - "
        f"{len(st.session_state.unique_ids)} unique people seen."
    )

if st.session_state.running:
    cap = st.session_state.cap
    resources = st.session_state.resources
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # >=1x is achieved by skipping frames (processing is already the
    # bottleneck below the source frame rate, so faster-than-realtime
    # playback has to come from covering more video per processed frame,
    # not from sleeping less); <1x instead stretches the per-frame delay.
    frame_step = max(1, round(speed_multiplier))
    target_dt = (1.0 / source_fps) * (1.0 / speed_multiplier if speed_multiplier < 1 else 1.0)

    while st.session_state.running:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            st.session_state.running = False
            st.session_state.video_ended = True
            break
        st.session_state.frame_index += 1
        if frame is None or frame.size == 0:
            continue

        try:
            result = process_frame(frame, resources, config)
        except Exception as exc:  # noqa: BLE001 - keep the UI alive, surface the error
            st.error(f"Detection failed on frame {st.session_state.frame_index}: {exc}")
            st.session_state.running = False
            break

        st.session_state.unique_ids.update(ps.track_id for ps in result.person_statuses)
        st.session_state.current_counts = result.counts
        st.session_state.current_compliance = result.compliance

        annotated = annotate_frame(frame, result.person_statuses, result.counts, result.compliance)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.session_state.last_frame_rgb = annotated_rgb
        video_ph.image(annotated_rgb, channels="RGB", width=DISPLAY_WIDTH)
        render_time()
        render_metrics()

        # Skip ahead for >1x speeds - these frames are read but never scored,
        # matching a real player's "fast forward" rather than slow-motion
        # detection on every single frame.
        for _ in range(frame_step - 1):
            skip_ret, _ = cap.read()
            if not skip_ret:
                st.session_state.running = False
                st.session_state.video_ended = True
                break
            st.session_state.frame_index += 1
        if not st.session_state.running:
            break

        now = time.time()
        st.session_state.frame_times.append(now)
        if len(st.session_state.frame_times) >= 2:
            span = st.session_state.frame_times[-1] - st.session_state.frame_times[0]
            st.session_state.fps_estimate = (
                (len(st.session_state.frame_times) - 1) / span if span > 0 else 0.0
            )

        elapsed = time.time() - loop_start
        remaining = target_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    if st.session_state.video_ended:
        st.rerun()
