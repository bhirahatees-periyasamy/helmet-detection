"""Model loading and class-mapping validation.

Loaded once (via st.cache_resource in app.py) and never re-created inside the
per-frame loop. Class indices are never hardcoded — every lookup resolves the
class name from the loaded model's own `model.names` mapping.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("helmet_detection")


class ModelLoadError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    model: object
    names: dict
    device: str


def _autoselect_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_yolo_model(weights_path: str) -> LoadedModel:
    import os

    from ultralytics import YOLO

    if not os.path.isfile(weights_path):
        raise ModelLoadError(f"Model weights not found: {weights_path}")

    try:
        model = YOLO(weights_path)
    except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
        raise ModelLoadError(f"Failed to load model '{weights_path}': {exc}") from exc

    device = _autoselect_device()
    names = model.names
    logger.info("Loaded model %s | device=%s | classes=%s", weights_path, device, names)
    return LoadedModel(model=model, names=names, device=device)


def find_class_id(names: dict, token: str) -> int | None:
    """Find the class id whose name contains `token` (case-insensitive)."""
    token = token.lower()
    for class_id, name in names.items():
        if token in str(name).lower():
            return class_id
    return None


def validate_person_model(names: dict) -> int:
    class_id = find_class_id(names, "person")
    if class_id is None:
        raise ModelLoadError(
            f"Person model classes {names} do not contain a 'person' class."
        )
    return class_id


def validate_helmet_model(names: dict) -> tuple[int, int]:
    """Accepts any model whose classes distinguish helmet vs. no-helmet under
    common naming conventions - e.g. 'Hardhat'/'NO-Hardhat' or
    'With helmet'/'Without helmet' - without hardcoding either model's exact
    labels.
    """
    helmet_id, no_helmet_id = None, None
    for class_id, name in names.items():
        lname = str(name).strip().lower()
        if "helmet" not in lname and "hardhat" not in lname:
            continue
        if "no" in lname or "without" in lname:
            no_helmet_id = class_id
        else:
            helmet_id = class_id

    if helmet_id is None or no_helmet_id is None:
        raise ModelLoadError(
            "Helmet model classes "
            f"{names} do not contain both a helmet and a no-helmet class. "
            "This app requires a model whose classes explicitly distinguish "
            "helmet vs. no-helmet (e.g. 'Hardhat'/'NO-Hardhat' or "
            "'With helmet'/'Without helmet')."
        )
    return helmet_id, no_helmet_id
