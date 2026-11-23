from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asi_project.tracking import get_data_fingerprint, get_git_sha, get_library_versions

METADATA_FILENAME = "metadata.json"
MODEL_FILENAME = "model.pkl"
AUTOGLUON_DIRNAME = "autogluon_model"

REQUIRED_METADATA_KEYS = (
    "model_name",
    "model_format",
    "features",
    "target",
    "positive_class",
    "metrics",
    "git_sha",
    "data_fingerprint",
    "library_versions",
)


def _is_autogluon_predictor(model: Any) -> bool:
    return hasattr(model, "save") and type(model).__name__ == "TabularPredictor"


def save_model_package(
    *,
    model: Any,
    package_dir: Path | str,
    model_name: str,
    features: list[str],
    target: str,
    positive_class: int | str,
    metrics: dict[str, Any],
    git_sha: str | None = None,
    data_fingerprint: str | None = None,
    library_versions: dict[str, str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    target_dir = Path(package_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if _is_autogluon_predictor(model):
        model_format = "autogluon"
        model.save(str(target_dir / AUTOGLUON_DIRNAME))
    else:
        model_format = "pickle"
        with (target_dir / MODEL_FILENAME).open("wb") as f:
            pickle.dump(model, f)

    metadata: dict[str, Any] = {
        "model_name": model_name,
        "model_format": model_format,
        "features": [str(col) for col in features],
        "target": str(target),
        "positive_class": positive_class,
        "metrics": metrics,
        "git_sha": git_sha or get_git_sha(),
        "data_fingerprint": data_fingerprint or get_data_fingerprint(),
        "library_versions": library_versions or get_library_versions(),
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    (target_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target_dir


def load_model_package(package_dir: Path | str) -> tuple[Any, dict[str, Any]]:
    source_dir = Path(package_dir)
    metadata_path = source_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {METADATA_FILENAME} in {source_dir}")

    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in metadata]
    if missing:
        raise ValueError(f"Missing required metadata keys: {missing}")

    model_format = metadata["model_format"]
    if model_format == "autogluon":
        from autogluon.tabular import TabularPredictor

        ag_path = source_dir / AUTOGLUON_DIRNAME
        if not ag_path.exists():
            raise FileNotFoundError(f"Missing AutoGluon directory: {ag_path}")
        model = TabularPredictor.load(str(ag_path))
    elif model_format == "pickle":
        model_path = source_dir / MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        with model_path.open("rb") as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Unsupported model_format: {model_format!r}")

    return model, metadata
