from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import wandb


def get_git_sha(repo_root: Path | str = ".") -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def get_data_fingerprint(
    fingerprint_path: Path | str = "data/01_raw/iranian_churn/fingerprint.json",
    fallback_data_path: Path | str = "data/01_raw/iranian_churn/Customer Churn.csv",
) -> str:
    fp_file = Path(fingerprint_path)
    if fp_file.exists():
        try:
            payload = json.loads(fp_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("sha256"):
                return str(payload["sha256"])
        except (json.JSONDecodeError, OSError):
            pass

    data_file = Path(fallback_data_path)
    if data_file.exists():
        return hashlib.sha256(data_file.read_bytes()).hexdigest()

    return "unknown"


def get_library_versions(
    packages: tuple[str, ...] = (
        "python",
        "scikit-learn",
        "pandas",
        "numpy",
        "kedro",
        "wandb",
        "autogluon.tabular",
    ),
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in packages:
        if pkg == "python":
            versions["python"] = platform.python_version()
            continue
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def log_wandb_run(
    *,
    run_name: str,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    data_mode: str = "full",
    dataset_name: str = "iranian_churn",
    git_sha: str | None = None,
    data_fingerprint: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
) -> str | None:
    resolved_entity = entity or os.getenv("WANDB_ENTITY") or None
    resolved_project = project or os.getenv("WANDB_PROJECT") or "asi-project"
    resolved_sha = git_sha or get_git_sha()
    resolved_fp = data_fingerprint or get_data_fingerprint()

    config_payload: dict[str, Any] = {
        "model_name": model_name,
        "dataset": dataset_name,
        "data_mode": data_mode,
        "git_sha": resolved_sha,
        "data_fingerprint": resolved_fp,
        **params,
    }

    flat_metrics = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }

    run = wandb.init(
        entity=resolved_entity,
        project=resolved_project,
        name=run_name,
        config=config_payload,
        tags=tags or [model_name, data_mode],
        reinit=True,
    )
    try:
        if run is not None:
            run.log(flat_metrics)
            run.summary.update(
                {
                    **flat_metrics,
                    "git_sha": resolved_sha,
                    "data_fingerprint": resolved_fp,
                }
            )
            return str(run.id)
        return None
    finally:
        wandb.finish()