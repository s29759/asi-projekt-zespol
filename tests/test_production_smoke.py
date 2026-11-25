"""Smoke test pakietu production bez W&B i bez treningu AutoGluona (działa w CI)."""

import importlib.util
from pathlib import Path

import pytest

from asi_project.model_package import AUTOGLUON_DIRNAME, METADATA_FILENAME, save_model_package
from asi_project.pipelines.data_processing.nodes import (
    apply_input_normalization,
    apply_training_filter,
    validate_contract,
)
from asi_project.pipelines.data_science.nodes import split_data, train_models

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "smoke_production", REPO_ROOT / "scripts" / "smoke_production.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def baseline_package(tmp_path, project_config):
    import pandas as pd

    if not project_config.ci_sample_path.is_file():
        pytest.skip("Brak próbki CI.")
    raw = pd.read_csv(project_config.ci_sample_path)
    data = apply_training_filter(apply_input_normalization(validate_contract(raw)))
    X_train, _, y_train, _ = split_data(data, {"test_size": 0.2, "random_state": 42})
    model, _ = train_models(X_train, y_train, {"max_iter": 1000, "random_state": 42})
    return save_model_package(
        model=model,
        package_dir=tmp_path / "production",
        model_name="logistic_regression",
        features=list(project_config.model_features),
        target="Churn",
        positive_class="1",
        metrics={"f1": 0.5},
        git_sha="test",
        data_fingerprint="test",
    )


def test_smoke_daje_5_predykcji_z_obu_klas(baseline_package, project_config):
    smoke = _load_smoke_module()
    result = smoke.run_smoke(baseline_package, project_config.ci_sample_path)

    assert len(result) == 5
    assert result["churn_probability"].between(0, 1).all()
    assert set(result["true_label"]) == {0, 1}


def test_pakiet_autogluon_trafia_do_katalogu_pakietu(tmp_path):
    class TabularPredictor:  # atrapa: ta sama nazwa i metody co w AutoGluonie
        def save(self, silent=False):
            raise AssertionError("save() zapisuje poza pakietem")

        def clone_for_deployment(self, path, dirs_exist_ok=False):
            Path(path).mkdir(parents=True, exist_ok=dirs_exist_ok)
            (Path(path) / "predictor.pkl").write_bytes(b"x")

    package_dir = save_model_package(
        model=TabularPredictor(),
        package_dir=tmp_path / "pkg",
        model_name="autogluon_medium_60s",
        features=["a"],
        target="Churn",
        positive_class="1",
        metrics={"f1": 0.9},
        git_sha="test",
        data_fingerprint="test",
    )

    assert (package_dir / AUTOGLUON_DIRNAME / "predictor.pkl").is_file()
    assert (package_dir / METADATA_FILENAME).is_file()
