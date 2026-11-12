"""Testy nodów pipeline'u data_science na próbce CI (bez uruchamiania Kedro)."""

import pandas as pd
import pytest

from asi_project.config import load_project_config
from asi_project.metrics import validate_metrics_record
from asi_project.pipelines.data_processing.nodes import (
    apply_input_normalization,
    apply_training_filter,
    validate_contract,
)
from asi_project.pipelines.data_science.nodes import evaluate_models, split_data, train_models

SPLIT_PARAMS = {"test_size": 0.2, "random_state": 42}


@pytest.fixture(scope="module")
def processed_sample():
    config = load_project_config()
    if not config.ci_sample_path.is_file():
        pytest.skip("Brak próbki CI — uruchom python scripts/make_ci_sample.py.")
    raw = pd.read_csv(config.ci_sample_path)
    return apply_training_filter(apply_input_normalization(validate_contract(raw)))


@pytest.fixture(scope="module")
def split(processed_sample):
    return split_data(processed_sample, SPLIT_PARAMS)


def test_split_ma_poprawna_strukture(processed_sample, split, project_config):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) + len(X_test) == len(processed_sample)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert len(X_test) == pytest.approx(0.2 * len(processed_sample), abs=1)
    assert list(X_train.columns) == list(project_config.model_features)


def test_split_bez_targetu_i_cech_excluded(split):
    X_train, X_test, _, _ = split
    for X in (X_train, X_test):
        assert "Churn" not in X.columns
        assert "Customer Value" not in X.columns


def test_split_jest_stratyfikowany(split):
    _, _, y_train, y_test = split
    assert y_test.mean() == pytest.approx(y_train.mean(), abs=0.05)


def test_split_jest_powtarzalny(processed_sample, split):
    X_train_again, *_ = split_data(processed_sample, SPLIT_PARAMS)
    pd.testing.assert_frame_equal(split[0], X_train_again)


def test_train_models_zwraca_wytrenowane_modele(split):
    X_train, X_test, y_train, _ = split
    model, dummy_model = train_models(X_train, y_train)
    for fitted in (model, dummy_model):
        predictions = fitted.predict(X_test)
        assert len(predictions) == len(X_test)


def test_evaluate_models_zwraca_poprawny_rekord_metryk(split, registry, project_config):
    X_train, X_test, y_train, y_test = split
    model, dummy_model = train_models(X_train, y_train)

    metrics = evaluate_models(model, dummy_model, X_train, X_test, y_test, SPLIT_PARAMS, "sample")

    contract = registry[project_config.dataset_id]
    assert validate_metrics_record(metrics, contract) == []
    assert metrics["data_mode"] == "sample"
    assert metrics["split"]["n_train"] == len(X_train)
    assert metrics["split"]["n_test"] == len(X_test)
    assert set(metrics["models"]) == {"logistic_regression", "dummy"}
