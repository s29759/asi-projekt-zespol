from dataclasses import replace

import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import (
    build_metrics_record,
    compute_metrics,
    validate_metrics_record,
)
from asi_project.model import build_baseline_pipeline, make_dummy
from asi_project.tracking import log_wandb_run


def split_data(
    data: pd.DataFrame,
    split_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    config = load_project_config()
    contract = get_contract(config)

    X, y = make_xy(
        df=data,
        contract=contract,
        features=config.model_features,
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X=X,
        y=y,
        contract=contract,
        test_size=float(split_params["test_size"]),
        random_state=int(split_params["random_state"]),
    )

    return X_train, X_test, y_train, y_test


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
):
    config = load_project_config()

    model = build_baseline_pipeline(
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
        max_iter=int(model_params["max_iter"]),
        random_state=int(model_params["random_state"]),
    )
    model.fit(X_train, y_train)

    dummy_model = make_dummy()
    dummy_model.fit(X_train, y_train)

    return model, dummy_model


def evaluate_models(
    model,
    dummy_model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    split_params: dict,
    model_params: dict,
    data_mode: str,
    wandb_params: dict | None = None,
) -> dict:
    config = load_project_config()
    contract = get_contract(config)

    lr_metrics = compute_metrics(y_test, model.predict(X_test))
    dummy_metrics = compute_metrics(y_test, dummy_model.predict(X_test))

    results = {
        "logistic_regression": lr_metrics,
        "dummy": dummy_metrics,
    }

    metrics_config = replace(
        config,
        test_size=float(split_params["test_size"]),
        random_state=int(split_params["random_state"]),
    )

    metrics = build_metrics_record(
        contract=contract,
        config=metrics_config,
        features=config.model_features,
        n_train=len(X_train),
        n_test=len(X_test),
        data_mode=data_mode,
        results=results,
    )

    problems = validate_metrics_record(metrics, contract)
    if problems:
        raise ValueError("Invalid metrics record:\n" + "\n".join(problems))

    wb_cfg = wandb_params or {}
    if wb_cfg.get("enabled", False) and data_mode == "full":
        run_params = {
            "test_size": float(split_params["test_size"]),
            "split_random_state": int(split_params["random_state"]),
            "max_iter": int(model_params["max_iter"]),
            "model_random_state": int(model_params["random_state"]),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(config.model_features),
        }
        log_wandb_run(
            run_name="baseline-logistic-regression",
            model_name="logistic_regression_baseline",
            params=run_params,
            metrics=lr_metrics,
            data_mode=data_mode,
            dataset_name=contract.dataset_id,
            entity=wb_cfg.get("entity") or None,
            project=wb_cfg.get("project") or None,
            tags=["baseline", "logistic_regression", data_mode],
        )

    return metrics
