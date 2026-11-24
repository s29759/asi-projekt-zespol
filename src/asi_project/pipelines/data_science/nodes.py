from dataclasses import replace
from pathlib import Path

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


def train_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    autogluon_params: dict,
    data_mode: str,
    wandb_params: dict | None = None,
) -> dict:

    if not autogluon_params.get("enabled", False):
        return {}

    from autogluon.tabular import TabularPredictor

    config = load_project_config()
    contract = get_contract(config)
    label = contract.target
    train_data = X_train.copy()
    train_data[label] = y_train.to_numpy()

    save_path = Path(autogluon_params["path"])
    eval_metric = autogluon_params.get("eval_metric", "f1")
    wb_cfg = wandb_params or {}

    results = {}
    for name, cfg in autogluon_params["configurations"].items():
        presets = str(cfg["presets"])
        time_limit = int(cfg["time_limit"])

        predictor = TabularPredictor(
            label=label,
            problem_type="binary",
            eval_metric=eval_metric,
            path=str(save_path / name),
        )
        predictor.fit(
            train_data,
            presets=presets,
            time_limit=time_limit,
        )

        metrics = compute_metrics(y_test, predictor.predict(X_test))
        results[name] = metrics

        if wb_cfg.get("enabled", False) and data_mode == "full":
            run_params = {
                "presets": presets,
                "time_limit": time_limit,
                "eval_metric": eval_metric,
                "best_model": predictor.model_best,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": len(config.model_features),
            }
            log_wandb_run(
                run_name=f"autogluon-{name}",
                model_name=f"autogluon_{name}",
                params=run_params,
                metrics=metrics,
                data_mode=data_mode,
                dataset_name=contract.dataset_id,
                entity=wb_cfg.get("entity") or None,
                project=wb_cfg.get("project") or None,
                tags=["autogluon", name, data_mode],
            )

    return results
