"""Zapisuje wybrany model jako pakiet i publikuje go w W&B z aliasem `production` (S3, krok 8).

Model wybrany w ADR-2: AutoGluon `medium_60s`, trenowany przez `kedro run`
(katalog z conf/base/parameters.yml -> autogluon.path / <nazwa konfiguracji>).

    python scripts/publish_production.py                  # pakiet + W&B Artifact
    python scripts/publish_production.py --no-wandb       # tylko pakiet lokalnie
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from asi_project.config import get_contract, load_project_config
from asi_project.metrics import compute_metrics
from asi_project.model_package import load_model_package, save_model_package
from asi_project.pipelines.data_processing.nodes import (
    apply_input_normalization,
    apply_training_filter,
    validate_contract,
)
from asi_project.pipelines.data_science.nodes import split_data

ARTIFACT_NAME = "churn-model"
PARAMS_PATH = Path("conf/base/parameters.yml")
CATALOG_PATH = Path("conf/base/catalog.yml")


def load_test_split(params: dict) -> tuple[pd.DataFrame, pd.Series, int]:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    raw = pd.read_csv(catalog["raw_data"]["filepath"])
    data = apply_training_filter(apply_input_normalization(validate_contract(raw)))
    X_train, X_test, _, y_test = split_data(data, params["split"])
    return X_test, y_test, len(X_train)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="medium_60s", help="konfiguracja AutoGluon z ADR-2")
    parser.add_argument("--output", type=Path, default=Path("data/06_models/production"))
    parser.add_argument("--no-wandb", action="store_true", help="bez publikacji w W&B")
    args = parser.parse_args()

    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    model_dir = Path(params["autogluon"]["path"]) / args.config
    if not (model_dir / "predictor.pkl").is_file():
        raise SystemExit(f"Brak modelu {model_dir}. Najpierw uruchom: kedro run")

    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(str(model_dir))
    config = load_project_config()
    contract = get_contract(config)
    features = list(config.model_features)

    X_test, y_test, n_train = load_test_split(params)
    metrics = compute_metrics(y_test, predictor.predict(X_test[features]))
    print(f"{args.config}: F1 na zbiorze testowym = {metrics['f1']:.4f}")

    package_dir = save_model_package(
        model=predictor,
        package_dir=args.output,
        model_name=f"autogluon_{args.config}",
        features=features,
        target=contract.target,
        positive_class=contract.positive_class,
        metrics=metrics,
        extra_metadata={
            "selected_in": "docs/adr/0002-wybor-modelu-produkcyjnego.md",
            "presets": params["autogluon"]["configurations"][args.config]["presets"],
            "time_limit": params["autogluon"]["configurations"][args.config]["time_limit"],
            "best_model": predictor.model_best,
            "data_mode": "full",
            "n_train": n_train,
            "n_test": len(X_test),
        },
    )

    # Kontrola przed publikacją: pakiet wczytuje się i daje ten sam wynik.
    loaded, metadata = load_model_package(package_dir)
    reloaded_f1 = compute_metrics(y_test, loaded.predict(X_test[features]))["f1"]
    if round(reloaded_f1, 6) != round(metrics["f1"], 6):
        raise SystemExit(f"Pakiet daje inne F1 ({reloaded_f1:.4f}) niż model ({metrics['f1']:.4f})")
    print(f"Pakiet zapisany w {package_dir}, odczyt OK (F1 {reloaded_f1:.4f})")

    if args.no_wandb:
        return

    import wandb

    wb = params["wandb"]
    run = wandb.init(
        entity=wb.get("entity") or None,
        project=wb.get("project") or None,
        job_type="publish-production",
        name=f"production-{args.config}",
        config={"model_name": metadata["model_name"], "git_sha": metadata["git_sha"]},
    )
    artifact = wandb.Artifact(
        ARTIFACT_NAME,
        type="model",
        description=f"Model produkcyjny (ADR-2): AutoGluon {args.config}",
        metadata={
            k: metadata[k]
            for k in ("model_name", "model_format", "metrics", "git_sha", "data_fingerprint")
        },
    )
    artifact.add_dir(str(package_dir))
    run.log_artifact(artifact, aliases=["production"])
    artifact.wait()
    print(
        f"W&B Artifact: {run.entity}/{run.project}/{ARTIFACT_NAME}:{artifact.version} (alias production)"
    )
    run.finish()


if __name__ == "__main__":
    main()
