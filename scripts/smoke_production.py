"""Smoke test modelu production (S3, krok 9), w osobnym procesie, bez treningu.

Pobiera artefakt `churn-model:production` z W&B, wczytuje go funkcją
`load_model_package` i robi predykcję dla 5 wierszy próbki CI.

    python scripts/smoke_production.py                                # z W&B
    python scripts/smoke_production.py --package-dir data/06_models/production
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from asi_project.data import normalize_column_names, normalize_input
from asi_project.model_package import load_model_package

ARTIFACT_NAME = "churn-model"
ALIAS = "production"
SAMPLE_PATH = Path("data/01_raw/sample.csv")
N_ROWS = 5


def download_production(target_dir: Path) -> Path:
    import wandb

    wb = yaml.safe_load(Path("conf/base/parameters.yml").read_text(encoding="utf-8"))["wandb"]
    ref = f"{wb['entity']}/{wb['project']}/{ARTIFACT_NAME}:{ALIAS}"
    artifact = wandb.Api().artifact(ref, type="model")
    print(f"Pobrano {ref} (wersja {artifact.version})")
    return Path(artifact.download(root=str(target_dir)))


def positive_class_proba(model, X: pd.DataFrame, positive_class) -> pd.Series:
    """Prawdopodobieństwo klasy pozytywnej dla AutoGluona (DataFrame) i sklearn (ndarray)."""
    proba = model.predict_proba(X)
    if isinstance(proba, pd.DataFrame):
        columns = list(proba.columns)
    else:
        columns = list(model.classes_)
        proba = pd.DataFrame(proba, index=X.index, columns=columns)
    matches = [c for c in columns if str(c) == str(positive_class)]
    if not matches:
        raise RuntimeError(f"Brak klasy pozytywnej {positive_class!r} w {columns}")
    return proba[matches[0]]


def pick_rows(sample: pd.DataFrame, target: str, positive_class, n_rows: int) -> pd.DataFrame:
    """Kilka wierszy z obu klas (jeśli próbka ma target), żeby było widać obie odpowiedzi."""
    if target not in sample:
        return sample.head(n_rows)
    positive = sample[sample[target].astype(str) == str(positive_class)].head(n_rows // 2)
    negative = sample.drop(positive.index).head(n_rows - len(positive))
    return pd.concat([negative, positive])


def run_smoke(
    package_dir: Path, sample_path: Path = SAMPLE_PATH, n_rows: int = N_ROWS
) -> pd.DataFrame:
    model, metadata = load_model_package(package_dir)
    features = metadata["features"]

    sample = normalize_input(normalize_column_names(pd.read_csv(sample_path)))
    rows = pick_rows(sample, metadata["target"], metadata["positive_class"], n_rows)
    X = rows[features]
    predictions = pd.Series(model.predict(X), index=X.index)
    churn_proba = positive_class_proba(model, X, metadata["positive_class"])

    result = pd.DataFrame({"prediction": predictions, "churn_probability": churn_proba.round(4)})
    if metadata["target"] in rows:
        result.insert(0, "true_label", rows[metadata["target"]])
    if len(result) != n_rows or not result["churn_probability"].between(0, 1).all():
        raise RuntimeError("Smoke test: zła liczba predykcji albo prawdopodobieństwa spoza [0, 1]")
    print(
        f"Model: {metadata['model_name']} ({metadata['model_format']}), F1 test: {metadata['metrics']['f1']:.4f}"
    )
    print(result.to_string())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, help="lokalny pakiet zamiast pobierania z W&B")
    args = parser.parse_args()

    if args.package_dir:
        run_smoke(args.package_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            run_smoke(download_production(Path(tmp)))
    print("Smoke test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
