import argparse
import subprocess
import sys
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd
import yaml
from measure_model_memory import load_candidate

from asi_project.metrics import compute_metrics
from asi_project.model_package import save_model_package
from asi_project.pipelines.data_processing.nodes import (
    apply_input_normalization,
    apply_training_filter,
    validate_contract,
)
from asi_project.pipelines.data_science.nodes import split_data, train_models


def benchmark_package(model_dir, X_test, y_test):
    model, metadata = load_candidate(model_dir)
    feature_names = metadata.get("features", X_test.columns)
    features = X_test.loc[:, feature_names]

    predictions = model.predict(features)
    f1 = compute_metrics(y_test, predictions)["f1"]

    timings_ms = []
    for _ in range(30):
        start = perf_counter()
        model.predict(features)
        timings_ms.append((perf_counter() - start) * 1000)
    prediction_ms = median(timings_ms)

    size_bytes = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
    size_mib = size_bytes / 1024**2

    probe_path = Path(__file__).with_name("measure_model_memory.py")
    memory_bytes = int(
        subprocess.check_output(
            [sys.executable, str(probe_path), str(model_dir)],
            text=True,
        ).strip()
    )
    memory_mib = memory_bytes / 1024**2

    model_name = metadata["model_name"]
    print(f"{model_name} F1: {f1:.4f}")
    print(f"Prediction time: {prediction_ms:.3f} ms / {len(features)} rows")
    print(f"Saved model size: {size_bytes} bytes ({size_bytes / 1024:.2f} KiB)")
    print(f"Model loading RSS change: {memory_mib:.2f} MiB")

    return f"| {model_name} | {f1:.4f} | {prediction_ms:.3f} | {size_mib:.4f} | {memory_mib:.2f} |"


def write_report(output_path, rows, test_size):
    report = [
        "# Porównanie modeli — Sprint 3",
        "",
        f"Wspólny zbiór testowy: {test_size} rekordów. F1 dla klasy `Churn=1`.",
        "",
        "| Model | F1 | Czas predykcji [ms / zbiór testowy] | Rozmiar [MiB] | Przyrost RSS [MiB] |",
        "| --- | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Czas predykcji to mediana z 30 wywołań po rozgrzewce na całym zbiorze testowym.",
        "Rozmiar obejmuje wszystkie pliki pakietu baseline lub katalogu AutoGluon.",
        "Przyrost RSS zmierzono po wczytaniu modelu w osobnym procesie,",
        "przed pierwszą predykcją. Pomiar obejmuje biblioteki wczytane w tym kroku.",
        "",
    ]
    output_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nMarkdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dirs",
        nargs="*",
        type=Path,
        help="Directories containing saved AutoGluon models or model packages",
    )
    parser.add_argument("--output", type=Path, help="Path to the Markdown report")
    args = parser.parse_args()

    params = yaml.safe_load(Path("conf/base/parameters.yml").read_text(encoding="utf-8"))
    catalog = yaml.safe_load(Path("conf/base/catalog.yml").read_text(encoding="utf-8"))
    raw_data = pd.read_csv(catalog["raw_data"]["filepath"])

    validated_data = validate_contract(raw_data)
    normalized_data = apply_input_normalization(validated_data)
    training_data = apply_training_filter(normalized_data)

    X_train, X_test, y_train, y_test = split_data(training_data, params["split"])
    model, _ = train_models(X_train, y_train, params["model"])
    metrics = compute_metrics(y_test, model.predict(X_test))

    print(f"Train: {len(X_train)}, test: {len(X_test)}")

    with TemporaryDirectory() as temp_dir:
        package_dir = Path(temp_dir) / "baseline"
        save_model_package(
            model=model,
            package_dir=package_dir,
            model_name="logistic_regression",
            features=X_train.columns.tolist(),
            target="Churn",
            positive_class=1,
            metrics=metrics,
        )
        rows = [benchmark_package(package_dir, X_test, y_test)]

    for model_dir in args.model_dirs:
        print(f"\nModel directory: {model_dir}")
        rows.append(benchmark_package(model_dir, X_test, y_test))

    if args.output is not None:
        write_report(args.output, rows, len(X_test))


if __name__ == "__main__":
    main()
