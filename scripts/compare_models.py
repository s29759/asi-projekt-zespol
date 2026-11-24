import subprocess
import sys
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd

from asi_project.metrics import compute_metrics
from asi_project.model_package import load_model_package, save_model_package
from asi_project.pipelines.data_processing.nodes import (
    apply_input_normalization,
    apply_training_filter,
    validate_contract,
)
from asi_project.pipelines.data_science.nodes import split_data, train_models


def benchmark_package(package_dir, X_test, y_test):
    model, metadata = load_model_package(package_dir)
    features = X_test.loc[:, metadata["features"]]

    predictions = model.predict(features)
    metrics = compute_metrics(y_test, predictions)

    timings_ms = []
    for _ in range(30):
        start = perf_counter()
        model.predict(features)
        timings_ms.append((perf_counter() - start) * 1000)

    size_bytes = sum(path.stat().st_size for path in package_dir.rglob("*") if path.is_file())

    probe_path = Path(__file__).with_name("measure_model_memory.py")
    memory_bytes = int(
        subprocess.check_output(
            [sys.executable, str(probe_path), str(package_dir)],
            text=True,
        ).strip()
    )

    print(f"{metadata['model_name']} F1: {metrics['f1']:.4f}")
    print(f"Prediction time: {median(timings_ms):.3f} ms / {len(features)} rows")
    print(f"Model package size: {size_bytes} bytes ({size_bytes / 1024:.2f} KiB)")
    print(f"Model loading RSS change: {memory_bytes / 1024**2:.2f} MiB")


def main():
    raw_data = pd.read_csv("data/01_raw/iranian_churn/Customer Churn.csv")
    validated_data = validate_contract(raw_data)
    normalized_data = apply_input_normalization(validated_data)
    training_data = apply_training_filter(normalized_data)

    X_train, X_test, y_train, y_test = split_data(
        training_data, {"test_size": 0.2, "random_state": 42}
    )
    model, _ = train_models(X_train, y_train, {"max_iter": 1000, "random_state": 42})
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
        benchmark_package(package_dir, X_test, y_test)


if __name__ == "__main__":
    main()
