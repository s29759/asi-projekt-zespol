from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd

from asi_project.metrics import compute_metrics
from asi_project.model_package import save_model_package
from asi_project.pipelines.data_processing.nodes import (
    apply_input_normalization,
    apply_training_filter,
    validate_contract,
)
from asi_project.pipelines.data_science.nodes import split_data, train_models


def main():
    raw_data = pd.read_csv("data/01_raw/iranian_churn/Customer Churn.csv")
    validated_data = validate_contract(raw_data)
    normalized_data = apply_input_normalization(validated_data)
    training_data = apply_training_filter(normalized_data)

    X_train, X_test, y_train, y_test = split_data(
        training_data, {"test_size": 0.2, "random_state": 42}
    )
    model, _ = train_models(X_train, y_train, {"max_iter": 1000, "random_state": 42})

    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)

    print(f"Train: {len(X_train)}, test: {len(X_test)}")
    print(f"LogisticRegression F1: {metrics['f1']:.4f}")

    timings_ms = []
    for _ in range(30):
        start = perf_counter()
        model.predict(X_test)
        timings_ms.append((perf_counter() - start) * 1000)

    print(f"Prediction time: {median(timings_ms):.3f} ms / {len(X_test)} rows")

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
        size_bytes = sum(path.stat().st_size for path in package_dir.rglob("*") if path.is_file())

    print(f"Model package size: {size_bytes} bytes ({size_bytes / 1024:.2f} KiB)")


if __name__ == "__main__":
    main()
