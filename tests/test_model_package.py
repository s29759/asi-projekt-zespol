import json

import pandas as pd

from asi_project.model import build_baseline_pipeline
from asi_project.model_package import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    REQUIRED_METADATA_KEYS,
    load_model_package,
    save_model_package,
)


def test_save_and_load_model_package_roundtrip(tmp_path):
    X = pd.DataFrame(
        {
            "num_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "num_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])

    model = build_baseline_pipeline(
        numeric_features=["num_a", "num_b"],
        categorical_features=[],
        max_iter=100,
        random_state=42,
    )
    model.fit(X, y)

    package_dir = tmp_path / "model_pkg"
    save_model_package(
        model=model,
        package_dir=package_dir,
        model_name="test_baseline",
        features=["num_a", "num_b"],
        target="Churn",
        positive_class=1,
        metrics={"f1": 0.85, "accuracy": 0.83},
        git_sha="abc1234",
        data_fingerprint="deadbeef",
    )

    assert (package_dir / MODEL_FILENAME).exists()
    assert (package_dir / METADATA_FILENAME).exists()

    raw_meta = json.loads((package_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    for key in REQUIRED_METADATA_KEYS:
        assert key in raw_meta

    loaded_model, loaded_meta = load_model_package(package_dir)
    assert loaded_meta["model_name"] == "test_baseline"
    assert loaded_meta["features"] == ["num_a", "num_b"]
    assert loaded_meta["target"] == "Churn"
    assert loaded_meta["positive_class"] == 1
    assert loaded_meta["git_sha"] == "abc1234"
    assert loaded_meta["data_fingerprint"] == "deadbeef"

    preds = loaded_model.predict(X.iloc[:5])
    assert len(preds) == 5