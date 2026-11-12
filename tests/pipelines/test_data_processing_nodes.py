"""Testy nodów pipeline'u data_processing na próbce CI (bez uruchamiania Kedro)."""

import numpy as np
import pandas as pd
import pytest

from asi_project.data import DataContractError
from asi_project.pipelines.data_processing.nodes import (
    apply_input_normalization,
    apply_training_filter,
    validate_contract,
)


@pytest.fixture
def raw_sample(project_config):
    if not project_config.ci_sample_path.is_file():
        pytest.skip("Brak próbki CI — uruchom python scripts/make_ci_sample.py.")
    return pd.read_csv(project_config.ci_sample_path)


def test_validate_contract_przepuszcza_poprawna_probke(raw_sample):
    validated = validate_contract(raw_sample)
    assert len(validated) == len(raw_sample)
    assert "Churn" in validated.columns


def test_validate_contract_odrzuca_dane_bez_targetu(raw_sample):
    with pytest.raises(DataContractError):
        validate_contract(raw_sample.drop(columns=["Churn"]))


def test_input_normalization_nie_usuwa_wierszy_i_jest_idempotentna(raw_sample):
    validated = validate_contract(raw_sample)
    once = apply_input_normalization(validated)
    twice = apply_input_normalization(once)
    assert len(once) == len(validated)
    pd.testing.assert_frame_equal(once, twice)


def test_training_filter_usuwa_duplikaty_i_wiersze_bez_targetu(raw_sample):
    normalized = apply_input_normalization(validate_contract(raw_sample))
    base = normalized.drop_duplicates()
    no_target = base.iloc[[0]].assign(Churn=np.nan)
    dirty = pd.concat([base, base.iloc[[0, 1]], no_target], ignore_index=True)

    filtered = apply_training_filter(dirty)

    assert len(filtered) == len(base)
    assert not filtered.duplicated().any()
    assert filtered["Churn"].notna().all()
