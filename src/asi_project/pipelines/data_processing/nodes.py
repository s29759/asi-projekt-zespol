import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.data import (
    assert_valid,
    filter_training_rows,
    normalize_column_names,
    normalize_input,
)


def validate_contract(raw_data: pd.DataFrame) -> pd.DataFrame:
    config = load_project_config()
    contract = get_contract(config)
    df = normalize_column_names(raw_data)
    assert_valid(df, contract, config)
    return df


def apply_input_normalization(validated_data: pd.DataFrame) -> pd.DataFrame:
    return normalize_input(validated_data)


def apply_training_filter(normalized_data: pd.DataFrame) -> pd.DataFrame:
    contract = get_contract(load_project_config())
    return filter_training_rows(normalized_data, contract)
