import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.data import make_xy, split_train_test


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
