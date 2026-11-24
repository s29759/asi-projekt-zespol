from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_models, split_data, train_autogluon, train_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    "processed_data",
                    "params:split",
                ],
                outputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                ],
                name="split_data_node",
            ),
            node(
                func=train_models,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:model",
                ],
                outputs=[
                    "trained_model",
                    "dummy_model",
                ],
                name="train_models_node",
            ),
            node(
                func=evaluate_models,
                inputs=[
                    "trained_model",
                    "dummy_model",
                    "X_train",
                    "X_test",
                    "y_test",
                    "params:split",
                    "params:model",
                    "params:data_mode",
                    "params:wandb",
                ],
                outputs="metrics",
                name="evaluate_models_node",
            ),
            node(
                func=train_autogluon,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:autogluon",
                    "params:data_mode",
                    "params:wandb",
                ],
                outputs="autogluon_metrics",
                name="train_autogluon_node",
            ),
        ]
    )
