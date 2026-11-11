from kedro.pipeline import Node, Pipeline

from .nodes import apply_input_normalization, apply_training_filter, validate_contract


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=validate_contract,
                inputs="raw_data",
                outputs="validated_data",
                name="validate_contract",
            ),
            Node(
                func=apply_input_normalization,
                inputs="validated_data",
                outputs="normalized_data",
                name="input_normalization",
            ),
            Node(
                func=apply_training_filter,
                inputs="normalized_data",
                outputs="processed_data",
                name="training_only_filtering",
            ),
        ]
    )
