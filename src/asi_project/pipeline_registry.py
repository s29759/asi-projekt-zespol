"""Rejestrowanie pipeline'ów Kedro (Sprint 2).

Po przygotowaniu ``create_pipeline()`` w podpakietach
``data_processing`` oraz ``data_science`` Kedro wykrywa je automatycznie.
Ten plik NIE zawiera implementacji nodów ani gotowego pipeline'u.
"""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Wykryj pipeline'y z ``src/asi_project/pipelines/``.

    ``__default__`` (czyli samo ``kedro run``) to data_processing + data_science.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
