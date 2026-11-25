import sys
from contextlib import redirect_stdout
from pathlib import Path

import psutil

from asi_project.model_package import load_model_package


def load_candidate(model_dir):
    if (model_dir / "predictor.pkl").is_file():
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(model_dir))
        return model, {"model_name": model_dir.name}

    return load_model_package(model_dir)


def main():
    model_dir = Path(sys.argv[1])
    process = psutil.Process()

    with redirect_stdout(sys.stderr):
        before = process.memory_info().rss
        loaded_candidate = load_candidate(model_dir)
        after = process.memory_info().rss

    print(after - before)
    del loaded_candidate


if __name__ == "__main__":
    main()
