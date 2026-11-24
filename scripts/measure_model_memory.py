import sys

import psutil

from asi_project.model_package import load_model_package


def main():
    process = psutil.Process()
    before = process.memory_info().rss
    model, _ = load_model_package(sys.argv[1])
    after = process.memory_info().rss
    print(after - before)
    del model


if __name__ == "__main__":
    main()
