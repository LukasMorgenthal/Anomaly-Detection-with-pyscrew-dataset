from __future__ import annotations

import pandas as pd

from benchmark_modular import BenchmarkConfig, run_benchmark


def main(frame: pd.DataFrame, target_column: str, task_type: str = "regression"):
    config = BenchmarkConfig(target_column=target_column, task_type=task_type)
    return run_benchmark(frame, config)


if __name__ == "__main__":
    raise SystemExit("Import main(frame, target_column) from the notebook or another script.")
