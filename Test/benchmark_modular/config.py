from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    target_column: str
    task_type: str = "classification"
    random_state: int = 42
    test_size: float = 0.2
    n_splits: int = 5
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    device: str = "cpu"
