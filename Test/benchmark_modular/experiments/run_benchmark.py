from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from ..config import BenchmarkConfig
from ..data import split_features_target
from ..metrics import classification_metrics, regression_metrics
from ..models import LSTMModel, TransformerModel, make_random_forest
from ..training import fit_torch_model, make_tensor_dataloader, predict_torch


def _score(y_true, y_pred, task_type: str) -> Dict[str, float]:
    if task_type == "classification":
        return classification_metrics(y_true, y_pred)
    return regression_metrics(y_true, y_pred)


def run_random_forest_reproduction(features, target, config: BenchmarkConfig) -> Dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=config.test_size, random_state=config.random_state)
    model = make_random_forest(task_type=config.task_type, random_state=config.random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return _score(y_test, predictions, config.task_type)


def run_random_forest_grid_search(features, target, config: BenchmarkConfig) -> Dict[str, Any]:
    model = make_random_forest(task_type=config.task_type, random_state=config.random_state)
    parameter_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    }
    search = GridSearchCV(model, parameter_grid, cv=config.n_splits, n_jobs=-1)
    search.fit(features, target)
    return {"best_params": search.best_params_, "best_score": float(search.best_score_), "estimator": search.best_estimator_}


def run_tabpfn_benchmark(features, target, config: BenchmarkConfig) -> Dict[str, float]:
    try:
        if config.task_type == "classification":
            from tabpfn import TabPFNClassifier as TabPFNModel
        else:
            from tabpfn import TabPFNRegressor as TabPFNModel
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("TabPFN is not installed in this environment.") from exc

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=config.test_size, random_state=config.random_state)
    model = TabPFNModel()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return _score(y_test, predictions, config.task_type)


def run_transformer_experiment(features, target, config: BenchmarkConfig) -> Dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=config.test_size, random_state=config.random_state)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=config.random_state)
    model = TransformerModel(input_dim=x_train.shape[1])
    model, _history = fit_torch_model(model, x_train, y_train, x_valid, y_valid, batch_size=config.batch_size, epochs=config.epochs, learning_rate=config.learning_rate, device=config.device, task_type=config.task_type)
    test_loader = make_tensor_dataloader(x_test, y_test, batch_size=config.batch_size, shuffle=False)
    predictions = predict_torch(model, test_loader, config.device, config.task_type)
    return _score(y_test, predictions, config.task_type)


def run_lstm_experiment(features, target, config: BenchmarkConfig) -> Dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=config.test_size, random_state=config.random_state)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=config.random_state)
    model = LSTMModel(input_dim=x_train.shape[1])
    model, _history = fit_torch_model(model, x_train, y_train, x_valid, y_valid, batch_size=config.batch_size, epochs=config.epochs, learning_rate=config.learning_rate, device=config.device, task_type=config.task_type)
    test_loader = make_tensor_dataloader(x_test, y_test, batch_size=config.batch_size, shuffle=False)
    predictions = predict_torch(model, test_loader, config.device, config.task_type)
    return _score(y_test, predictions, config.task_type)


def run_benchmark(frame: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    features, target = split_features_target(frame, config.target_column)

    rf_score = run_random_forest_reproduction(features, target, config)
    tabpfn_score = run_tabpfn_benchmark(features, target, config)
    transformer_score = run_transformer_experiment(features, target, config)
    lstm_score = run_lstm_experiment(features, target, config)

    rows = [
        {"model": "random_forest", **rf_score},
        {"model": "tabpfn", **tabpfn_score},
        {"model": "transformer", **transformer_score},
        {"model": "lstm", **lstm_score},
    ]
    return pd.DataFrame(rows)
