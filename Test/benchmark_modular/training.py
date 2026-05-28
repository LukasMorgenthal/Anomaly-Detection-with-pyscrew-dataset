from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
try:
    from tabpfn import TabPFNClassifier
except Exception:
    TabPFNClassifier = None


def make_tensor_dataloader(x_array, y_array, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create DataLoader from numpy-like arrays.

    - Features are converted to float32 tensors.
    - Integer targets -> `torch.long` 1D (for CrossEntropy).
    - Float targets -> `torch.float32` (kept as 1D or 2D depending on input) useful for BCE/regression.
    - Non-numeric targets raise a clear TypeError (label-encode in the notebook or use NCV wrappers which handle encoding).
    """
    features = torch.as_tensor(np.asarray(x_array), dtype=torch.float32)
    y_np = np.asarray(y_array)

    # Integer labels -> long 1D (CrossEntropy expects class indices)
    if np.issubdtype(y_np.dtype, np.integer):
        targets = torch.as_tensor(y_np, dtype=torch.long).view(-1)
    # Floating point targets -> float tensor (regression or single-logit BCE)
    elif np.issubdtype(y_np.dtype, np.floating):
        targets = torch.as_tensor(y_np.astype(np.float32), dtype=torch.float32)
        if targets.ndim == 1:
            targets = targets.view(-1, 1)
    else:
        # Give a clear error rather than silently converting strings
        raise TypeError("y_array contains non-numeric dtype; please label-encode string labels before calling this function.")

    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _prepare_sequence_batch(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2:
        return features.unsqueeze(1)
    return features


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str, task_type: str) -> float:
    model.train()
    losses: List[float] = []
    for features, targets in loader:
        features = _prepare_sequence_batch(features.to(device))
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        if task_type == "classification" and outputs.ndim > 1 and outputs.size(-1) > 1:
            targets = targets.long().view(-1)
        loss = criterion(outputs, targets if task_type != "classification" or outputs.size(-1) == 1 else targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_torch(model: nn.Module, loader: DataLoader, device: str, task_type: str):
    model.eval()
    predictions = []
    for features, _targets in loader:
        features = _prepare_sequence_batch(features.to(device))
        outputs = model(features)
        if task_type == "classification" and outputs.ndim > 1 and outputs.size(-1) > 1:
            batch_predictions = torch.argmax(outputs, dim=-1)
        elif task_type == "classification":
            batch_predictions = (torch.sigmoid(outputs) > 0.5).long().view(-1)
        else:
            batch_predictions = outputs.view(-1)
        predictions.append(batch_predictions.detach().cpu().numpy())
    if not predictions:
        return np.array([])
    return np.concatenate(predictions)


@torch.no_grad()
def validate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str, task_type: str) -> float:
    """Calculate validation loss for one epoch."""
    model.eval()
    losses: List[float] = []
    for features, targets in loader:
        features = _prepare_sequence_batch(features.to(device))
        targets = targets.to(device)
        outputs = model(features)
        if task_type == "classification" and outputs.ndim > 1 and outputs.size(-1) > 1:
            targets = targets.long().view(-1)
        loss = criterion(outputs, targets if task_type != "classification" or outputs.size(-1) == 1 else targets)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def fit_torch_model(model: nn.Module, x_train, y_train, x_valid, y_valid, batch_size: int, epochs: int, learning_rate: float, device: str, task_type: str, patience: int = None, weight_decay: float = 0.01, early_stopping_metric: str = "val_loss"):
    """Train a PyTorch model with automatic loss selection based on model output shape.

    For classification:
    - If model outputs logits with dim >1 on last axis -> `CrossEntropyLoss` (multi-class).
    - Else -> `BCEWithLogitsLoss` (binary single-logit).
    For regression use `MSELoss`.
    
    Args:
        patience: Early stopping patience (None = no early stopping)
        weight_decay: L2 regularization weight for AdamW, default 0.01
        early_stopping_metric: "val_loss" (default) or "val_f1_macro"
    
    Returns both train and validation loss per epoch in history.
    """
    model = model.to(device)

    # Probe model output shape with a single sample to decide loss
    model.eval()
    with torch.no_grad():
        sample_x = np.asarray(x_train[:1])
        sample = torch.as_tensor(sample_x, dtype=torch.float32)
        if sample.ndim == 2:
            sample = sample.unsqueeze(0)
        sample = _prepare_sequence_batch(sample.to(device))
        sample_out = model(sample)

    # Choose criterion
    if task_type == "classification":
        # Project-wide assumption: all problems are classification -> use CrossEntropy
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    train_loader = make_tensor_dataloader(x_train, y_train, batch_size=batch_size, shuffle=True)
    valid_loader = make_tensor_dataloader(x_valid, y_valid, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)

    history = []
    best_metric = None
    best_state = None
    epochs_without_impr = 0

    if early_stopping_metric not in ("val_loss", "val_f1_macro"):
        raise ValueError("early_stopping_metric must be either 'val_loss' or 'val_f1_macro'")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, task_type)
        val_loss = validate_epoch(model, valid_loader, criterion, device, task_type)
        valid_predictions = predict_torch(model, valid_loader, device, task_type)

        val_f1_macro = float("nan")
        if task_type == "classification":
            try:
                y_valid_np = np.asarray(y_valid).reshape(-1)
                valid_predictions_np = np.asarray(valid_predictions).reshape(-1)
                val_f1_macro = float(f1_score(y_valid_np, valid_predictions_np, average='macro'))
            except Exception:
                val_f1_macro = float("nan")

        history.append({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1_macro": val_f1_macro,
            "valid_predictions": valid_predictions,
        })

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Early stopping on selected metric
        current_metric = val_loss if early_stopping_metric == "val_loss" else val_f1_macro
        if np.isnan(current_metric):
            current_metric = val_loss

        is_improved = (
            best_metric is None
            or (current_metric < best_metric if early_stopping_metric == "val_loss" else current_metric > best_metric)
        )

        if is_improved:
            best_metric = current_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_impr = 0
        else:
            epochs_without_impr += 1
            if patience is not None and epochs_without_impr >= patience:
                # restore best weights and stop
                if best_state is not None:
                    model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in best_state.items()})
                break

    # ensure best weights are loaded
    if best_state is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in best_state.items()})
    return model, history


class TrainingTracker:
    """
    Trackt Training und Fold-Ergebnisse für PyTorch-Modelle.
    Sammelt Hyperparameter, Trainingsverlauf und Test-Metriken in eine einheitliche Struktur.
    """
    def __init__(self, model_name: str, hyperparams: dict):
        self.model_name = model_name
        self.hyperparams = hyperparams
        self.folds = []  # Fold-Ergebnisse für evaluate_folds()
        self.histories = []  # Trainings-Historien (Loss pro Epoch)
        self.best_params_per_fold = []
    
    def add_fold(self, fold_num: int, history: List[dict], true_labels: np.ndarray, pred_labels: np.ndarray, usage: np.ndarray, best_params: dict = None):
        """Füge ein trainiertes Fold mit seinen Metriken hinzu."""
        self.histories.append(history)
        self.best_params_per_fold.append(best_params)
        self.folds.append({
            'fold': fold_num,
            'True_label': true_labels,
            'Pred_label': pred_labels,
            'workpiece_usage': usage,
            'best_params': best_params,
        })
    
    def get_fold_results(self):
        """Gib Fold-Ergebnisse für evaluate_folds() zurück."""
        return self.folds

    def get_best_params(self):
        """Gib die pro Fold gewählten Hyperparameter zurück."""
        return self.best_params_per_fold
    
    def get_summary(self) -> pd.DataFrame:
        """Zeige Hyperparameter + Trainings-Statistiken."""
        summary_data = {
            'Model': self.model_name,
            **self.hyperparams,
            'Folds': len(self.folds),
            'Avg_Final_Train_Loss': np.mean([h[-1]['train_loss'] if h else np.nan for h in self.histories]),
        }
        return pd.DataFrame([summary_data])
    
    def plot_training_history(self, figsize: tuple = (12, 4)):
        """Visualisiere Train- und Validierungsverlauf pro Fold.
        
        - Train Loss: blaue Linie
        - Val Loss: orange Linie
        - Wenn Val Loss wieder steigt → Overfitting
        - Wenn Val Loss = Train Loss → gute Generalisierung
        - Wenn Val Loss >> Train Loss → Underfitting
        """
        n_folds = len(self.histories)
        fig, axes = plt.subplots(1, n_folds, figsize=figsize)
        
        if n_folds == 1:
            axes = [axes]
        
        for fold_num, history in enumerate(self.histories):
            train_losses = [h['train_loss'] for h in history]
            val_losses = [h.get('val_loss', np.nan) for h in history]
            
            epochs = range(1, len(train_losses) + 1)
            axes[fold_num].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3, linewidth=2)
            axes[fold_num].plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3, linewidth=2, linestyle='--')
            axes[fold_num].set_title(f"Fold {fold_num + 1}")
            axes[fold_num].set_xlabel('Epoch')
            axes[fold_num].set_ylabel('Loss')
            axes[fold_num].legend()
            axes[fold_num].grid(True, alpha=0.3)
        
        fig.suptitle(f"{self.model_name} - Training History (Blue=Train, Orange=Validation)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def __repr__(self):
        return f"TrainingTracker({self.model_name}, folds={len(self.folds)})"


def train_rf_ncv(X, y, usage, n_splits: int = 5, n_estimators: int = 400, random_state: int = 42):
    """
    Train RandomForest mit Nested Cross Validation (StratifiedKFold).
    
    Args:
        X: Feature matrix (numpy array)
        y: Target labels (numpy array)
        usage: Usage column (numpy array)
        n_splits: Anzahl der Folds
        n_estimators: RandomForest Estimators
        random_state: Random seed
        
    Returns:
        List von Dicts mit Keys: 'fold', 'True_label', 'Pred_label', 'workpiece_usage'
    """
    all_folds_results = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_num = 0
    
    for train_idx, test_idx in skf.split(X, y):
        fold_num += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        usage_test = usage[test_idx]
        
        # Train RF
        rf = RandomForestClassifier(
            random_state=random_state, 
            n_estimators=n_estimators, 
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Speichere Ergebnisse im richtigen Format für pred_filter
        all_folds_results.append({
            'fold': fold_num,
            'True_label': y_test,
            'Pred_label': y_pred,
            'workpiece_usage': usage_test
        })
        
    return all_folds_results


def train_transformer_ncv(X, y, usage, model_class, n_splits: int = 5, epochs: int = 50, 
                          batch_size: int = 32, learning_rate: float = 0.001, 
                          hidden_dim: int = 64, inner_folds: int = 3, param_list: list = None,
                          patience: int = 4, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1,
                          device: str = "cpu", random_state: int = 42, weight_decay: float = 0.01,
                          early_stopping_metric: str = "val_loss"):
    """
    Train Transformer mit Nested Cross Validation (StratifiedKFold).
    
    Args:
        X: Feature matrix (numpy array)
        y: Target labels (numpy array)
        usage: Usage column (numpy array)
        model_class: Transformer Model Class (z.B. TransformerModel aus models.py)
        n_splits: Anzahl der Folds
        epochs: Trainings-Epochen pro Fold
        batch_size: Batch size für DataLoader
        learning_rate: Learning rate
        hidden_dim: Hidden dimension
        device: "cpu" oder "cuda"
        random_state: Random seed
        
    Returns:
        TrainingTracker mit Fold-Ergebnissen + Trainings-Historie
    """
    tracker = TrainingTracker(
        "Transformer_NCV",
        {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_dim': hidden_dim,
            'inner_folds': inner_folds,
            'param_list': param_list,
            'patience': patience,
            'nhead': nhead,
            'num_layers': num_layers,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'early_stopping_metric': early_stopping_metric,
        }
    )
    
    # If y contains string/object labels, encode them to integers for training
    encoder = None
    y_np = np.asarray(y)
    if y_np.dtype.kind in ("U", "S", "O"):
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_np)
    else:
        y_encoded = y_np

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_num = 0

    for train_idx, test_idx in skf.split(X, y_encoded):
        fold_num += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        usage_test = usage[test_idx]
        # keep original true labels for storage (inverse transform if encoder present)
        true_labels_original = np.asarray(y)[test_idx] if encoder is not None else y_test

        # Standardisiere Daten pro Fold (fit auf Train, transform auf Test - kein Data Leakage)
        scaler = StandardScaler()
        X_train_shape = X_train.shape
        X_test_shape = X_test.shape
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_train = scaler.fit_transform(X_train_flat).reshape(X_train_shape)
        X_test = scaler.transform(X_test_flat).reshape(X_test_shape)

        # Initial parameter list default
        if param_list is None:
            param_list = [
                {
                    'd_model': hidden_dim,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'lr': learning_rate,
                    'batch_size': batch_size,
                    'weight_decay': weight_decay,
                }
            ]

        # Inner CV to select best hyperparameters
        best_params = None
        if param_list and len(param_list) > 1:
            inner_skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
            best_score = -np.inf
            for p in param_list:
                inner_scores = []
                for in_tr_idx, in_va_idx in inner_skf.split(X_train, y_train):
                    X_in_tr, X_in_va = X_train[in_tr_idx], X_train[in_va_idx]
                    y_in_tr, y_in_va = y_train[in_tr_idx], y_train[in_va_idx]

                    # build model with params
                    input_dim = X_in_tr.shape[-1] if X_in_tr.ndim > 1 else 1
                    model_inner = model_class(input_dim=input_dim, d_model=p.get('d_model', hidden_dim),
                                              nhead=p.get('nhead', nhead), num_layers=p.get('num_layers', num_layers),
                                              dropout=p.get('dropout', dropout), output_dim=len(np.unique(y_train)))

                    # train on inner train, validate on inner val
                    _, history_inner = fit_torch_model(
                        model_inner,
                        X_in_tr, y_in_tr,
                        X_in_va, y_in_va,
                        batch_size=p.get('batch_size', batch_size),
                        epochs=epochs,
                        learning_rate=p.get('lr', learning_rate),
                        device=device,
                        task_type="classification",
                        patience=patience,
                        weight_decay=p.get('weight_decay', weight_decay),
                        early_stopping_metric=early_stopping_metric,
                    )

                    # evaluate inner val using last predictions
                    preds_inner = history_inner[-1]['valid_predictions'] if history_inner and 'valid_predictions' in history_inner[-1] else predict_torch(model_inner, make_tensor_dataloader(X_in_va, y_in_va, batch_size=p.get('batch_size', batch_size), shuffle=False), device, 'classification')
                    try:
                        score = float(f1_score(y_in_va, preds_inner, average='macro'))
                    except Exception:
                        score = float(-np.inf)
                    inner_scores.append(score)

                mean_inner = float(np.mean(inner_scores)) if inner_scores else -np.inf
                if mean_inner > best_score:
                    best_score = mean_inner
                    best_params = p
        else:
            best_params = param_list[0]

        # Final model for outer fold uses best_params
        input_dim = X_train.shape[-1] if X_train.ndim > 1 else 1
        model = model_class(input_dim=input_dim, d_model=best_params.get('d_model', hidden_dim),
                            nhead=best_params.get('nhead', nhead), num_layers=best_params.get('num_layers', num_layers),
                            dropout=best_params.get('dropout', dropout), output_dim=len(np.unique(y_train)))

        # Train Transformer
        model, history = fit_torch_model(
            model,
            X_train, y_train,
            X_test, y_test,
            batch_size=best_params.get('batch_size', batch_size),
            epochs=epochs,
            learning_rate=best_params.get('lr', learning_rate),
            device=device,
            task_type="classification",
            patience=patience,
            weight_decay=best_params.get('weight_decay', weight_decay),
            early_stopping_metric=early_stopping_metric,
        )

        # Predictions auf Test-Set
        test_loader = make_tensor_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)
        y_pred = predict_torch(model, test_loader, device, "classification")

        # If we encoded labels earlier, inverse-transform predictions for storage
        if encoder is not None:
            try:
                y_pred_labels = encoder.inverse_transform(y_pred.astype(int))
                true_labels_to_store = true_labels_original
            except Exception:
                # fallback: store numeric predictions
                y_pred_labels = y_pred
                true_labels_to_store = true_labels_original
        else:
            y_pred_labels = y_pred
            true_labels_to_store = y_test

        # Speichere Fold-Ergebnisse + Trainings-Historie
        tracker.add_fold(fold_num, history, true_labels_to_store, y_pred_labels, usage_test, best_params=best_params)
        print(f"Fold {fold_num}/{n_splits}")

    return tracker


def train_transformer_cv(X, y, usage, model_class, n_splits: int = 5, epochs: int = 50,
                        batch_size: int = 32, learning_rate: float = 0.001,
                        hidden_dim: int = 64, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1,
                        patience: int = 4, device: str = "cpu", random_state: int = 42, weight_decay: float = 0.01,
                        early_stopping_metric: str = "val_loss"):
    """
    Train Transformer mit einfacher Cross Validation (StratifiedKFold) - KEINE nested CV.
    Diese Variante ist schneller als train_transformer_ncv() da sie nur äußere Folds hat.
    
    Args:
        X: Feature matrix (numpy array)
        y: Target labels (numpy array)
        usage: Usage column (numpy array)
        model_class: Transformer Model Class (z.B. TransformerModel aus models.py)
        n_splits: Anzahl der Folds (äußere Folds)
        epochs: Trainings-Epochen pro Fold
        batch_size: Batch size für DataLoader
        learning_rate: Learning rate
        hidden_dim: Hidden dimension
        device: "cpu" oder "cuda"
        random_state: Random seed
        
    Returns:
        TrainingTracker mit Fold-Ergebnissen + Trainings-Historie
    """
    tracker = TrainingTracker(
        "Transformer_CV",
        {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_dim': hidden_dim,
            'nhead': nhead,
            'num_layers': num_layers,
            'dropout': dropout,
            'n_splits': n_splits,
            'weight_decay': weight_decay,
            'early_stopping_metric': early_stopping_metric,
        }
    )
    
    # Encode string labels if present
    encoder = None
    y_np = np.asarray(y)
    if y_np.dtype.kind in ("U", "S", "O"):
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_np)
    else:
        y_encoded = y_np

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_num = 0

    for train_idx, test_idx in skf.split(X, y_encoded):
        fold_num += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        usage_test = usage[test_idx]
        true_labels_original = np.asarray(y)[test_idx] if encoder is not None else y_test

        # Standardisiere Daten pro Fold (fit auf Train, transform auf Test - kein Data Leakage)
        scaler = StandardScaler()
        X_train_shape = X_train.shape
        X_test_shape = X_test.shape
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_train = scaler.fit_transform(X_train_flat).reshape(X_train_shape)
        X_test = scaler.transform(X_test_flat).reshape(X_test_shape)

        # Initialisiere Transformer-Modell
        input_dim = X_train.shape[-1] if X_train.ndim > 1 else 1
        model = model_class(input_dim=input_dim, d_model=hidden_dim, nhead=nhead, num_layers=num_layers, dropout=dropout, output_dim=len(np.unique(y_train)))

        # Train Transformer
        model, history = fit_torch_model(
            model,
            X_train, y_train,
            X_test, y_test,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            task_type="classification",
            patience=patience,
            weight_decay=weight_decay,
            early_stopping_metric=early_stopping_metric,
        )

        # Predictions auf Test-Set
        test_loader = make_tensor_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)
        y_pred = predict_torch(model, test_loader, device, "classification")

        # Inverse-transform if needed
        if encoder is not None:
            try:
                y_pred_labels = encoder.inverse_transform(y_pred.astype(int))
                true_labels_to_store = true_labels_original
            except Exception:
                y_pred_labels = y_pred
                true_labels_to_store = true_labels_original
        else:
            y_pred_labels = y_pred
            true_labels_to_store = y_test

        # Speichere Fold-Ergebnisse + Trainings-Historie
        tracker.add_fold(fold_num, history, true_labels_to_store, y_pred_labels, usage_test)
        print(f"Fold {fold_num}/{n_splits} complete")

    return tracker


def train_lstm_ncv(X, y, usage, model_class, n_splits: int = 5, epochs: int = 50,
                   batch_size: int = 32, learning_rate: float = 0.001,
                   hidden_dim: int = 64, device: str = "cpu", random_state: int = 42):
    """
    Train LSTM mit Nested Cross Validation (StratifiedKFold).
    Gleiches Pattern wie train_transformer_ncv().
    
    Args: (analog zu train_transformer_ncv)
        
    Returns:
        TrainingTracker mit Fold-Ergebnissen + Trainings-Historie
    """
    tracker = TrainingTracker(
        "LSTM",
        {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_dim': hidden_dim,
        }
    )
    
    # Encode string labels if present
    encoder = None
    y_np = np.asarray(y)
    if y_np.dtype.kind in ("U", "S", "O"):
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_np)
    else:
        y_encoded = y_np

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_num = 0

    for train_idx, test_idx in skf.split(X, y_encoded):
        fold_num += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        usage_test = usage[test_idx]
        true_labels_original = np.asarray(y)[test_idx] if encoder is not None else y_test

        # Standardisiere Daten pro Fold (fit auf Train, transform auf Test - kein Data Leakage)
        scaler = StandardScaler()
        X_train_shape = X_train.shape
        X_test_shape = X_test.shape
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_train = scaler.fit_transform(X_train_flat).reshape(X_train_shape)
        X_test = scaler.transform(X_test_flat).reshape(X_test_shape)

        # Initialisiere LSTM-Modell
        input_dim = X_train.shape[-1] if X_train.ndim > 1 else 1
        model = model_class(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=len(np.unique(y_train)))

        # Train LSTM
        model, history = fit_torch_model(
            model,
            X_train, y_train,
            X_test, y_test,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            task_type="classification"
        )

        # Predictions auf Test-Set
        test_loader = make_tensor_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)
        y_pred = predict_torch(model, test_loader, device, "classification")

        # inverse-transform if needed
        if encoder is not None:
            try:
                y_pred_labels = encoder.inverse_transform(y_pred.astype(int))
                true_labels_to_store = true_labels_original
            except Exception:
                y_pred_labels = y_pred
                true_labels_to_store = true_labels_original
        else:
            y_pred_labels = y_pred
            true_labels_to_store = y_test

        # Speichere Fold-Ergebnisse + Trainings-Historie
        tracker.add_fold(fold_num, history, true_labels_to_store, y_pred_labels, usage_test)
        print(f"Fold {fold_num}/{n_splits} ✓")

    return tracker


def reproduce_rf(X, y, usage, seeds, split_ratio=0.65, n_estimators: int = 400):
    """Reproduce the simple seed-based RF evaluation from the notebook.

    Returns list of dicts with keys 'seed', 'True_label', 'Pred_label', 'workpiece_usage'.
    """
    all_folds_results = []
    n_samples = len(X)
    split_idx = int(n_samples * split_ratio)
    for s in seeds:
        X_s, y_s, usage_s = shuffle(X, y, usage, random_state=s)
        X_train, X_test = X_s[:split_idx], X_s[split_idx:]
        y_train, y_test = y_s[:split_idx], y_s[split_idx:]
        usage_train, usage_test = usage_s[:split_idx], usage_s[split_idx:]

        rf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        all_folds_results.append({
            'seed': s+1,
            'True_label': y_test,
            'Pred_label': y_pred_rf,
            'workpiece_usage': usage_test
        })
    return all_folds_results


def reproduce_rf_2f(X_reshape, y, usage, seeds, split_ratio=0.65, n_estimators: int = 400):
    """Same as reproduce_rf but for the 2-feature reshaped input."""
    return reproduce_rf(X_reshape, y, usage, seeds, split_ratio=split_ratio, n_estimators=n_estimators)


def run_tabpfn(X, y, usage, seeds, split_ratio=0.65, device='auto', n_estimators=4):
    """Run TabPFN loop as in the notebook. Requires `tabpfn` to be installed.

    Returns list of dicts same format as reproduce_rf.
    """
    if TabPFNClassifier is None:
        raise ImportError("tabpfn not available in this environment")
    all_seeds_tabpfn_results = []
    n_samples = len(X)
    split_idx = int(n_samples * split_ratio)
    test_scores = []
    for s in seeds:
        X_s, y_s, usage_s = shuffle(X, y, usage, random_state=s)
        X_train_val, X_test = X_s[:split_idx], X_s[split_idx:]
        y_train_val, y_test = y_s[:split_idx], y_s[split_idx:]
        usage_train_val, usage_test = usage_s[:split_idx], usage_s[split_idx:]

        tabpfn = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True, device=device, n_estimators=n_estimators)
        tabpfn.fit(X_train_val, y_train_val)
        y_test_pred = tabpfn.predict(X_test)

        score = f1_score(y_test, y_test_pred, average='macro')
        test_scores.append(score)

        all_seeds_tabpfn_results.append({
            'seed': s+1,
            'True_label': y_test,
            'Pred_label': y_test_pred,
            'workpiece_usage': usage_test
        })
    return all_seeds_tabpfn_results


def nested_cv_rf(X, y, usage, outer_folds: int = 5, inner_folds: int = 3, param_grid=None, random_state: int = 42):
    """Run pure nested cross validation for RandomForest.

    This version does not use an outer seed loop or a separate seed-based train/test split.
    The outer folds cover the full dataset, and the inner folds are used for model selection.

    Returns:
        all_folds_results, mean_test_score, std_test_score
    """
    if param_grid is None:
        param_grid = {'n_estimators': [100, 300, 500]}

    all_folds_results = []
    test_scores = []
    outer_skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)

    for fold_num, (train_idx, test_idx) in enumerate(outer_skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        usage_test = usage[test_idx]

        rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=inner_folds, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        test_f1_score = f1_score(y_test, y_pred, average='macro')
        test_scores.append(test_f1_score)

        all_folds_results.append({
            'fold': fold_num,
            'True_label': y_test,
            'Pred_label': y_pred,
            'workpiece_usage': usage_test,
            'best_params': grid_search.best_params_
        })

    return all_folds_results, float(np.mean(test_scores)), float(np.std(test_scores))


def nested_cv_over_seeds(X, y, usage, seeds, split_ratio=0.65, outer_folds=5, inner_folds=3, param_grid=None, random_state=42):
    """Backward-compatible wrapper for older notebook cells.

    Prefer nested_cv_rf() for the pure NCV workflow.
    """
    return nested_cv_rf(X, y, usage, outer_folds=outer_folds, inner_folds=inner_folds, param_grid=param_grid, random_state=random_state)


def cv_tabpfn(X, y, usage, n_folds: int = 5, device='auto', random_state: int = 42, n_estimators: int = 1):
    """Simple cross-validation for TabPFN (Foundation Model, no hyperparameter tuning).

    TabPFN is a pre-trained Foundation Model with no hyperparameters to optimize.
    This function simply evaluates it across folds to estimate generalization error.
    
    Returns: all_folds_results, mean_test_score, std_test_score
    """
    if TabPFNClassifier is None:
        raise ImportError("tabpfn not available in this environment")

    all_folds_results = []
    test_scores = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        usage_test = usage[test_idx]

        # Train TabPFN (no hyperparameters to tune)
        model = TabPFNClassifier(random_state=random_state, ignore_pretraining_limits=True, device=device, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        test_scores.append(test_f1)

        all_folds_results.append({
            'fold': fold_num,
            'True_label': y_test,
            'Pred_label': y_pred,
            'workpiece_usage': usage_test
        })

    return all_folds_results, float(np.mean(test_scores)), float(np.std(test_scores))


