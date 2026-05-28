from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def pred_filter(fold_results, label="binary", usage1st=True):
    """Auswertung der Fold-Ergebnisse mit Label- und Usage-Filtering (aus deinem Notebook)."""
    metriken = {"Accuracy": [], "Precision": [], "Recall": [], "F1_macro": []}
    if label == "binary":
        average_method = "binary"
    elif label in ["single", "grouped"]:
        average_method = "macro"
    
    for fold in fold_results:
        y_val = fold["True_label"]
        y_pred = fold["Pred_label"]
        usage_val = fold["workpiece_usage"]
        if usage1st == True:
            usage_filter = usage_val == 0
            y_val_filtered = y_val[usage_filter]
            y_pred_filtered = y_pred[usage_filter]
        else:
            y_val_filtered = y_val
            y_pred_filtered = y_pred
        if label == "binary":
            y_val_binary = []
            y_pred_binary = []
            for l in y_val_filtered:
                if str(l).startswith('0'):
                    y_val_binary.append(0)
                else:
                    y_val_binary.append(1)
            for l in y_pred_filtered:
                if str(l).startswith('0'):
                    y_pred_binary.append(0)
                else:
                    y_pred_binary.append(1)
            y_val_filtered = np.array(y_val_binary)
            y_pred_filtered = np.array(y_pred_binary)
        elif label == "grouped":
            y_val_group = []
            y_pred_group = []
            for l in y_val_filtered:
                if str(l).startswith('0'):
                    y_val_group.append(0)
                elif str(l).startswith('1'):
                    y_val_group.append(1)
                elif str(l).startswith('2') or str(l).startswith('3'):
                    y_val_group.append(2)
                elif str(l).startswith('4') or str(l).startswith('5'):
                    y_val_group.append(3)
                elif str(l).startswith('6') or str(l).startswith('7'):
                    y_val_group.append(4)
            for l in y_pred_filtered:
                if str(l).startswith('0'):
                    y_pred_group.append(0)
                elif str(l).startswith('1'):
                    y_pred_group.append(1)
                elif str(l).startswith('2') or str(l).startswith('3'):
                    y_pred_group.append(2)
                elif str(l).startswith('4') or str(l).startswith('5'):
                    y_pred_group.append(3)
                elif str(l).startswith('6') or str(l).startswith('7'):
                    y_pred_group.append(4)
            y_val_filtered = np.array(y_val_group)
            y_pred_filtered = np.array(y_pred_group)
        elif label == "single":
            y_val_filtered = np.array(y_val_filtered)
            y_pred_filtered = np.array(y_pred_filtered)
        metriken["Accuracy"].append(accuracy_score(y_val_filtered, y_pred_filtered))
        metriken["Precision"].append(precision_score(y_val_filtered, y_pred_filtered, average=average_method, zero_division=0))
        metriken["Recall"].append(recall_score(y_val_filtered, y_pred_filtered, average=average_method, zero_division=0))
        metriken["F1_macro"].append(f1_score(y_val_filtered, y_pred_filtered, average=average_method, zero_division=0))
    
    stats = {"Accuracy_mean": np.mean(metriken["Accuracy"]), "Accuracy_std": np.std(metriken["Accuracy"]), "Precision_mean": np.mean(metriken["Precision"]), "Precision_std": np.std(metriken["Precision"]), "Recall_mean": np.mean(metriken["Recall"]), "Recall_std": np.std(metriken["Recall"]), "F1_macro_mean": np.mean(metriken["F1_macro"]), "F1_macro_std": np.std(metriken["F1_macro"])}
    for i in stats:
        if i != "Accuracy_std" and i != "Precision_std" and i != "Recall_std" and i != "F1_macro_std":
            stats[i] *= 100
    df = pd.DataFrame([stats])
    return df


def evaluate_folds(fold_results):
    """
    Vereinheitlichte Auswertung: Evaluiere alle Label-Strategien × Usage-Modi in einem Schritt.
    
    Args:
        fold_results: Liste von Dicts mit Keys 'True_label', 'Pred_label', 'workpiece_usage'
        
    Returns:
        DataFrame mit 6 Reihen (binary/single/grouped × usage1st/all) + 'Labelstrategie' Spalte
    """
    # Rufe pred_filter 6x auf (3 Label-Strategien × 2 Usage-Modi)
    df_binary_s1st = pred_filter(fold_results, label="binary", usage1st=True)
    df_binary_all = pred_filter(fold_results, label="binary", usage1st=False)
    df_single_s1st = pred_filter(fold_results, label="single", usage1st=True)
    df_single_all = pred_filter(fold_results, label="single", usage1st=False)
    df_grouped_s1st = pred_filter(fold_results, label="grouped", usage1st=True)
    df_grouped_all = pred_filter(fold_results, label="grouped", usage1st=False)
    
    # Concateniere alle Ergebnisse
    results = pd.concat([
        df_binary_s1st, df_binary_all,
        df_single_s1st, df_single_all,
        df_grouped_s1st, df_grouped_all
    ], ignore_index=True)
    
    # Füge Labelstrategie-Spalte hinzu
    results.insert(0, 'Labelstrategie', [
        'Binary_s1st', 'Binary_all',
        'Single_s1st', 'Single_all',
        'Grouped_s1st', 'Grouped_all'
    ])
    
    return results
