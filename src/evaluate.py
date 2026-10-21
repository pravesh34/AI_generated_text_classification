"""
Model evaluation, metrics computation, and visualization utilities.
Implements Phase 3 (Section 6.3) of the NIT Sikkim Project Report.
"""

from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Computes standard benchmark metrics:
    Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix", ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plots styled confusion matrix with counts and percentages.
    TN, FP, FN, TP layout matching Section 6.3 of the report.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    labels = [
        [f"{cm[0, 0]:,}\n({cm_percent[0, 0]:.1f}%)", f"{cm[0, 1]:,}\n({cm_percent[0, 1]:.1f}%)"],
        [f"{cm[1, 0]:,}\n({cm_percent[1, 0]:.1f}%)", f"{cm[1, 1]:,}\n({cm_percent[1, 1]:.1f}%)"]
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Pred: Human (0)", "Pred: AI (1)"],
        yticklabels=["Actual: Human (0)", "Actual: AI (1)"]
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    return ax


def plot_roc_curves(model_probs: Dict[str, np.ndarray], y_true: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plots overlay ROC curves for multiple models.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_val = roc_auc_score(y_true, prob)
        ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("ROC Curves Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    return ax
