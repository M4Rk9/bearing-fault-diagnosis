"""Evaluation helpers for bearing fault classifiers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score


CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print accuracy, macro-F1 and class-wise report."""
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Macro-F1:", round(f1_score(y_true, y_pred, average="macro"), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | None = None) -> None:
    """Plot a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    display.plot(xticks_rotation=30, cmap="Blues")
    plt.title("Bearing Fault Classification Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
