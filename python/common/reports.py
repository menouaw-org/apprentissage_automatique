import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from python.common.dataset_64x64 import CLASSES


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Aucune ligne à écrire dans {path}")
        fieldnames = list(rows[0].keys())

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(
        history_rows: list[dict[str, object]],
        output_path: Path,
        title: str,
        metric_name: str = "accuracy",
) -> None:
    if not history_rows:
        raise ValueError("Aucun historique disponible pour tracer les courbes.")

    train_key = f"train_{metric_name}"
    validation_key = f"validation_{metric_name}"

    plt.figure(figsize=(10, 6))

    folds = sorted({int(row["fold"]) for row in history_rows})

    for fold in folds:
        fold_rows = [row for row in history_rows if int(row["fold"]) == fold]
        epochs = [int(row["epoch"]) for row in fold_rows]
        train_values = [float(row[train_key]) for row in fold_rows]
        validation_values = [float(row[validation_key]) for row in fold_rows]

        plt.plot(
            epochs,
            train_values,
            linestyle="--",
            marker="o",
            alpha=0.6,
            label=f"Fold {fold} train",
        )
        plt.plot(
            epochs,
            validation_values,
            linestyle="-",
            marker="o",
            alpha=0.9,
            label=f"Fold {fold} validation",
        )

    plt.title(title)
    plt.xlabel("Époque")
    plt.ylabel(metric_name.capitalize())
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES)
    plt.yticks(tick_marks, CLASSES)

    threshold = matrix.max() / 2.0 if matrix.max() > 0 else 0.0

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value > threshold else "black"

            plt.text(
                column_index,
                row_index,
                str(value),
                horizontalalignment="center",
                color=color,
            )

    plt.ylabel("Classe réelle")
    plt.xlabel("Classe prédite")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()