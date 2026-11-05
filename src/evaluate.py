"""Évaluation finale sur le jeu de test : métriques, matrice de confusion et courbes.

Usage : python -m src.evaluate
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data_loader import PROCESSED_DIR, ROOT
from src.preprocessing import load_preprocessor, texts_to_matrix, tokenize
from src.model import predict_ensemble
from src.train import HISTORY_FILE, MODELS_DIR, PREPROCESSOR_FILE, model_files

METRICS_FILE = MODELS_DIR / "metrics.json"
CURVES_FILE = MODELS_DIR / "training_curves.png"
CONFUSION_FILE = MODELS_DIR / "confusion_matrix.png"

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
INK = "#52514e"
SHORT_QUESTION = 30  # nombre maximal de jetons d'une question « courte »
LOW_CONFIDENCE = 0.60
CONTROL_FILE = ROOT / "data" / "questions_controle.csv"


def wilson_interval(accuracy, n, z=1.96):
    """Intervalle de confiance à 95 % de l'exactitude : le test ne contient que n questions."""
    denominator = 1 + z**2 / n
    center = (accuracy + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(accuracy * (1 - accuracy) / n + z**2 / (4 * n**2)) / denominator
    return float(center - margin), float(center + margin)


def compute_metrics(y_true, y_pred, labels):
    """Calcule toutes les métriques à partir des prédictions réelles sur le test."""
    ids = np.arange(len(labels))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro = precision_recall_fscore_support(y_true, y_pred, labels=ids, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=ids, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    majority_share = float(np.bincount(y_true, minlength=len(labels)).max() / len(y_true))
    low, high = wilson_interval(accuracy, len(y_true))
    return {
        "n_test": int(len(y_true)),
        "accuracy": accuracy,
        "accuracy_ci95": [low, high],
        "baseline_majority_accuracy": majority_share,
        "macro": dict(zip(("precision", "recall", "f1"), map(float, macro[:3]))),
        "weighted": dict(zip(("precision", "recall", "f1"), map(float, weighted[:3]))),
        "per_class": {
            label: {
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int(per_class[3][i]),
            }
            for i, label in enumerate(labels)
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def confidence_split(models, vocab, labels, max_len):
    """Sur la validation : réussite du modèle au-dessous et au-dessus du seuil de confiance."""
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    y_true = val_df["label"].map({label: i for i, label in enumerate(labels)}).to_numpy()
    probabilities = predict_ensemble(models, texts_to_matrix(val_df["question"], vocab, max_len))
    correct = probabilities.argmax(axis=1) == y_true
    low = probabilities.max(axis=1) < LOW_CONFIDENCE
    return {
        "threshold": LOW_CONFIDENCE,
        "low_n": int(low.sum()),
        "low_accuracy": float(correct[low].mean()),
        "high_n": int((~low).sum()),
        "high_accuracy": float(correct[~low].mean()),
    }


def control_questions(models, vocab, labels, max_len):
    """Questions courtes rédigées à la main : un contrôle indicatif, hors du dataset."""
    control = pd.read_csv(CONTROL_FILE)
    probabilities = predict_ensemble(models, texts_to_matrix(control["question"], vocab, max_len))
    control["predicted"] = [labels[i] for i in probabilities.argmax(axis=1)]
    control["confidence"] = probabilities.max(axis=1).astype(float)
    control["correct"] = control["label"] == control["predicted"]
    return {
        "accuracy": {name: float(part["correct"].mean()) for name, part in control.groupby("ensemble")},
        "n": {name: int(len(part)) for name, part in control.groupby("ensemble")},
        "details": control.to_dict(orient="records"),
    }


def _style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d6d5d0")
    ax.tick_params(colors=INK, length=0)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(history, best_epoch, path):
    """Deux panneaux séparés : la perte et l'exactitude n'ont pas la même échelle."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    panels = (
        ("loss", "Perte (Loss)"),
        ("accuracy", "Exactitude (Accuracy)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, title) in zip(axes, panels):
        ax.plot(epochs, history[key], color=TRAIN_COLOR, linewidth=2, label="Entraînement")
        ax.plot(epochs, history[f"val_{key}"], color=VAL_COLOR, linewidth=2, label="Validation")
        ax.axvline(best_epoch, color="#8f8e88", linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            f"Modèle conservé (époque {best_epoch})",
            xy=(best_epoch, 0.5),
            xycoords=("data", "axes fraction"),
            xytext=(5, 0),
            textcoords="offset points",
            color=INK,
            fontsize=9,
        )
        ax.set_title(title, loc="left", color="#0b0b0b", fontsize=12)
        ax.set_xlabel("Époque", color=INK)
        ax.legend(frameon=False, labelcolor=INK)
        _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(matrix, labels, path):
    """Matrice de confusion : une seule teinte, et le nombre écrit dans chaque case."""
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", color=INK)
    ax.set_yticks(range(len(labels)), labels, color=INK)
    ax.set_xlabel("Matière prédite", color=INK)
    ax.set_ylabel("Matière réelle", color=INK)
    ax.set_title("Matrice de confusion (jeu de test)", loc="left", color="#0b0b0b", fontsize=12)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    threshold = matrix.max() * 0.55
    for row, col in np.ndindex(matrix.shape):
        ax.text(
            col,
            row,
            matrix[row, col],
            ha="center",
            va="center",
            color="white" if matrix[row, col] > threshold else "#0b0b0b",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate():
    models = [tf.keras.models.load_model(path) for path in model_files()]
    vocab, labels, max_len = load_preprocessor(PREPROCESSOR_FILE)
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    x_test = texts_to_matrix(test_df["question"], vocab, max_len)
    y_true = test_df["label"].map({label: i for i, label in enumerate(labels)}).to_numpy()
    probabilities = predict_ensemble(models, x_test)
    y_pred = probabilities.argmax(axis=1)

    histories = json.loads(Path(HISTORY_FILE).read_text(encoding="utf-8"))
    best_epochs = [int(np.argmin(h["val_loss"])) + 1 for h in histories]
    history, best_epoch = histories[0], best_epochs[0]  # courbes du premier réseau

    metrics = compute_metrics(y_true, y_pred, labels)
    metrics["best_epoch"] = best_epoch
    metrics["best_epochs"] = best_epochs
    metrics["epochs_first_model"] = len(history["loss"])
    metrics["n_models"] = len(models)
    metrics["vocab_size"] = len(vocab) + 2
    metrics["params_per_model"] = int(models[0].count_params())
    metrics["low_confidence_validation"] = confidence_split(models, vocab, labels, max_len)
    metrics["control_questions"] = control_questions(models, vocab, labels, max_len)
    # Les longs extraits sont plus faciles à classer : on mesure aussi les questions courtes.
    correct = y_pred == y_true
    is_short = test_df["question"].map(lambda q: len(tokenize(q)) <= SHORT_QUESTION).to_numpy()
    metrics["accuracy_by_length"] = {
        "short": {"n": int(is_short.sum()), "accuracy": float(correct[is_short].mean())},
        "long": {"n": int((~is_short).sum()), "accuracy": float(correct[~is_short].mean())},
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_curves(history, best_epoch, CURVES_FILE)
    plot_confusion(metrics["confusion_matrix"], labels, CONFUSION_FILE)

    low, high = metrics["accuracy_ci95"]
    print(f"Test : {metrics['n_test']} questions")
    print(f"Exactitude : {metrics['accuracy']:.3f} (IC 95 % : {low:.3f} - {high:.3f})")
    print(f"Référence (toujours prédire la matière la plus fréquente) : {metrics['baseline_majority_accuracy']:.3f}")
    print("Par longueur :", metrics["accuracy_by_length"])
    print("Réseaux :", metrics["n_models"], "| Époques retenues :", best_epochs, "| Vocabulaire :", metrics["vocab_size"], "| Paramètres par réseau :", metrics["params_per_model"])
    print("Validation, seuil de confiance :", metrics["low_confidence_validation"])
    print("Questions de contrôle :", metrics["control_questions"]["accuracy"], metrics["control_questions"]["n"])
    for row in metrics["control_questions"]["details"]:
        if not row["correct"]:
            print("  ERREUR", row["ensemble"], "|", row["question"], "|", row["label"], "->", row["predicted"], round(row["confidence"], 2))
    print("Macro :", {k: round(v, 3) for k, v in metrics["macro"].items()})
    for label, values in metrics["per_class"].items():
        print(label, {k: round(v, 3) for k, v in values.items()})
    print(np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    evaluate()
