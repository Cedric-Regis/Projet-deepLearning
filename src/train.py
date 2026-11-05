"""Entraînement du modèle à partir de zéro : plusieurs réseaux dont on moyenne les prédictions.

Usage : python -m src.train
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data_loader import PROCESSED_DIR, SEED, prepare_data, training_examples
from src.model import build_model
from src.preprocessing import PAD_ID, build_vocab, save_preprocessor, texts_to_matrix

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.json"
HISTORY_FILE = MODELS_DIR / "history.json"

N_MODELS = 5  # moyenne de 5 réseaux : prédictions plus stables, choisie en validation croisée
MAX_EPOCHS = 80
BATCH_SIZE = 32
MAX_LEN = 60  # choisi sur la validation : meilleure perte que 160 sur 3 graines aléatoires


def model_files():
    """Liste les réseaux enregistrés, dans l'ordre."""
    return sorted(MODELS_DIR.glob("classifier_*.keras"))


def train_one(seed, vocab_size, data, class_weight):
    """Entraîne un réseau. Seule la graine change : ses poids de départ sont donc différents."""
    tf.keras.utils.set_random_seed(seed)
    x_train, y_train, x_val, y_val = data
    model = build_model(vocab_size, MAX_LEN, len(class_weight))
    # On s'arrête quand la perte de validation ne progresse plus, et on garde le meilleur état.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=2,
    )
    return model, {key: [float(v) for v in values] for key, values in history.history.items()}


def train():
    if not (PROCESSED_DIR / "train.csv").exists():
        prepare_data()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    examples = training_examples(train_df)
    texts = [text for text, _ in examples]

    # Le vocabulaire vient de l'entraînement seul : pas de fuite de données.
    vocab = build_vocab(texts)
    labels = sorted(train_df["label"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}

    x_train = texts_to_matrix(texts, vocab, MAX_LEN)
    y_train = np.array([label_to_id[label] for _, label in examples])
    x_val = texts_to_matrix(val_df["question"], vocab, MAX_LEN)
    y_val = val_df["label"].map(label_to_id).to_numpy()

    # Les matières rares comptent davantage dans la perte, pour compenser le déséquilibre.
    weights = compute_class_weight("balanced", classes=np.arange(len(labels)), y=y_train)
    class_weight = dict(enumerate(weights))

    vocab_size = len(vocab) + 2  # +2 pour les identifiants PAD et OOV
    print(f"Exemples : {len(examples)} | Vocabulaire : {vocab_size} | Longueur : {MAX_LEN} | PAD = {PAD_ID}")

    MODELS_DIR.mkdir(exist_ok=True)
    for old_file in model_files():
        old_file.unlink()

    histories = []
    for index in range(N_MODELS):
        print(f"--- Réseau {index + 1} sur {N_MODELS}")
        data = (x_train, y_train, x_val, y_val)
        model, history = train_one(SEED + index, vocab_size, data, class_weight)
        model.save(MODELS_DIR / f"classifier_{index + 1}.keras")
        histories.append(history)
        tf.keras.backend.clear_session()

    save_preprocessor(PREPROCESSOR_FILE, vocab, labels, MAX_LEN)
    HISTORY_FILE.write_text(json.dumps(histories), encoding="utf-8")
    print(f"{N_MODELS} réseaux enregistrés dans {MODELS_DIR}")


if __name__ == "__main__":
    train()
