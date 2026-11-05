"""Définition du réseau de neurones, entièrement initialisé au hasard."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_model(vocab_size, max_len, num_classes, embedding_dim=64, dense_units=32, dropout=0.5):
    """Embedding -> Global Average Pooling -> Dense -> Softmax."""
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(max_len,), dtype="int32", name="sequence"),
            # mask_zero=True : les zéros de padding sont ignorés par la moyenne qui suit.
            layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="embedding"),
            layers.GlobalAveragePooling1D(name="moyenne_des_mots"),
            layers.Dropout(dropout, name="dropout_1"),
            layers.Dense(
                dense_units,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
                name="dense",
            ),
            layers.Dropout(dropout, name="dropout_2"),
            layers.Dense(num_classes, activation="softmax", name="sortie"),
        ],
        name="classifieur_questions",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_ensemble(models, matrix):
    """Moyenne des probabilités de plusieurs réseaux entraînés séparément."""
    return np.mean([model.predict(matrix, verbose=0, batch_size=256) for model in models], axis=0)
