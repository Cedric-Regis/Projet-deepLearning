"""Application Streamlit de classification des questions éducatives.

Usage : streamlit run app.py
"""

import json
import re

import altair as alt
import pandas as pd
import streamlit as st
import tensorflow as tf

from src.data_loader import PROCESSED_DIR
from src.evaluate import CONFUSION_FILE, CURVES_FILE, METRICS_FILE
from src.model import predict_ensemble
from src.preprocessing import OOV_ID, encode, load_preprocessor, pad_sequences, tokenize
from src.train import PREPROCESSOR_FILE, model_files

LOW_CONFIDENCE = 0.60  # réussite mesurée de part et d'autre : voir models/metrics.json
PREDICTED_COLOR = "#2a78d6"
OTHER_COLOR = "#a3a29c"

EXAMPLES = [
    "Qu'est-ce que la photosynthèse ?",
    "Résoudre l'équation 2x + 3 = 7.",
    "Expliquez la deuxième loi de Newton.",
    "Quelles sont les causes de la Seconde Guerre mondiale ?",
    "Quel est le pH d'une solution d'acide chlorhydrique ?",
]


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_resources():
    models = [tf.keras.models.load_model(path) for path in model_files()]
    vocab, labels, max_len = load_preprocessor(PREPROCESSOR_FILE)
    return models, vocab, labels, max_len


def analyse(question, models, vocab, labels, max_len):
    """Prétraite la question, puis renvoie les étapes et les probabilités du modèle."""
    tokens = tokenize(question)
    ids = encode(question, vocab)
    unknown = [t for t, i in zip(tokens, ids) if i == OOV_ID]
    result = {"tokens": tokens, "ids": ids, "unknown": unknown, "probabilities": None}
    # Sans aucun mot connu, le modèle n'a rien sur quoi s'appuyer : on ne prédit pas.
    if len(unknown) < len(tokens):
        padded = pad_sequences([ids], max_len)
        probabilities = predict_ensemble(models, padded)[0]
        result["probabilities"] = dict(zip(labels, map(float, probabilities)))
    return result


def probability_chart(probabilities):
    rows = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    best = rows[0][0]
    data = pd.DataFrame(
        {
            "Matière": [f"{label}  {value * 100:.1f} %".replace(".", ",") for label, value in rows],
            "Probabilité": [value for _, value in rows],
            "Couleur": [PREDICTED_COLOR if label == best else OTHER_COLOR for label, _ in rows],
            "Nom": [label for label, _ in rows],
        }
    )
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, size=20)
        .encode(
            y=alt.Y("Matière:N", sort=None, title=None, axis=alt.Axis(labelLimit=220, ticks=False)),
            x=alt.X(
                "Probabilité:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", title=None, tickCount=5),
            ),
            color=alt.Color("Couleur:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("Nom:N", title="Matière"),
                alt.Tooltip("Probabilité:Q", title="Probabilité", format=".1%"),
            ],
        )
        .properties(height=190)
    )
    st.altair_chart(chart, width="stretch")


def set_example(question):
    st.session_state.question = question
    st.session_state.result = None


def classifier_tab(models, vocab, labels, max_len):
    st.markdown("**Exemples à essayer**")
    for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        column.button(
            example,
            on_click=set_example,
            args=(example,),
            width="stretch",
        )

    question = st.text_area(
        "Votre question (en français)",
        key="question",
        height=100,
        placeholder="Exemple : Quelle est la structure d'un atome ?",
    )

    if st.button("Classifier", type="primary"):
        if question.strip():
            st.session_state.result = analyse(question, models, vocab, labels, max_len)
        else:
            st.session_state.result = None
            st.warning("Saisissez d'abord une question.")

    result = st.session_state.get("result")
    if not result:
        return

    probabilities = result["probabilities"]
    if probabilities is None:
        st.warning(
            "Cette question ne contient aucun mot exploitable (mots-outils ou mots jamais vus "
            "à l'entraînement) : le modèle ne peut pas proposer de matière."
        )
        return

    best = max(probabilities, key=probabilities.get)
    confidence = probabilities[best]
    left, right = st.columns(2)
    left.metric("Matière prédite", best)
    right.metric("Confiance", f"{confidence * 100:.1f} %".replace(".", ","))

    if confidence < LOW_CONFIDENCE:
        st.warning("Confiance faible : cette prédiction est à prendre avec prudence.")
    if result["unknown"]:
        names = ", ".join(result["unknown"])
        st.info(f"Mots absents du vocabulaire d'entraînement : {names}")

    st.markdown("**Probabilités par matière**")
    probability_chart(probabilities)

    with st.expander("Voir le prétraitement de cette question"):
        st.markdown("**Jetons après nettoyage, retrait des mots-outils et racinisation**")
        st.code(" | ".join(result["tokens"]), language=None)
        st.markdown(f"**Séquence numérique** (ensuite complétée par des 0 jusqu'à {max_len} valeurs)")
        st.code(str(result["ids"]), language=None)


def performance_tab():
    if not METRICS_FILE.exists():
        st.error("Métriques introuvables. Lancez d'abord : python -m src.evaluate")
        return

    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    low, high = metrics["accuracy_ci95"]
    st.markdown(f"Résultats mesurés sur **{metrics['n_test']} questions de test**, jamais vues à l'entraînement.")

    columns = st.columns(4)
    columns[0].metric("Exactitude", f"{metrics['accuracy'] * 100:.1f} %".replace(".", ","))
    columns[1].metric("Précision (macro)", f"{metrics['macro']['precision']:.3f}".replace(".", ","))
    columns[2].metric("Rappel (macro)", f"{metrics['macro']['recall']:.3f}".replace(".", ","))
    columns[3].metric("Score F1 (macro)", f"{metrics['macro']['f1']:.3f}".replace(".", ","))
    length = metrics["accuracy_by_length"]
    caption = (
        f"Intervalle de confiance à 95 % de l'exactitude : {low * 100:.1f} % à {high * 100:.1f} %. "
        f"Référence (toujours prédire la matière la plus fréquente) : "
        f"{metrics['baseline_majority_accuracy'] * 100:.1f} %. "
        f"Questions courtes (30 jetons ou moins, {length['short']['n']} questions) : "
        f"{length['short']['accuracy'] * 100:.1f} %. "
        f"Questions longues ({length['long']['n']} questions, surtout en histoire) : "
        f"{length['long']['accuracy'] * 100:.1f} %."
    )
    st.caption(re.sub(r"(\d)\.(\d)", r"\1,\2", caption))

    table = pd.DataFrame(metrics["per_class"]).T.rename(
        columns={"precision": "Précision", "recall": "Rappel", "f1": "F1", "support": "Questions"}
    )
    table["Questions"] = table["Questions"].astype(int)
    table = table.rename_axis("Matière")
    st.dataframe(table.style.format(precision=3, decimal=","), width="stretch")

    st.image(str(CURVES_FILE), caption="Évolution de la perte et de l'exactitude pendant l'entraînement")
    st.image(str(CONFUSION_FILE), caption="Lignes : matière réelle. Colonnes : matière prédite.", width=560)


def model_tab(models):
    st.markdown(
        "**Dataset.** [openai/MMMLU](https://huggingface.co/datasets/openai/MMMLU), configuration "
        "française : questions de MMLU traduites par des traducteurs professionnels. "
        "Quinze sous-matières existantes sont regroupées en cinq matières."
    )
    splits = {name: pd.read_csv(PROCESSED_DIR / f"{name}.csv") for name in ("train", "val", "test")}
    counts = pd.DataFrame({name: part["label"].value_counts() for name, part in splits.items()})
    counts = counts.rename(columns={"train": "Entraînement", "val": "Validation", "test": "Test"})
    counts = counts.rename_axis("Matière")
    st.dataframe(counts, width="stretch")

    st.markdown(
        "**Réseau de neurones.** Tous les poids partent de valeurs aléatoires et sont appris "
        f"pendant notre entraînement. Aucun modèle préentraîné n'est utilisé. L'application fait la "
        f"moyenne de {len(models)} réseaux identiques, entraînés séparément. Architecture de chacun :"
    )
    layers = pd.DataFrame(
        [
            {"Couche": layer.name, "Type": type(layer).__name__, "Paramètres": layer.count_params()}
            for layer in models[0].layers
        ]
    )
    st.dataframe(layers, width="stretch", hide_index=True)
    st.caption(f"Total : {models[0].count_params():,} paramètres appris par réseau.".replace(",", " "))


def main():
    st.set_page_config(page_title="Classification de questions", page_icon="🎓", layout="wide")
    st.title("Système intelligent de classification des questions éducatives")
    st.markdown(
        "Ce réseau de neurones, entraîné à partir de zéro, identifie la matière d'une question "
        "parmi **Biologie, Chimie, Histoire, Mathématiques et Physique**. "
        "Il ne répond pas à la question : il la classe, et choisit toujours parmi ces cinq matières."
    )

    if not model_files():
        st.error("Modèle introuvable. Lancez d'abord : python -m src.train")
        st.stop()

    models, vocab, labels, max_len = load_resources()
    classify, performance, about = st.tabs(["Classifier", "Performances", "Données et modèle"])
    with classify:
        classifier_tab(models, vocab, labels, max_len)
    with performance:
        performance_tab()
    with about:
        model_tab(models)


if __name__ == "__main__":
    main()
