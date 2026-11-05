"""Téléchargement du dataset, sélection des matières et séparation train / val / test.

Usage : python -m src.data_loader
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import tokenize

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "mmmlu_fr.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# Révision figée du dépôt Hugging Face : tout le monde télécharge exactement les mêmes données.
REVISION = "325a01dc3e173cac1578df94120499aaca2e2504"
DATASET_URL = (
    f"https://huggingface.co/datasets/openai/MMMLU/resolve/{REVISION}/test/mmlu_FR-FR.csv"
)

# Regroupement de sous-matières existantes du dataset en cinq matières scolaires.
# Aucune question n'est réétiquetée individuellement.
SUBJECT_TO_LABEL = {
    "abstract_algebra": "Mathématiques",
    "college_mathematics": "Mathématiques",
    "elementary_mathematics": "Mathématiques",
    "high_school_mathematics": "Mathématiques",
    "college_biology": "Biologie",
    "high_school_biology": "Biologie",
    "college_physics": "Physique",
    "conceptual_physics": "Physique",
    "high_school_physics": "Physique",
    "college_chemistry": "Chimie",
    "high_school_chemistry": "Chimie",
    "high_school_european_history": "Histoire",
    "high_school_us_history": "Histoire",
    "high_school_world_history": "Histoire",
    "prehistory": "Histoire",
}

SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def download_raw():
    """Télécharge le CSV français une seule fois, puis réutilise la copie locale."""
    if not RAW_FILE.exists():
        RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATASET_URL).to_csv(RAW_FILE, index=False)
    return pd.read_csv(RAW_FILE)


def select_subjects(raw):
    """Garde les cinq matières et retire les doublons, qui fausseraient le test."""
    df = raw[raw["Subject"].isin(SUBJECT_TO_LABEL)].copy()
    df = df.rename(columns={"Question": "question", "Subject": "source_subject"})
    df["label"] = df["source_subject"].map(SUBJECT_TO_LABEL)
    df = df[["question", "label", "source_subject"]].dropna(subset=["question"])

    # Clé de comparaison : deux questions identiques après prétraitement sont des doublons.
    df["key"] = df["question"].map(lambda q: " ".join(tokenize(q, use_stem=False)))
    df = df[df["key"] != ""]
    labels_per_key = df.groupby("key")["label"].transform("nunique")
    df = df[labels_per_key == 1]  # une même question sous deux matières : ambiguë, retirée
    return df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)


def split_dataset(df):
    """Sépare en 70 % / 15 % / 15 % en conservant la proportion de chaque matière."""
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
    )
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val["label"],
        random_state=SEED,
    )
    return train, val, test


def training_examples(train_df):
    """Exemples d'entraînement : la question, puis les textes réels associés du dataset.

    Pour chaque question, on ajoute la question suivie de ses quatre choix, puis la bonne
    réponse seule. Ces textes enrichissent le vocabulaire de chaque matière. Seul
    l'entraînement en profite : la validation et le test ne contiennent que des questions.
    """
    choices = download_raw().drop_duplicates("Question").set_index("Question")
    rows = list(zip(train_df["question"], train_df["label"]))
    for question, label in zip(train_df["question"], train_df["label"]):
        row = choices.loc[question]
        rows.append((f"{question} {row['A']} {row['B']} {row['C']} {row['D']}", label))
        rows.append((str(row[row["Answer"]]), label))
    return [(text, label) for text, label in rows if tokenize(text)]  # jamais de texte vide


def prepare_data():
    """Exécute toute la préparation et écrit les trois fichiers dans data/processed/."""
    raw = download_raw()
    df = select_subjects(raw)
    train, val, test = split_dataset(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    print(f"Dataset brut : {len(raw)} questions, {raw['Subject'].nunique()} sous-matières")
    print(f"Après sélection et dédoublonnage : {len(df)} questions")
    print(df["label"].value_counts().to_string())
    print(f"Entraînement : {len(train)} | Validation : {len(val)} | Test : {len(test)}")


if __name__ == "__main__":
    prepare_data()
