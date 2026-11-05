"""Prétraitement TALN : nettoyage, tokenisation, vocabulaire, séquences et padding."""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

PAD_ID = 0  # identifiant réservé au remplissage (padding)
OOV_ID = 1  # identifiant réservé aux mots inconnus (out of vocabulary)
NUM_TOKEN = "<num>"  # tous les nombres partagent le même jeton

# Mots-outils français (écrits sans accents, car le nettoyage retire les accents).
# Ils apparaissent dans toutes les matières et n'aident donc pas à les distinguer.
STOPWORDS = set(
    """
    a ai au aux avec ce ces cet cette d dans de des du elle elles en est et etre eu il ils
    j je l la le les leur leurs lui m ma mais me mes meme n ne nos notre nous on ont ou par
    pas pour qu que qui s sa se ses si son sont sur t ta te tes toi ton tu un une vos votre
    vous y c
    """.split()
)

# Un jeton est soit un mot alphanumérique (« h2o »), soit un symbole mathématique utile.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[=+*/^<>%√π∫∑]")


def clean_text(text):
    """Met en minuscules, retire les accents et normalise les caractères spéciaux."""
    text = str(text).lower().replace("œ", "oe").replace("æ", "ae")
    decomposed = unicodedata.normalize("NFKD", text)  # « é » devient « e » + accent
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def stem(token):
    """Racinisation légère : « dérivée » et « dérivées » deviennent le même jeton."""
    if len(token) <= 4 or token.startswith("<"):
        return token
    if token[-1] in "sx":
        token = token[:-1]
    while len(token) > 4 and token[-1] == "e":
        token = token[:-1]
    return token


def tokenize(text, use_stem=True):
    """Découpe un texte en jetons, retire les mots-outils et réduit les mots à leur racine."""
    tokens = []
    for token in TOKEN_PATTERN.findall(clean_text(text)):
        if token in STOPWORDS:
            continue
        if token.isdigit():
            tokens.append(NUM_TOKEN)
        else:
            tokens.append(stem(token) if use_stem else token)
    return tokens


def build_vocab(texts, max_size=20000, min_freq=2):
    """Construit le vocabulaire à partir des textes d'ENTRAÎNEMENT uniquement."""
    counts = Counter(token for text in texts for token in tokenize(text))
    frequent = [t for t, n in counts.most_common(max_size - 2) if n >= min_freq]
    return {token: index for index, token in enumerate(frequent, start=2)}


def encode(text, vocab):
    """Convertit un texte en liste d'identifiants numériques."""
    return [vocab.get(token, OOV_ID) for token in tokenize(text)]


def pad_sequences(sequences, max_len):
    """Tronque ou complète chaque séquence pour obtenir la longueur max_len."""
    output = np.full((len(sequences), max_len), PAD_ID, dtype="int32")
    for row, sequence in enumerate(sequences):
        truncated = sequence[:max_len]
        output[row, : len(truncated)] = truncated
    return output


def texts_to_matrix(texts, vocab, max_len):
    """Chaîne complète : textes -> matrice d'entiers prête pour le réseau."""
    return pad_sequences([encode(text, vocab) for text in texts], max_len)


def save_preprocessor(path, vocab, labels, max_len):
    """Sauvegarde ce qu'il faut pour traiter une nouvelle question comme à l'entraînement."""
    payload = {"vocab": vocab, "labels": labels, "max_len": max_len}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_preprocessor(path):
    """Recharge le vocabulaire, la liste des classes et la longueur des séquences."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["vocab"], payload["labels"], payload["max_len"]
