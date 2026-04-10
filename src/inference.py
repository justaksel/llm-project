import re
from functools import lru_cache
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "results" / "model_logreg.pkl"
VECTORIZER_PATH = BASE_DIR / "results" / "tfidf_vectorizer.pkl"


def clean_text(text: str) -> str:
    """
    Normalize raw review text to match the preprocessing logic
    used during training.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


@lru_cache(maxsize=1)
def load_artifacts():
    """
    Load the trained model and TF-IDF vectorizer once, then cache them.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_review(text: str) -> dict:
    """
    Predict whether a hotel review is truthful or deceptive.
    Returns the predicted label, numeric class, cleaned text,
    and confidence scores when available.
    """
    if not text or not text.strip():
        raise ValueError("Review text cannot be empty.")

    model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = int(model.predict(vectorized)[0])

    truthful_prob = None
    deceptive_prob = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized)[0]
        truthful_prob = float(probabilities[0])
        deceptive_prob = float(probabilities[1])

    label_map = {
        0: "Real / Truthful",
        1: "Fake / Deceptive",
    }

    return {
        "input_text": text,
        "clean_text": cleaned,
        "prediction_numeric": prediction,
        "prediction_label": label_map[prediction],
        "truthful_probability": truthful_prob,
        "deceptive_probability": deceptive_prob,
    }