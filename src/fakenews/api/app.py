from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import spacy
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
import uvicorn
from contextlib import asynccontextmanager

from fakenews.core.config import (
    BERT_MODEL_NAME, MAX_SEQUENCE_LENGTH, RF_MODEL_PATH,
    MIN_TEXT_LENGTH, MIN_INFO_TOKENS
)

# Global variables to hold models and tokenizer
resources = {
    "nlp": None,
    "bert_tokenizer": None,
    "bert_model": None,
    "rf_classifier": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Loading resources...")

    # Load Spacy
    try:
        resources["nlp"] = spacy.load("pt_core_news_sm")
    except:
        import subprocess
        subprocess.run(["python3", "-m", "spacy", "download", "pt_core_news_sm"])
        resources["nlp"] = spacy.load("pt_core_news_sm")

    # Load BERT
    resources["bert_tokenizer"] = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    resources["bert_model"] = AutoModel.from_pretrained(BERT_MODEL_NAME)
    resources["bert_model"].eval()

    # Load Random Forest
    try:
        resources["rf_classifier"] = joblib.load(RF_MODEL_PATH)
        print("Resources loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file {RF_MODEL_PATH} not found. Please train the model first.")

    yield
    # Shutdown logic (if any)
    resources.clear()

app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting fake news using a Hybrid BERT + Stylometric model",
    lifespan=lifespan
)

class NewsRequest(BaseModel):
    text: str

class NewsResponse(BaseModel):
    prediction: str
    confidence: float
    metrics: dict

def is_valid_news(text):
    """
    Validates if the input text is likely a news piece or just a random message.
    Criteria:
    1. Minimum length (defined in config).
    2. Minimum information density (defined in config).
    """
    if len(text.strip()) < MIN_TEXT_LENGTH:
        return False, f"O texto é muito curto para ser analisado como uma notícia (mínimo {MIN_TEXT_LENGTH} caracteres)."

    doc = resources["nlp"](text[:5000])
    # Count nouns (NOUN, PROPN) and verbs (VERB)
    info_tokens = sum(1 for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"])

    if info_tokens < MIN_INFO_TOKENS:
        return False, f"O texto não contém informações suficientes (falta de substantivos ou verbos, mínimo {MIN_INFO_TOKENS}) para ser classificado como notícia."

    return True, ""

def get_stylometric_features(text):
    # This logic should ideally be in model_utils.py, but keeping it here for API speed
    # unless you want a shared call.
    nlp_model = resources["nlp"]
    doc = nlp_model(text[:5000])
    tamanho = len(doc) if len(doc) > 0 else 1
    verbos = sum(1 for token in doc if token.pos_ == "VERB") / tamanho * 100
    adjetivos = sum(1 for token in doc if token.pos_ == "ADJ") / tamanho * 100
    pronomes = sum(1 for token in doc if token.pos_ == "PRON") / tamanho * 100

    from fakenews.core.config import SENSATIONALIST_WORDS, SCORE_EXCLAMACAO_MULT, SCORE_SENSACIONAL_MULT

    text_lower = text.lower()
    n_exclamacao = text.count("!")
    n_sensacional = sum(text_lower.count(p) for p in SENSATIONALIST_WORDS)
    n_maiusculas = sum(1 for p in text.split() if p.isupper() and len(p) > 1)
    palavras = max(len(text.split()), 1)
    score = min(round((n_exclamacao * SCORE_EXCLAMACAO_MULT + n_sensacional * SCORE_SENSACIONAL_MULT + n_maiusculas) / palavras * 100, 2), 10)

    return [verbos, adjetivos, pronomes, score]

def get_bert_embedding(text):
    tokenizer = resources["bert_tokenizer"]
    model = resources["bert_model"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_SEQUENCE_LENGTH)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token embedding
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

@app.post("/predict", response_model=NewsResponse)
async def predict(request: NewsRequest):
    if resources["rf_classifier"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 0. Validate if it's actually a news piece
    is_valid, error_msg = is_valid_news(request.text)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid input", "message": error_msg}
        )

    # 1. Extract Features
    stylometric = get_stylometric_features(request.text)
    bert_emb = get_bert_embedding(request.text)

    # 2. Fuse Features
    X = np.hstack([bert_emb, stylometric]).reshape(1, -1)

    # 3. Predict
    prediction_prob = resources["rf_classifier"].predict_proba(X)[0]
    prediction_class = resources["rf_classifier"].predict(X)[0]

    label = "fake" if prediction_class == 1 else "true"
    confidence = float(np.max(prediction_prob))

    return NewsResponse(
        prediction=label,
        confidence=confidence,
        metrics={
            "stylometry": {
                "perc_verbos": stylometric[0],
                "perc_adjetivos": stylometric[1],
                "perc_pronomes": stylometric[2],
                "score_emocional": stylometric[3]
            }
        }
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": resources["rf_classifier"] is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
