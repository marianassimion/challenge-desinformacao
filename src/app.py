from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import spacy
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
import uvicorn
from pathlib import Path

# --- CONFIGURATION ---
BERT_MODEL = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 128
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODELS_DIR / "rf_model.joblib"

app = FastAPI(title="Fake News Detector API", description="API for detecting fake news using a Hybrid BERT + Stylometric model")

# Global variables to hold models and tokenizer
nlp = None
bert_tokenizer = None
bert_model = None
rf_classifier = None

class NewsRequest(BaseModel):
    text: str

class NewsResponse(BaseModel):
    prediction: str
    confidence: float
    metrics: dict

def load_resources():
    global nlp, bert_tokenizer, bert_model, rf_classifier
    print("Loading resources...")

    # Load Spacy
    try:
        nlp = spacy.load("pt_core_news_sm")
    except:
        import subprocess
        subprocess.run(["python3", "-m", "spacy", "download", "pt_core_news_sm"])
        nlp = spacy.load("pt_core_news_sm")

    # Load BERT
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    bert_model = AutoModel.from_pretrained(BERT_MODEL)
    bert_model.eval()

    # Load Random Forest
    try:
        rf_classifier = joblib.load(MODEL_SAVE_PATH)
        print("Resources loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file {MODEL_SAVE_PATH} not found. Please train the model first.")

@app.on_event("startup")
async def startup_event():
    load_resources()

def get_stylometric_features(text):
    doc = nlp(text[:5000])
    tamanho = len(doc) if len(doc) > 0 else 1
    verbos = sum(1 for token in doc if token.pos_ == "VERB") / tamanho * 100
    adjetivos = sum(1 for token in doc if token.pos_ == "ADJ") / tamanho * 100
    pronomes = sum(1 for token in doc if token.pos_ == "PRON") / tamanho * 100

    palavras_sensacionalistas = [
        "urgente", "chocante", "bomba", "escândalo", "revelado", "segredo",
        "não vão acreditar", "atenção", "alerta", "exclusivo", "inacreditável",
        "impressionante", "cuidado", "compartilhe", "antes que apaguem"
    ]
    text_lower = text.lower()
    n_exclamacao = text.count("!")
    n_sensacional = sum(text_lower.count(p) for p in palavras_sensacionalistas)
    n_maiusculas = sum(1 for p in text.split() if p.isupper() and len(p) > 1)
    palavras = max(len(text.split()), 1)
    score = min(round((n_exclamacao * 1.5 + n_sensacional * 3 + n_maiusculas) / palavras * 100, 2), 10)

    return [verbos, adjetivos, pronomes, score]

def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use CLS token embedding
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

@app.post("/predict", response_model=NewsResponse)
async def predict(request: NewsRequest):
    if rf_classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Extract Features
    stylometric = get_stylometric_features(request.text)
    bert_emb = get_bert_embedding(request.text)

    # 2. Fuse Features
    X = np.hstack([bert_emb, stylometric]).reshape(1, -1)

    # 3. Predict
    prediction_prob = rf_classifier.predict_proba(X)[0]
    prediction_class = rf_classifier.predict(X)[0]

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
    return {"status": "ok", "model_loaded": rf_classifier is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
