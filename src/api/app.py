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

# --- CONFIGURATION ---
BERT_MODEL = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 128
# Updated path to point to the models directory
MODEL_SAVE_PATH = "models/rf_model.joblib"

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
    resources["bert_tokenizer"] = AutoTokenizer.from_pretrained(BERT_MODEL)
    resources["bert_model"] = AutoModel.from_pretrained(BERT_MODEL)
    resources["bert_model"].eval()

    # Load Random Forest
    try:
        resources["rf_classifier"] = joblib.load(MODEL_SAVE_PATH)
        print("Resources loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file {MODEL_SAVE_PATH} not found. Please train the model first.")

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

def get_stylometric_features(text):
    nlp_model = resources["nlp"]
    doc = nlp_model(text[:5000])
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
    tokenizer = resources["bert_tokenizer"]
    model = resources["bert_model"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token embedding
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

@app.post("/predict", response_model=NewsResponse)
async def predict(request: NewsRequest):
    if resources["rf_classifier"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

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
