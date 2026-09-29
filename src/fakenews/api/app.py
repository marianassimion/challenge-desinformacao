import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional

from fakenews.core.config import MIN_TEXT_LENGTH, MIN_INFO_TOKENS
from fakenews.core.model_manager import ModelManager
from fakenews.core.features import FeatureExtractor

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Initializing ML Models...")
    ModelManager()  # Trigger singleton initialization
    yield
    # Shutdown logic (if any)

app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting fake news using a Hybrid BERT + Stylometric model",
    lifespan=lifespan
)

# Middleware to measure latency
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Request to {request.url.path} took {process_time:.4f}s")
    response.headers["X-Process-Time"] = str(process_time)
    return response

class NewsRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None

class NewsResponse(BaseModel):
    prediction: str
    confidence: float
    metrics: dict
    extracted_text: Optional[str] = None

def is_valid_news(text):
    """
    Validates if the input text is likely a news piece or just a random message.
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False, f"O texto é muito curto para ser analisado como uma notícia (mínimo {MIN_TEXT_LENGTH} caracteres)."

    # Access nlp via ModelManager
    nlp = ModelManager().nlp
    doc = nlp(text[:5000])
    info_tokens = sum(1 for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"])

    if info_tokens < MIN_INFO_TOKENS:
        return False, f"O texto não contém informações suficientes (falta de substantivos ou verbos, mínimo {MIN_INFO_TOKENS}) para ser classificado como notícia."

    return True, ""

@app.post("/predict", response_model=NewsResponse)
async def predict(request: NewsRequest):
    model_manager = ModelManager()

    if model_manager.rf_classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Logic to determine if we are analyzing text or URL
    text_to_analyze = None
    extracted_text = None

    if request.url:
        try:
            print(f"Extracting text from URL: {request.url}")
            from fakenews.processing.scraper import extract_text_from_url
            extracted_text = await extract_text_from_url(request.url)
            text_to_analyze = extracted_text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao extrair texto da URL: {str(e)}")
    elif request.text:
        text_to_analyze = request.text
    else:
        raise HTTPException(status_code=400, detail="Por favor, forneça 'text' ou 'url' na requisição.")

    # 0. Validate if it's actually a news piece
    is_valid, error_msg = is_valid_news(text_to_analyze)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid input", "message": error_msg}
        )

    # 1. Extract Features using the new FeatureExtractor
    extractor = FeatureExtractor()
    stylometric, bert_emb = extractor.extract_all(text_to_analyze)

    # 2. Fuse Features
    X = np.hstack([bert_emb, stylometric]).reshape(1, -1)

    # 3. Predict
    prediction_prob = model_manager.rf_classifier.predict_proba(X)[0]
    prediction_class = model_manager.rf_classifier.predict(X)[0]

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
        },
        extracted_text=extracted_text
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": ModelManager().rf_classifier is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
