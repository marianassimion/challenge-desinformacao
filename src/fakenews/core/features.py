import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from fakenews.core.config import (
    BERT_MODEL_NAME, MAX_SEQUENCE_LENGTH, BATCH_SIZE,
    SENSATIONALIST_WORDS, SCORE_EXCLAMACAO_MULT, SCORE_SENSACIONAL_MULT
)

# Global NLP object (will be initialized by ModelManager or a setup function)
nlp = None

def init_nlp():
    """Initializes the Spacy NLP model. Should be called once at startup."""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("pt_core_news_sm")
        except Exception:
            # Note: In a production environment, Spacy models should be installed via requirements/setup script.
            # This fallback is kept for compatibility but discouraged.
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", "pt_core_news_sm"])
            nlp = spacy.load("pt_core_news_sm")
    return nlp

def get_stylometric_features(text):
    \"\"\"
    Extracts stylometric features from a given text.
    Returns: [verb_percentage, adj_percentage, pron_percentage, sensationalism_score]
    \"\"\"
    _nlp = init_nlp()
    doc = _nlp(text[:5000])
    tamanho = len(doc) if len(doc) > 0 else 1
    verbos = sum(1 for token in doc if token.pos_ == "VERB") / tamanho * 100
    adjetivos = sum(1 for token in doc if token.pos_ == "ADJ") / tamanho * 100
    pronomes = sum(1 for token in doc if token.pos_ == "PRON") / tamanho * 100

    text_lower = text.lower()
    n_exclamacao = text.count("!")
    n_sensacional = sum(text_lower.count(p) for p in SENSATIONALIST_WORDS)
    n_maiusculas = sum(1 for p in text.split() if p.isupper() and len(p) > 1)
    palavras = max(len(text.split()), 1)
    score = min(round((n_exclamacao * SCORE_EXCLAMACAO_MULT + n_sensacional * SCORE_SENSACIONAL_MULT + n_maiusculas) / palavras * 100, 2), 10)

    return [verbos, adjetivos, pronomes, score]

def get_bert_embeddings(texts):
    \"\"\"
    Generates BERT embeddings (CLS token) for a list of texts.
    \"\"\"
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BERT Embeddings"):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt")
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embeddings)

    return np.vstack(embeddings)
