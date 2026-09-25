import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from fakenews.core.config import (
    BERT_MODEL_NAME, MAX_SEQUENCE_LENGTH, BATCH_SIZE,
    SENSATIONALIST_WORDS, SCORE_EXCLAMACAO_MULT, SCORE_SENSACIONAL_MULT
)

# Fallback nlp instance for standalone use (e.g. the training script), so it
# doesn't need a ModelManager. The API always passes its own nlp/tokenizer/
# model (loaded once by ModelManager) so resources are never loaded twice.
_default_nlp = None


def _get_default_nlp():
    global _default_nlp
    if _default_nlp is None:
        _default_nlp = spacy.load("pt_core_news_sm")
    return _default_nlp


def extrair_features_gramaticais(texto, nlp=None):
    """
    Extrai a densidade de verbos, adjetivos e pronomes usando SpaCy.
    Returns: dict with perc_verbos, perc_adjetivos, perc_pronomes (0-100).
    """
    _nlp = nlp or _get_default_nlp()
    doc = _nlp(texto[:5000])
    tamanho = len(doc) if len(doc) > 0 else 1

    verbos = sum(1 for token in doc if token.pos_ == "VERB")
    adjetivos = sum(1 for token in doc if token.pos_ == "ADJ")
    pronomes = sum(1 for token in doc if token.pos_ == "PRON")

    return {
        "perc_verbos": (verbos / tamanho) * 100,
        "perc_adjetivos": (adjetivos / tamanho) * 100,
        "perc_pronomes": (pronomes / tamanho) * 100
    }


def calcular_score_emocional(texto):
    """Calcula o grau de sensacionalismo do texto baseado em heurísticas."""
    texto_lower = texto.lower()
    n_exclamacao = texto.count("!")
    n_sensacional = sum(texto_lower.count(p) for p in SENSATIONALIST_WORDS)
    n_maiusculas = sum(1 for p in texto.split() if p.isupper() and len(p) > 1)

    palavras = max(len(texto.split()), 1)
    raw = (n_exclamacao * SCORE_EXCLAMACAO_MULT + n_sensacional * SCORE_SENSACIONAL_MULT + n_maiusculas) / palavras * 100
    return min(round(raw, 2), 10)


def get_stylometric_features(text, nlp=None):
    """
    Extracts stylometric features from a given text.
    Returns: [verb_percentage, adj_percentage, pron_percentage, sensationalism_score]
    """
    gram = extrair_features_gramaticais(text, nlp=nlp)
    score = calcular_score_emocional(text)
    return [gram["perc_verbos"], gram["perc_adjetivos"], gram["perc_pronomes"], score]


def get_bert_embeddings(texts, tokenizer=None, model=None):
    """
    Generates BERT embeddings (CLS token) for a list of texts.
    If tokenizer/model aren't provided (standalone/training use), they are
    loaded from BERT_MODEL_NAME.
    """
    _tokenizer = tokenizer or AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    _model = model or AutoModel.from_pretrained(BERT_MODEL_NAME)
    _model.eval()
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BERT Embeddings"):
            batch = texts[i : i + BATCH_SIZE]
            inputs = _tokenizer(batch, padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt")
            outputs = _model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


class FeatureExtractor:
    """
    Used by the API (src/fakenews/api/app.py). Pulls the nlp/BERT resources
    that ModelManager already loaded once at startup, and reuses the exact
    same feature functions the training script uses, so train and inference
    never drift apart.
    """

    def __init__(self):
        from fakenews.core.model_manager import ModelManager
        self._manager = ModelManager()

    def extract_all(self, text):
        stylometric = get_stylometric_features(text, nlp=self._manager.nlp)
        bert_emb = get_bert_embeddings(
            [text], tokenizer=self._manager.bert_tokenizer, model=self._manager.bert_model
        )[0]
        return stylometric, bert_emb
