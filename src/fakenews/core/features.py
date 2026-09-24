import numpy as np
import torch
from fakenews.core.config import SENSATIONALIST_WORDS, SCORE_EXCLAMACAO_MULT, SCORE_SENSACIONAL_MULT, MAX_SEQUENCE_LENGTH
from fakenews.core.model_manager import ModelManager

class FeatureExtractor:
    """
    Responsible for transforming raw text into numerical features for the ML model.
    """
    def __init__(self):
        self.models = ModelManager()

    def extract_all(self, text: str):
        """
        Extracts both stylometric and BERT embeddings.
        """
        stylometric = self.get_stylometric_features(text)
        bert_emb = self.get_bert_embedding(text)
        return stylometric, bert_emb

    def get_stylometric_features(self, text: str):
        nlp_model = self.models.nlp
        doc = nlp_model(text[:5000])
        tamanho = len(doc) if len(doc) > 0 else 1

        verbos = sum(1 for token in doc if token.pos_ == "VERB") / tamanho * 100
        adjetivos = sum(1 for token in doc if token.pos_ == "ADJ") / tamanho * 100
        pronomes = sum(1 for token in doc if token.pos_ == "PRON") / tamanho * 100

        text_lower = text.lower()
        n_exclamacao = text.count("!")
        n_sensacional = sum(text_lower.count(p) for p in SENSATIONALIST_WORDS)
        n_maiusculas = sum(1 for p in text.split() if p.isupper() and len(p) > 1)
        palavras = max(len(text.split()), 1)

        score = min(round((n_exclamacao * SCORE_EXCLAMACAO_MULT +
                           n_sensacional * SCORE_SENSACIONAL_MULT +
                           n_maiusculas) / palavras * 100, 2), 10)

        return [verbos, adjetivos, pronomes, score]

    def get_bert_embedding(self, text: str):
        tokenizer = self.models.bert_tokenizer
        model = self.models.bert_model

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_SEQUENCE_LENGTH)
        with torch.no_grad():
            outputs = model(**inputs)

        return outputs.last_hidden_state[:, 0, :].numpy().flatten()
