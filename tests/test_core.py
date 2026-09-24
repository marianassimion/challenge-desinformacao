import pytest
from fakenews.core.config import SENSATIONALIST_WORDS, SCORE_EXCLAMACAO_MULT, SCORE_SENSACIONAL_MULT
from fakenews.processing.model_utils import calcular_score_emocional, extrair_features_gramaticais

def test_calcular_score_emocional_sensacional():
    # Texto com várias palavras sensacionalistas e exclamações
    texto = "URGENTE! BOMBA REVELADA! O MUNDO ESTÁ EM CHOQUE! COMPARTILHE AGORA!"
    score = calcular_score_emocional(texto)
    # Espera-se que o score seja alto (limite é 10)
    assert score > 5.0
    assert score <= 10.0

def test_calcular_score_emocional_neutro():
    # Texto neutro, sem sensacionalismo
    texto = "O gato dorme no sofá durante a tarde em um dia ensolarado."
    score = calcular_score_emocional(texto)
    assert score < 2.0

def test_extrair_features_gramaticais_structure():
    texto = "O presidente viajou para a Europa para discutir acordos comerciais."
    features = extrair_features_gramaticais(texto)
    assert "perc_verbos" in features
    assert "perc_adjetivos" in features
    assert "perc_pronomes" in features
    for val in features.values():
        assert 0 <= val <= 100

def test_extrair_features_gramaticais_empty():
    texto = ""
    features = extrair_features_gramaticais(texto)
    assert features["perc_verbos"] == 0
