import spacy
import pandas as pd
import joblib
import re
import os

# Carregamento do modelo de linguagem do SpaCy
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    # Comando para instalar caso não esteja no ambiente
    os.system("python -m spacy download pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm")

# Lista de palavras sensacionalistas para o score emocional
PALAVRAS_SENSACIONALISTAS = [
    "urgente", "chocante", "bomba", "escândalo", "revelado", "segredo",
    "não vão acreditar", "atenção", "alerta", "exclusivo", "inacreditável",
    "impressionante", "cuidado", "compartilhe", "antes que apaguem"
]

def extrair_features_gramaticais(texto):
    """Extrai a densidade de verbos, adjetivos e pronomes usando SpaCy."""
    doc = nlp(texto[:5000])
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
    n_sensacional = sum(texto_lower.count(p) for p in PALAVRAS_SENSACIONALISTAS)
    n_maiusculas = sum(1 for p in texto.split() if p.isupper() and len(p) > 1)

    palavras = max(len(texto.split()), 1)
    raw = (n_exclamacao * 1.5 + n_sensacional * 3 + n_maiusculas) / palavras * 100
    return min(round(raw, 2), 10)

def prever_risco_desinformacao(texto, modelo_path='rf_model.joblib'):
    """Pipeline completo: Texto -> Features -> Modelo -> Probabilidade."""
    try:
        # 1. Extrair Features
        features_gram = extrair_features_gramaticais(texto)
        score_emocional = calcular_score_emocional(texto)

        # 2. Montar DataFrame para o modelo (mesma ordem do treino)
        df_input = pd.DataFrame([{
            'perc_verbos': features_gram['perc_verbos'],
            'perc_adjetivos': features_gram['perc_adjetivos'],
            'perc_pronomes': features_gram['perc_pronomes'],
            'score_emocional': score_emocional
        }])

        # 3. Carregar modelo e predizer
        modelo = joblib.load(modelo_path)
        # predict_proba retorna [[prob_classe_0, prob_classe_1]]
        probabilidade_fake = modelo.predict_proba(df_input)[0][1] * 100

        return {
            "risco": round(probabilidade_fake, 1),
            "score_emocional": score_emocional,
            "perc_adjetivos": round(features_gram['perc_adjetivos'], 1),
            "sucesso": True
        }
    except Exception as e:
        print(f"Erro na predição: {e}")
        return {"sucesso": False, "erro": str(e)}
