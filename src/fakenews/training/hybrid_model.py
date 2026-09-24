import pandas as pd
import numpy as np
import os
import glob
import spacy
import torch
import joblib
import mlflow
import mlflow.sklearn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from fakenews.core.config import (
    BERT_MODEL_NAME, MAX_SEQUENCE_LENGTH, BATCH_SIZE,
    RF_MODEL_PATH, DATA_DIR, SENSATIONALIST_WORDS,
    SCORE_EXCLAMACAO_MULT, SCORE_SENSACIONAL_MULT, RF_N_ESTIMATORS, RANDOM_STATE
)

# Load Spacy
try:
    nlp = spacy.load("pt_core_news_sm")
except:
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "pt_core_news_sm"])
    nlp = spacy.load("pt_core_news_sm")

def load_datasets():
    print("Loading and unifying datasets...")
    # 1. FakeRecogna
    try:
        arquivo_fakerecogna = glob.glob(str(DATA_DIR / "FakeRecogna/**/*.xlsx"), recursive=True)[0]
        df_recogna = pd.read_excel(arquivo_fakerecogna)
        df_recogna = df_recogna.rename(columns={
            "Titulo": "title", "Subtitulo": "subtitle", "Noticia": "text",
            "Categoria": "category", "Data": "date", "Autor": "author",
            "URL": "link", "Classe": "label"
        })
        df_recogna["label"] = df_recogna["label"].map({0: "fake", 1: "true"})
        df_recogna["source"] = "FakeRecogna"
    except Exception as e:
        print(f"Error loading FakeRecogna: {e}")
        df_recogna = pd.DataFrame()

    # 2. Fake.br-Corpus
    try:
        base_fakebr = DATA_DIR / "Fake.br-Corpus/full_texts"
        registros_fakebr = []
        for label in ["fake", "true"]:
            pasta = base_fakebr / label
            arquivos = glob.glob(str(pasta / "*.txt"), recursive=True)
            for arquivo in arquivos:
                with open(arquivo, "r", encoding="utf-8", errors="ignore") as f:
                    texto = f.read().strip()
                registros_fakebr.append({"text": texto, "label": label, "source": "Fake.Br"})
        df_fakebr = pd.DataFrame(registros_fakebr)
    except Exception as e:
        print(f"Error loading Fake.Br: {e}")
        df_fakebr = pd.DataFrame()

    # 3. FACTCK.BR
    try:
        df_factck = pd.read_csv(str(DATA_DIR / "FACTCK.BR/FACTCKBR.tsv"), sep="\t")
        df_factck = df_factck.rename(columns={
            "URL": "link", "Author": "author", "datePublished": "date",
            "claimReviewed": "claim", "reviewBody": "review", "title": "title",
            "ratingValue": "rating", "bestRating": "best_rating", "alternativeName": "label"
        })
        def classificar_factck(valor):
            v = str(valor).strip().lower()
            if v == "falso": return "fake"
            if v == "verdadeiro": return "true"
            return None
        df_factck["label"] = df_factck["label"].apply(classificar_factck)
        df_factck = df_factck.dropna(subset=["label"])
        df_factck["text"] = df_factck["review"].fillna("") + " " + df_factck["claim"].fillna("")
        df_factck["source"] = "FACTCK.BR"
    except Exception as e:
        print(f"Error loading FACTCK.BR: {e}")
        df_factck = pd.DataFrame()

    df_final = pd.concat([df_fakebr, df_recogna, df_factck], ignore_index=True)
    df_final = df_final.dropna(subset=["text", "label"])
    df_final = df_final[df_final["text"].str.strip() != ""]
    return df_final

def get_stylometric_features(text):
    doc = nlp(text[:5000])
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
    print(f"Generating BERT embeddings for {len(texts)} texts...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt")
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

def main():
    # MLflow setup
    mlflow.set_experiment("Fake_News_Detection_Hybrid")

    with mlflow.start_run():
        # Log Hyperparameters
        mlflow.log_param("bert_model", BERT_MODEL_NAME)
        mlflow.log_param("rf_n_estimators", RF_N_ESTIMATORS)
        mlflow.log_param("max_len", MAX_SEQUENCE_LENGTH)

        df = load_datasets()
        print(f"Unified Dataset Size: {df.shape}")
        mlflow.log_param("dataset_size", df.shape[0])

        print("Extracting stylometric features...")
        stylometry = np.array([get_stylometric_features(t) for t in tqdm(df["text"])])
        texts = df["text"].tolist()
        bert_features = get_bert_embeddings(texts)

        print("Fusing features...")
        X = np.hstack([bert_features, stylometry])
        y = df["label"].map({"fake": 1, "true": 0}).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

        print("Training Hybrid Random Forest Model...")
        clf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Log Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        print("\n--- HYBRID MODEL REPORT ---")
        print(classification_report(y_test, y_pred, target_names=['Verdadeiro (0)', 'Falso (1)']))

        print(f"Saving model to {RF_MODEL_PATH}...")
        joblib.dump(clf, RF_MODEL_PATH)

        # Log Model to MLflow
        mlflow.sklearn.log_model(clf, "random_forest_model")

        print("Model saved successfully and logged to MLflow!")

if __name__ == "__main__":
    main()
