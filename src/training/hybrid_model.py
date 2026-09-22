import pandas as pd
import numpy as np
import os
import glob
import spacy
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm

# --- CONFIGURATION ---
BERT_MODEL = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 128
BATCH_SIZE = 32
MODEL_SAVE_PATH = "rf_model.joblib"

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
        arquivo_fakerecogna = glob.glob("FakeRecogna/**/*.xlsx", recursive=True)[0]
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
        base_fakebr = "Fake.br-Corpus/full_texts"
        registros_fakebr = []
        for label in ["fake", "true"]:
            pasta = os.path.join(base_fakebr, label)
            arquivos = glob.glob(os.path.join(pasta, "*.txt"))
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
        df_factck = pd.read_csv("FACTCK.BR/FACTCKBR.tsv", sep="\t")
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

def get_bert_embeddings(texts):
    print(f"Generating BERT embeddings for {len(texts)} texts...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

def main():
    df = load_datasets()
    print(f"Unified Dataset Size: {df.shape}")
    print("Extracting stylometric features...")
    stylometry = np.array([get_stylometric_features(t) for t in tqdm(df["text"])])
    texts = df["text"].tolist()
    bert_features = get_bert_embeddings(texts)
    print("Fusing features...")
    X = np.hstack([bert_features, stylometry])
    y = df["label"].map({"fake": 1, "true": 0}).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training Hybrid Random Forest Model...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n--- HYBRID MODEL REPORT ---")
    print(classification_report(y_test, y_pred, target_names=['Verdadeiro (0)', 'Falso (1)']))
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    joblib.dump(clf, MODEL_SAVE_PATH)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
