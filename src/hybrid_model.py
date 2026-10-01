import pandas as pd
import numpy as np
import spacy
import torch
import joblib
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm


BERT_MODEL = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 512 
BATCH_SIZE = 32
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODELS_DIR / "xgb_model.joblib"

# --- ACELERAÇÃO APPLE SILICON (M4) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Aceleração de Treinamento Ativada: Apple M4 (MPS)")
else:
    device = torch.device("cpu")

try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "pt_core_news_sm"])
    nlp = spacy.load("pt_core_news_sm")

def load_datasets():
    print("Carregando e unificando datasets...")
    try:
        arquivo_fakerecogna = next((DATA_DIR / "FakeRecogna").rglob("*.xlsx"))
        df_recogna = pd.read_excel(arquivo_fakerecogna)
        df_recogna = df_recogna.rename(columns={"Noticia": "text", "Classe": "label"})
        df_recogna["label"] = df_recogna["label"].map({0: "fake", 1: "true"})
    except Exception as e:
        df_recogna = pd.DataFrame()

    try:
        base_fakebr = DATA_DIR / "Fake.br-Corpus" / "full_texts"
        registros_fakebr = []
        for label in ["fake", "true"]:
            for arquivo in (base_fakebr / label).glob("*.txt"):
                with open(arquivo, "r", encoding="utf-8", errors="ignore") as f:
                    registros_fakebr.append({"text": f.read().strip(), "label": label})
        df_fakebr = pd.DataFrame(registros_fakebr)
    except Exception as e:
        df_fakebr = pd.DataFrame()

    df_final = pd.concat([df_fakebr, df_recogna], ignore_index=True).dropna(subset=["text", "label"])
    return df_final[df_final["text"].str.strip() != ""]

def get_stylometric_features(text):
    doc = nlp(text[:5000])
    tamanho = max(len(doc), 1)
    verbos = sum(1 for token in doc if token.pos_ == "VERB") / tamanho * 100
    adjetivos = sum(1 for token in doc if token.pos_ == "ADJ") / tamanho * 100
    pronomes = sum(1 for token in doc if token.pos_ == "PRON") / tamanho * 100

    palavras_sensacionalistas = ["urgente", "chocante", "bomba", "escândalo", "revelado", "segredo", "atenção", "alerta", "exclusivo", "inacreditável"]
    text_lower = text.lower()
    score = min(round((text.count("!") * 1.5 + sum(text_lower.count(p) for p in palavras_sensacionalistas) * 3 + sum(1 for p in text.split() if p.isupper() and len(p) > 1)) / max(len(text.split()), 1) * 100, 2), 10)
    return [verbos, adjetivos, pronomes, score]

def get_bert_embeddings(texts):
    print(f"Gerando embeddings do BERT para {len(texts)} textos usando o chip M4...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL).to(device)
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()} # Envia os textos para a GPU do Mac
            
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() # Traz de volta para a memória RAM
            embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

def main():
    df = load_datasets()
    print(f"Tamanho do Dataset Unificado: {df.shape}")
    
    print("Extraindo features estilométricas...")
    stylometry = np.array([get_stylometric_features(t) for t in tqdm(df["text"])])
    
    bert_features = get_bert_embeddings(df["text"].tolist())
    
    print("Treinando o Modelo Híbrido XGBoost...")
    X = np.hstack([bert_features, stylometry])
    y = df["label"].map({"fake": 1, "true": 0}).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    
    print("\n--- RELATÓRIO DO MODELO ---")
    print(classification_report(y_test, clf.predict(X_test), target_names=['Verdadeiro (0)', 'Falso (1)']))
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"Sucesso! Modelo salvo em {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()