import pandas as pd
import os
import glob
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Ensure spacy model is downloaded
try:
    nlp = spacy.load("pt_core_news_sm")
except:
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "pt_core_news_sm"])
    nlp = spacy.load("pt_core_news_sm")

print("Loading datasets...")

# --- 1. FakeRecogna ---
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
    df_recogna["language"] = "pt"
    print(f"Loaded FakeRecogna: {df_recogna.shape}")
except Exception as e:
    print(f"Error loading FakeRecogna: {e}")
    df_recogna = pd.DataFrame()

# --- 2. Fake.br-Corpus ---
try:
    base_fakebr = "Fake.br-Corpus/full_texts"
    registros_fakebr = []
    for label in ["fake", "true"]:
        pasta = os.path.join(base_fakebr, label)
        arquivos = glob.glob(os.path.join(pasta, "*.txt"))
        for arquivo in arquivos:
            with open(arquivo, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read().strip()
            registros_fakebr.append({
                "text": texto, "label": label, "source": "Fake.Br", "language": "pt"
            })
    df_fakebr = pd.DataFrame(registros_fakebr)
    print(f"Loaded Fake.Br: {df_fakebr.shape}")
except Exception as e:
    print(f"Error loading Fake.Br: {e}")
    df_fakebr = pd.DataFrame()

# --- 3. FACTCK.BR ---
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
    df_factck["source"] = "FACTCK.BR"
    df_factck["language"] = "pt"
    # To match the others, we take the 'review' or 'claim' as the text
    df_factck["text"] = df_factck["review"].fillna("") + " " + df_factck["claim"].fillna("")
    print(f"Loaded FACTCK.BR: {df_factck.shape}")
except Exception as e:
    print(f"Error loading FACTCK.BR: {e}")
    df_factck = pd.DataFrame()

# --- Unify ---
df_final = pd.concat([df_fakebr, df_recogna, df_factck], ignore_index=True)
df_final = df_final.dropna(subset=["text", "label"])
df_final = df_final[df_final["text"].str.strip() != ""]
print(f"Final Unified Dataset Shape: {df_final.shape}")

# --- Feature Engineering ---
def contar_classes_gramaticais(texto):
    doc = nlp(texto[:5000])
    verbos = sum(1 for token in doc if token.pos_ == "VERB")
    adjetivos = sum(1 for token in doc if token.pos_ == "ADJ")
    pronomes = sum(1 for token in doc if token.pos_ == "PRON")
    tamanho = len(doc) if len(doc) > 0 else 1
    return pd.Series([(verbos/tamanho)*100, (adjetivos/tamanho)*100, (pronomes/tamanho)*100])

print("Extracting grammar features... (This may take a while)")
df_final[['perc_verbos', 'perc_adjetivos', 'perc_pronomes']] = df_final['text'].apply(contar_classes_gramaticais)

palavras_sensacionalistas = [
    "urgente", "chocante", "bomba", "escândalo", "revelado", "segredo",
    "não vão acreditar", "atenção", "alerta", "exclusivo", "inacreditável",
    "impressionante", "cuidado", "compartilhe", "antes que apaguem"
]

def score_emocional(texto):
    texto_lower = texto.lower()
    n_exclamacao = texto.count("!")
    n_sensacional = sum(texto_lower.count(p) for p in palavras_sensacionalistas)
    n_maiusculas = sum(1 for p in texto.split() if p.isupper() and len(p) > 1)
    palavras = max(len(texto.split()), 1)
    raw = (n_exclamacao * 1.5 + n_sensacional * 3 + n_maiusculas) / palavras * 100
    return min(round(raw, 2), 10)

df_final["score_emocional"] = df_final["text"].apply(score_emocional)

# --- Modeling ---
X = df_final[['perc_verbos', 'perc_adjetivos', 'perc_pronomes', 'score_emocional']]
y = df_final['label'].map({'fake': 1, 'true': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_rf = RandomForestClassifier(random_state=42, n_estimators=100)
modelo_rf.fit(X_train, y_train)

y_pred = modelo_rf.predict(X_test)
print("\n--- FINAL UNIFIED REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Verdadeiro (0)', 'Falso (1)']))

importancias = pd.DataFrame(
    modelo_rf.feature_importances_,
    index=X.columns,
    columns=['Importância']
).sort_values('Importância', ascending=False)
print("\n--- FEATURE IMPORTANCE ---")
print(importancias)
