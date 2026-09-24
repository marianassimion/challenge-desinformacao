import pandas as pd
import os
import glob
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def analyze():
    # Load Recogna
    try:
        arquivo_fakerecogna = next((DATA_DIR / "FakeRecogna").rglob("*.xlsx"))
        df_recogna = pd.read_excel(arquivo_fakerecogna)
        df_recogna = df_recogna.rename(columns={"Titulo": "title", "Subtitulo": "subtitle", "Noticia": "text", "Classe": "label"})
        df_recogna["label"] = df_recogna["label"].map({0: "fake", 1: "true"})
    except Exception as e:
        df_recogna = pd.DataFrame()

    # Load Fake.br
    try:
        base_fakebr = DATA_DIR / "Fake.br-Corpus" / "full_texts"
        registros_fakebr = []
        for label in ["fake", "true"]:
            pasta = base_fakebr / label
            for arquivo in pasta.glob("*.txt"):
                with open(arquivo, "r", encoding="utf-8", errors="ignore") as f:
                    registros_fakebr.append({"text": f.read().strip(), "label": label})
        df_fakebr = pd.DataFrame(registros_fakebr)
    except Exception as e:
        df_fakebr = pd.DataFrame()

    # Load FactCK
    try:
        df_factck = pd.read_csv(DATA_DIR / "FACTCK.BR" / "FACTCKBR.tsv", sep="\t")
        df_factck["label"] = df_factck["alternativeName"].apply(lambda x: "fake" if str(x).lower() == "falso" else ("true" if str(x).lower() == "verdadeiro" else None))
        df_factck = df_factck.dropna(subset=["label"])
        df_factck["text"] = df_factck["review"].fillna("") + " " + df_factck["claim"].fillna("")
    except Exception as e:
        df_factck = pd.DataFrame()

    # Global stats
    total_rows = len(df_recogna) + len(df_fakebr) + len(df_factck)
    
    print(f"TOTAL_ROWS: {total_rows}")
    
    # Lengths
    all_texts = pd.concat([
        df_recogna["text"] if not df_recogna.empty else pd.Series(),
        df_fakebr["text"] if not df_fakebr.empty else pd.Series(),
        df_factck["text"] if not df_factck.empty else pd.Series()
    ])
    lengths = all_texts.str.len()
    print(f"LENGTH_MEAN: {lengths.mean():.2f}")
    print(f"LENGTH_MEDIAN: {lengths.median():.2f}")
    print(f"LENGTH_MAX: {lengths.max()}")
    print(f"LENGTH_MIN: {lengths.min()}")

    # Label Dist
    labels = pd.concat([
        df_recogna["label"] if not df_recogna.empty else pd.Series(),
        df_fakebr["label"] if not df_fakebr.empty else pd.Series(),
        df_factck["label"] if not df_factck.empty else pd.Series()
    ])
    print(f"DIST:\n{labels.value_counts(normalize=True) * 100}")

    # Nulls Recogna
    if not df_recogna.empty:
        print(f"RECOGNA_NULLS:\n{df_recogna.isnull().sum()}")

analyze()
