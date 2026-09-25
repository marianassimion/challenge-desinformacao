import pandas as pd
import glob
from pathlib import Path
from fakenews.core.config import DATA_DIR

def load_datasets():
    """
    Loads and unifies datasets from multiple sources: FakeRecogna, Fake.br-Corpus, and FACTCK.BR.
    Returns:
        pd.DataFrame: A unified dataframe containing 'text' and 'label' columns.
    """
    print("Loading and unifying datasets...")

    # 1. FakeRecogna
    try:
        # Use glob to find the excel file regardless of exact name
        arquivos = glob.glob(str(DATA_DIR / "FakeRecogna/**/*.xlsx"), recursive=True)
        if not arquivos:
            print("Warning: No FakeRecogna files found.")
            df_recogna = pd.DataFrame()
        else:
            df_recogna = pd.read_excel(arquivos[0])
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
        print(f"Error loading Fake.br-Corpus: {e}")
        df_fakebr = pd.DataFrame()

    # 3. FACTCK.BR
    try:
        df_factck = pd.read_csv(str(DATA_DIR / "FACTCK.BR/FACTCKBR.tsv"), sep="\\t")
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

    if "text" not in df_final.columns or "label" not in df_final.columns or df_final.empty:
        raise RuntimeError(
            "Nenhum dado foi carregado de FakeRecogna, Fake.br-Corpus ou FACTCK.BR. "
            "Os datasets são git submodules: rode 'git submodule update --init --recursive' "
            "na raiz do repositório para baixá-los antes de treinar o modelo."
        )

    df_final = df_final.dropna(subset=["text", "label"])
    df_final = df_final[df_final["text"].str.strip() != ""]

    return df_final
