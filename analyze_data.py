import pandas as pd
import os
import glob

def get_dataset_info():
    # 1. FakeRecogna
    try:
        arquivo_fakerecogna = glob.glob("FakeRecogna/**/*.xlsx", recursive=True)[0]
        df_recogna = pd.read_excel(arquivo_fakerecogna)
        recogna_info = {
            "rows": len(df_recogna),
            "cols": len(df_recogna.columns),
            "nulls": df_recogna.isnull().sum().to_dict(),
            "label_dist": df_recogna["Classe"].value_counts().to_dict() if "Classe" in df_recogna.columns else "N/A"
        }
    except Exception as e:
        recogna_info = {"error": str(e)}

    # 2. Fake.br-Corpus
    try:
        base_fakebr = "Fake.br-Corpus/full_texts"
        fake_files = glob.glob(os.path.join(base_fakebr, "fake", "*.txt"))
        true_files = glob.glob(os.path.join(base_fakebr, "true", "*.txt"))
        fakebr_info = {
            "rows": len(fake_files) + len(true_files),
            "cols": 2, # text and label
            "nulls": "N/A (files)",
            "label_dist": {"fake": len(fake_files), "true": len(true_files)}
        }
    except Exception as e:
        fakebr_info = {"error": str(e)}

    # 3. FACTCK.BR
    try:
        df_factck = pd.read_csv("FACTCK.BR/FACTCKBR.tsv", sep="\t")
        factck_info = {
            "rows": len(df_factck),
            "cols": len(df_factck.columns),
            "nulls": df_factck.isnull().sum().to_dict(),
            "label_dist": df_factck["alternativeName"].value_counts().to_dict() if "alternativeName" in df_factck.columns else "N/A"
        }
    except Exception as e:
        factck_info = {"error": str(e)}

    print("RECOGNA:", recogna_info)
    print("FAKEBR:", fakebr_info)
    print("FACTCK:", factck_info)

if __name__ == "__main__":
    get_dataset_info()
