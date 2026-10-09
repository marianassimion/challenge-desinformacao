import argparse
import hashlib
import pickle

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)
from tqdm import tqdm

from fakenews.processing.dataloader import load_datasets, _join_text
from fakenews.core.features import get_stylometric_features, get_bert_embeddings
from fakenews.core.config import (
    BERT_MODEL_NAME, MAX_SEQUENCE_LENGTH, BATCH_SIZE,
    XGB_MODEL_PATH, DATA_DIR, SENSATIONALIST_WORDS,
    SCORE_EXCLAMACAO_MULT, SCORE_SENSACIONAL_MULT, RANDOM_STATE
)

# Feature extraction is now centralized in fakenews.core.features
# Data loading is now centralized in fakenews.processing.dataloader

XGB_N_ESTIMATORS = 200
EXTRA_SOURCE = "BancoProprio"


CACHE_PATH = DATA_DIR / "cache" / "bert_embeddings.pkl"
CACHE_CHUNK = 512


def _key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cached_bert_embeddings(texts):
    """
    Embeddings do BERT com cache em disco (chave = hash do texto).
    Textos já calculados em rodadas anteriores não são recalculados, e o cache é
    salvo a cada bloco, então dá para interromper (Ctrl+C) e continuar depois.
    """
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)

    pendentes = {}
    for t in texts:
        k = _key(t)
        if k not in cache and k not in pendentes:
            pendentes[k] = t
    print(f"Embeddings: {len(texts) - len(pendentes)} no cache, {len(pendentes)} a calcular")

    if pendentes:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        items = list(pendentes.items())
        for i in range(0, len(items), CACHE_CHUNK):
            bloco = items[i:i + CACHE_CHUNK]
            embs = get_bert_embeddings([t for _, t in bloco], tokenizer=tokenizer, model=model)
            for (k, _), e in zip(bloco, embs):
                cache[k] = e
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)

    return np.vstack([cache[_key(t)] for t in texts])


def build_features(texts):
    texts = list(texts)
    stylometry = np.array([get_stylometric_features(t) for t in tqdm(texts, desc="Estilometria")])
    bert_features = cached_bert_embeddings(texts)
    return np.hstack([bert_features, stylometry])


def evaluate_banco_holdout(clf, use_title):
    """Recall em fake inédita: o holdout do banco nunca entra no treino."""
    path = DATA_DIR / "extra" / "banco_holdout.csv"
    if not path.exists():
        print("Holdout do banco não encontrado — pulando.")
        return None
    h = pd.read_csv(path)
    if use_title:
        texts = [_join_text(t, st, tx) for t, st, tx in zip(h["title"], h["subtitle"], h["text"])]
    else:
        texts = h["text"].tolist()
    pred = clf.predict(build_features(texts))
    return float((pred == 1).mean()), len(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-title", action="store_true", help="não concatena título/subtítulo ao texto")
    ap.add_argument("--no-banco", action="store_true", help="não usa o banco próprio no treino")
    ap.add_argument("--no-save", action="store_true", help="não sobrescreve models/xgb_model.joblib (use em experimentos)")
    ap.add_argument("--clf", choices=["xgb", "hgb"], default="xgb",
                    help="xgb = XGBoost (modelo oficial); hgb = HistGradientBoosting do scikit-learn "
                         "(só para experimentos em máquina sem libomp; exige --no-save)")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    use_title, use_banco = not args.no_title, not args.no_banco
    if args.clf == "hgb" and not args.no_save:
        ap.error("--clf hgb só pode ser usado com --no-save (o modelo oficial da API é XGBoost).")

    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Fake_News_Detection_Hybrid")

    run_name = args.run_name or f"title={use_title}_banco={use_banco}"
    with mlflow.start_run(run_name=run_name):
        # Log Hyperparameters
        mlflow.log_param("bert_model", BERT_MODEL_NAME)
        mlflow.log_param("xgb_n_estimators", XGB_N_ESTIMATORS)
        mlflow.log_param("max_len", MAX_SEQUENCE_LENGTH)
        mlflow.log_param("use_title", use_title)
        mlflow.log_param("use_banco", use_banco)
        mlflow.log_param("classifier", args.clf)

        df = load_datasets(use_title=use_title, use_banco=use_banco)
        print(f"Unified Dataset Size: {df.shape}")
        mlflow.log_param("dataset_size", df.shape[0])

        print("Extracting features...")
        X = build_features(df["text"].tolist())
        y = df["label"].map({"fake": 1, "true": 0}).values

        # Split FIXO: só as fontes originais entram no split 80/20 (mesma partição de antes,
        # pois o banco é concatenado por último). O banco entra só no treino.
        is_extra = (df["source"] == EXTRA_SOURCE).values
        idx = np.arange(len(df))
        base_idx = idx[~is_extra]
        train_base, test_idx = train_test_split(
            base_idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y[base_idx]
        )
        train_idx = np.concatenate([train_base, idx[is_extra]])
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        mlflow.log_param("n_train", len(train_idx))
        mlflow.log_param("n_test", len(test_idx))
        mlflow.log_param("n_banco_treino", int(is_extra.sum()))

        print(f"Training Hybrid Model ({args.clf})...")
        if args.clf == "xgb":
            # XGBoost: mais resistente a Domain Shift
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=XGB_N_ESTIMATORS, random_state=RANDOM_STATE, eval_metric="logloss")
        else:
            clf = HistGradientBoostingClassifier(max_iter=XGB_N_ESTIMATORS, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Log Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)  # 1 - taxa de falso positivo (notícia verdadeira marcada como fake)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("specificity", specificity)

        res = evaluate_banco_holdout(clf, use_title)
        if res:
            mlflow.log_metric("holdout_banco_recall", res[0])
            print(f"Recall no holdout do banco ({res[1]} fakes inéditas): {res[0]:.3f}")

        print("\n--- HYBRID MODEL REPORT ---")
        print(classification_report(y_test, y_pred, target_names=["Verdadeiro (0)", "Falso (1)"]))
        print(f"Specificity (verdadeiras corretas): {specificity:.3f}")

        if args.no_save:
            print("--no-save: modelo NÃO foi salvo em models/.")
        else:
            print(f"Saving model to {XGB_MODEL_PATH}...")
            XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, XGB_MODEL_PATH)

        # Log Model to MLflow (só o XGBoost oficial)
        if args.clf == "xgb":
            mlflow.xgboost.log_model(clf, "xgboost_model")

        print("Run logged to MLflow!")


if __name__ == "__main__":
    main()
