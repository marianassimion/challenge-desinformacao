import numpy as np
from fakenews.processing.dataloader import load_datasets
import spacy
import torch
import joblib
import mlflow
import mlflow.sklearn
from fakenews.core.features import get_stylometric_features, get_bert_embeddings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from fakenews.core.config import (
    BERT_MODEL_NAME, MAX_SEQUENCE_LENGTH, BATCH_SIZE,
    RF_MODEL_PATH, DATA_DIR, SENSATIONALIST_WORDS,
    SCORE_EXCLAMACAO_MULT, SCORE_SENSACIONAL_MULT, RF_N_ESTIMATORS, RANDOM_STATE
)

# Feature extraction is now centralized in fakenews.core.features
# Data loading is now centralized in fakenews.processing.dataloader

def main():
    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
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
        mlflow.sklearn.log_model(clf, "random_forest_model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        print("Model saved successfully and logged to MLflow!")

if __name__ == "__main__":
    main()
