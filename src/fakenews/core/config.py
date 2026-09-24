import os
from pathlib import Path

# Base Paths
# Path relative to this file: src/fakenews/core/config.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Model Paths
RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"

# BERT Hyperparameters
BERT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_SEQUENCE_LENGTH = 128
BATCH_SIZE = 32

# Random Forest Hyperparameters
RF_N_ESTIMATORS = 200
RANDOM_STATE = 42

# Stylometric Constants (Consolidated from across the project)
SENSATIONALIST_WORDS = [
    "urgente", "bomba", "revelado", "cuidado", "alerta",
    "choque", "absurdo", "escândalo", "mundo", "agora",
    "chocante", "segredo", "não vão acreditar", "atenção",
    "exclusivo", "inacreditável", "impressionante", "compartilhe",
    "antes que apaguem"
]
SCORE_EXCLAMACAO_MULT = 1.5
SCORE_SENSACIONAL_MULT = 3.0

# Validation
MIN_TEXT_LENGTH = 40
MIN_INFO_TOKENS = 3
