import spacy
import joblib
from transformers import AutoTokenizer, AutoModel
from fakenews.core.config import BERT_MODEL_NAME, RF_MODEL_PATH

class ModelManager:
    """
    Singleton class to manage the loading and access of ML models.
    Prevents reloading models on every request and avoids global variables.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.nlp = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.rf_classifier = None
        self._load_resources()
        self._initialized = True

    def _load_resources(self):
        print("Loading ML resources...")
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except OSError:
            raise RuntimeError(
                "Modelo Spacy 'pt_core_news_sm' não encontrado. Instale-o antes de "
                "iniciar a API: python -m spacy download pt_core_news_sm "
                "(veja o README, seção de instalação)."
            )

        self.bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        self.bert_model.eval()

        try:
            self.rf_classifier = joblib.load(RF_MODEL_PATH)
            print("Resources loaded successfully!")
        except FileNotFoundError:
            print(f"Error: Model file {RF_MODEL_PATH} not found.")
