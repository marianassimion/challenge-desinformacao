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

        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            self.bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
            self.bert_model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Não foi possível carregar o modelo BERT '{BERT_MODEL_NAME}'. "
                "Verifique sua conexão com a internet (o modelo é baixado do "
                f"Hugging Face na primeira execução) e o nome configurado em "
                f"fakenews.core.config.BERT_MODEL_NAME. Erro original: {e}"
            )

        try:
            self.rf_classifier = joblib.load(RF_MODEL_PATH)
            print("Resources loaded successfully!")
        except Exception as e:
            print(f"Error: Model file {RF_MODEL_PATH} could not be loaded: {e}")
            self.rf_classifier = None
