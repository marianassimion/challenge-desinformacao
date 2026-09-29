import spacy
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# --- CONFIGURAÇÕES ---
TOKEN = "8800042993:AAGCfClabRLOkrd3JwQkuvxQb8qSHuDznuQ"
BERT_MODEL = "models/bert"
MAX_LEN = 512

print("Iniciando carregamento do sistema e modelos...")
try:
    nlp = spacy.load("pt_core_news_sm")
except:
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "pt_core_news_sm"])
    nlp = spacy.load("pt_core_news_sm")

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
bert = AutoModel.from_pretrained(BERT_MODEL)
bert.eval()
modelo_xgb = joblib.load('models/xgb_model.joblib')
print("Modelos carregados com sucesso!")

def analisar_noticia(texto):
    doc = nlp(texto[:5000])
    tamanho = len(doc) if len(doc) > 0 else 1
    verbos = sum(1 for token in doc if token.pos_ == "VERB") / tamanho * 100
    adjetivos = sum(1 for token in doc if token.pos_ == "ADJ") / tamanho * 100
    pronomes = sum(1 for token in doc if token.pos_ == "PRON") / tamanho * 100

    palavras_sensacionalistas = [
        "urgente", "chocante", "bomba", "escândalo", "revelado", "segredo",
        "não vão acreditar", "atenção", "alerta", "exclusivo", "inacreditável",
        "impressionante", "cuidado", "compartilhe", "antes que apaguem"
    ]
    text_lower = texto.lower()
    n_exclamacao = texto.count("!")
    n_sensacional = sum(text_lower.count(p) for p in palavras_sensacionalistas)
    n_maiusculas = sum(1 for p in texto.split() if p.isupper() and len(p) > 1)
    palavras = max(len(texto.split()), 1)
    score_emocional = min(round((n_exclamacao * 1.5 + n_sensacional * 3 + n_maiusculas) / palavras * 100, 2), 10)
    
    stylometry = np.array([[verbos, adjetivos, pronomes, score_emocional]])

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    with torch.no_grad():
        outputs = bert(**inputs)
    bert_emb = outputs.last_hidden_state[:, 0, :].numpy()

    X = np.hstack([bert_emb, stylometry])
    risco = modelo_xgb.predict_proba(X)[0][1] * 100
    
    return risco, score_emocional, adjetivos

# --- HANDLERS DO TELEGRAM (v13.x Síncrono) ---
def start(update, context):
    update.message.reply_text(
        "👋 Olá! Eu sou o Detector de Fake News.\n\n"
        "Me envie o texto da notícia que você quer analisar e eu calcularei o risco de desinformação."
    )

def responder_mensagem(update, context):
    try:
        texto_usuario = update.message.text
        if not texto_usuario:
            update.message.reply_text("Por favor, envie o texto da notícia que você quer analisar.")
            return

        if len(texto_usuario) < 40:
            update.message.reply_text("O texto é muito curto. Envie a notícia completa para uma análise precisa.")
            return

        update.message.reply_text("🔍 Analisando a notícia no motor híbrido...")
        
        risco, score_emocional_val, perc_adjetivos = analisar_noticia(texto_usuario)
        
        resposta = (
            f"📊 **Matriz de Confiança**\n"
            f"Risco de Desinformação: {risco:.1f}%\n\n"
            f"⚠️ **Sinais Encontrados:**\n"
            f"- Carga Emocional Semântica: Nível {score_emocional_val}/10\n"
            f"- Uso de Adjetivos: {perc_adjetivos:.1f}% do texto\n\n"
            f"Reflita antes de compartilhar e busque fontes confiáveis!"
        )
        update.message.reply_text(resposta, parse_mode='Markdown')

    except Exception as e:
        print(f"Erro interno detectado: {e}")
        update.message.reply_text("Desculpe, ocorreu um erro interno ao processar a notícia. Tente novamente.")

if __name__ == "__main__":
    print("O Bot está online! Abra o Telegram e envie /start.")
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, responder_mensagem))
    
    updater.start_polling()
    updater.idle()