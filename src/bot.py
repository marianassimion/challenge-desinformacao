import os
import spacy
import numpy as np
import torch
import joblib
import trafilatura
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


load_dotenv()  
TOKEN = os.getenv("TELEGRAM_TOKEN")

BERT_MODEL = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 512

print("Iniciando carregamento do sistema e detectando hardware...")


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Aceleração de Hardware Ativada: Apple M4 (MPS)")
else:
    device = torch.device("cpu")
    print("Rodando na CPU.")

try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "pt_core_news_sm"])
    nlp = spacy.load("pt_core_news_sm")

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
# Envia o modelo BERT para a GPU do Mac
bert = AutoModel.from_pretrained(BERT_MODEL).to(device)
bert.eval()

# Carrega o seu XGBoost treinado
modelo_xgb = joblib.load('models/xgb_model.joblib')
print("Modelos carregados com sucesso!")

def extrair_texto_de_url(url):
    """Acessa o site e extrai apenas o corpo da notícia."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        return trafilatura.extract(downloaded)
    return None

def analisar_noticia(texto):
    doc = nlp(texto[:5000])
    tamanho = max(len(doc), 1)
    verbos = sum(1 for token in doc if token.pos_ == "VERB") / tamanho * 100
    adjetivos = sum(1 for token in doc if token.pos_ == "ADJ") / tamanho * 100
    pronomes = sum(1 for token in doc if token.pos_ == "PRON") / tamanho * 100

    palavras_sensacionalistas = [
        "urgente", "chocante", "bomba", "escândalo", "revelado", "segredo",
        "atenção", "alerta", "exclusivo", "inacreditável", "compartilhe"
    ]
    text_lower = texto.lower()
    n_exclamacao = texto.count("!")
    n_sensacional = sum(text_lower.count(p) for p in palavras_sensacionalistas)
    n_maiusculas = sum(1 for p in texto.split() if p.isupper() and len(p) > 1)
    palavras = max(len(texto.split()), 1)
    score_emocional = min(round((n_exclamacao * 1.5 + n_sensacional * 3 + n_maiusculas) / palavras * 100, 2), 10)
    
    stylometry = np.array([[verbos, adjetivos, pronomes, score_emocional]])

    # Tokeniza e envia os dados para a GPU do Mac
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert(**inputs)
    
    # Traz o resultado da GPU de volta para a CPU para o XGBoost
    bert_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    X = np.hstack([bert_emb, stylometry])
    risco = modelo_xgb.predict_proba(X)[0][1] * 100
    
    return risco, score_emocional, adjetivos


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Olá! Eu sou o Detector de Fake News.\n\n"
        "Me envie o **texto** de uma notícia ou apenas o **link (URL)** e eu analisarei o risco."
    )

async def responder_mensagem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        texto_usuario = update.message.text
        if not texto_usuario:
            await update.message.reply_text("Por favor, envie texto ou um link válido.")
            return

        # Detecção de URL
        if texto_usuario.startswith("http://") or texto_usuario.startswith("https://"):
            await update.message.reply_text("🔗 Link detectado! Lendo a página...")
            texto_extraido = extrair_texto_de_url(texto_usuario)
            
            if not texto_extraido or len(texto_extraido) < 40:
                await update.message.reply_text("❌ Bloqueio anti-bot do site ou texto indisponível. Cole o texto manualmente.")
                return
            
            texto_usuario = texto_extraido

        if len(texto_usuario) < 40:
            await update.message.reply_text("O texto é muito curto. Envie a notícia completa.")
            return

        await update.message.reply_text("🔍 Processando na Neural Engine do Mac...")
        
        risco, score_emocional_val, perc_adjetivos = analisar_noticia(texto_usuario)
        
        resposta = (
            f"📊 **Matriz de Confiança**\n"
            f"Risco de Desinformação: {risco:.1f}%\n\n"
            f"⚠️ **Sinais Encontrados:**\n"
            f"- Carga Emocional Semântica: Nível {score_emocional_val}/10\n"
            f"- Uso de Adjetivos: {perc_adjetivos:.1f}% do texto\n\n"
            f"Reflita antes de compartilhar e busque fontes confiáveis!"
        )
        await update.message.reply_text(resposta, parse_mode='Markdown')

    except Exception as e:
        print(f"Erro interno detectado: {e}")
        await update.message.reply_text("Desculpe, ocorreu um erro interno. Tente colar o texto em vez do link.")

if __name__ == "__main__":
    print("O Bot está online! Abra o Telegram e envie /start.")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder_mensagem))
    app.run_polling()