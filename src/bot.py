import os
import httpx
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

from fakenews.processing.fact_checker import buscar_fontes_confiaveis

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
API_URL = "https://informio-api-500797299406.southamerica-east1.run.app/predict"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Olá! Eu sou o Detector de Fake News.\n\n"
        "Me envie o texto de uma notícia ou apenas o link (URL) "
        "e eu analisarei o risco cruzando IA com portais de notícias."
    )

async def responder_mensagem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        texto_usuario = update.message.text

        if not texto_usuario:
            await update.message.reply_text("Por favor, envie um texto ou um link válido.")
            return

        msg_espera = await update.message.reply_text("🔍 Analisando a estrutura da notícia...")

        if texto_usuario.startswith("http://") or texto_usuario.startswith("https://"):
            payload = {"url": texto_usuario}
        else:
            if len(texto_usuario) < 40:
                await msg_espera.edit_text("❌ O texto é muito curto. Envie uma notícia com pelo menos 40 caracteres.")
                return
            payload = {"text": texto_usuario}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(API_URL, json=payload)

        if response.status_code != 200:
            try:
                erro = response.json()
            except Exception:
                erro = response.text
            await msg_espera.edit_text(f"❌ Erro ao analisar a notícia.\n\n{erro}")
            return

        resultado = response.json()
        prediction = resultado["prediction"]
        confidence = resultado["confidence"]
        stylometry = resultado["metrics"]["stylometry"]
        confianca = confidence * 100
        carga_emocional = stylometry['score_emocional']

        # FUSÃO DE IA SEMÂNTICA + WEB SEARCH RSS
        if prediction == "fake":
            classificacao = "🚨 **POSSÍVEL DESINFORMAÇÃO**"
            explicacao = (
                "🧠 *Por que é suspeito?*\n"
                "Nossa Inteligência Artificial identificou padrões narrativos "
                "frequentemente associados a manipulação de contexto ou desinformação."
            )
        else:
            await msg_espera.edit_text("✅ Estrutura legítima. Buscando no G1, UOL, CNN, Estadão e BBC...")
            fontes = buscar_fontes_confiaveis(texto_usuario)
            
            if fontes:
                classificacao = "✅ **NOTÍCIA VERIFICADA NA WEB**"
                links_formatados = "\n".join([f"🔗 [{f['titulo']}]({f['link']})" for f in fontes])
                explicacao = (
                    "🧠 *Confirmação Factual:*\n"
                    "A IA validou a estrutura do texto e nós encontramos "
                    "esta narrativa sendo relatada nos principais portais jornalísticos.\n\n"
                    f"📰 **Fontes Encontradas:**\n{links_formatados}"
                )
            else:
                classificacao = "🟡 **ESTRUTURA VERDADEIRA (SEM FONTES EXTERNAS)**"
                explicacao = (
                    "🧠 *Aviso de Cautela:*\n"
                    "O texto está muito bem escrito e parece autêntico para a nossa Inteligência Artificial. "
                    "No entanto, **não encontramos reportagens sobre isso nos 5 principais portais agora**.\n\n"
                    "Pode ser um evento muito recente, um artigo de opinião local, ou um texto manipulado com boa gramática."
                )

        resposta = (
            f"📊 **Resultado da Análise**\n\n"
            f"{classificacao}\n"
            f"🎯 Confiança da IA: {confianca:.1f}%\n\n"
            f"{explicacao}\n\n"
            f"⚠️ **Raio-X Estilométrico:**\n"
            f"- Carga emocional: {carga_emocional:.1f}/10\n"
            f"- Verbos: {stylometry['perc_verbos']:.1f}%\n"
            f"- Adjetivos: {stylometry['perc_adjetivos']:.1f}%\n"
            f"- Pronomes: {stylometry['perc_pronomes']:.1f}%\n\n"
            f"Reflita sempre antes de partilhar!"
        )

        await msg_espera.edit_text(resposta, parse_mode="Markdown")

    except httpx.TimeoutException:
        await msg_espera.edit_text("⏳ A análise demorou demais. O servidor pode estar sobrecarregado.")
    except Exception as e:
        logging.error(f"Erro interno: {e}")
        try:
            await msg_espera.edit_text("❌ Ocorreu um erro interno ao analisar a notícia.")
        except:
            await update.message.reply_text("❌ Ocorreu um erro interno ao analisar a notícia.")

def main():
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    print("🚀 Iniciando o Bot (Com Fact-Checking via Google News)...")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder_mensagem))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()