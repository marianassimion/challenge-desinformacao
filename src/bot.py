import os
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")

API_URL = "https://informio-api-500797299406.southamerica-east1.run.app/predict"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Olá! Eu sou o Detector de Fake News.\n\n"
        "Me envie o texto de uma notícia ou apenas o link (URL) "
        "e eu analisarei o risco."
    )


async def responder_mensagem(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    try:
        texto_usuario = update.message.text

        if not texto_usuario:
            await update.message.reply_text(
                "Por favor, envie um texto ou um link válido."
            )
            return

        await update.message.reply_text(
            "🔍 Analisando a notícia..."
        )

        # Se for URL
        if texto_usuario.startswith("http://") or texto_usuario.startswith("https://"):
            payload = {
                "url": texto_usuario
            }
        else:
            # Se for texto
            if len(texto_usuario) < 40:
                await update.message.reply_text(
                    "❌ O texto é muito curto. "
                    "Envie uma notícia com pelo menos 40 caracteres."
                )
                return

            payload = {
                "text": texto_usuario
            }

        # Chama a API
        response = requests.post(
            API_URL,
            json=payload,
            timeout=120
        )

        # Erro da API
        if response.status_code != 200:
            try:
                erro = response.json()
            except Exception:
                erro = response.text

            await update.message.reply_text(
                f"❌ Erro ao analisar a notícia.\n\n"
                f"{erro}"
            )
            return

        resultado = response.json()

        prediction = resultado["prediction"]
        confidence = resultado["confidence"]

        stylometry = resultado["metrics"]["stylometry"]

        # Converte para porcentagem
        confianca = confidence * 100

        if prediction == "fake":
            classificacao = "🚨 POSSÍVEL DESINFORMAÇÃO"
        else:
            classificacao = "✅ PROVAVELMENTE VERDADEIRA"

        resposta = (
            f"📊 **Resultado da análise**\n\n"
            f"{classificacao}\n\n"
            f"🎯 Confiança: {confianca:.1f}%\n\n"
            f"⚠️ **Sinais encontrados:**\n"
            f"- Verbos: {stylometry['perc_verbos']:.1f}%\n"
            f"- Adjetivos: {stylometry['perc_adjetivos']:.1f}%\n"
            f"- Pronomes: {stylometry['perc_pronomes']:.1f}%\n"
            f"- Carga emocional: {stylometry['score_emocional']}/10\n\n"
            f"Reflita antes de compartilhar e busque fontes confiáveis!"
        )

        await update.message.reply_text(
            resposta,
            parse_mode="Markdown"
        )

    except requests.exceptions.Timeout:
        await update.message.reply_text(
            "⏳ A análise demorou demais. Tente novamente."
        )

    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão com a API: {e}")

        await update.message.reply_text(
            "❌ Não consegui conectar à API."
        )

    except Exception as e:
        print(f"Erro interno: {e}")

        await update.message.reply_text(
            "❌ Ocorreu um erro interno ao analisar a notícia."
        )


if __name__ == "__main__":
    print("🤖 Bot conectado à API!")
    
    app = Application.builder().token(TOKEN).build()

    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            responder_mensagem
        )
    )

    app.run_polling()