
import os
import uvicorn
import httpx

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
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

async def responder_mensagem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        texto_usuario = update.message.text

        if not texto_usuario:
            await update.message.reply_text("Por favor, envie um texto ou um link válido.")
            return

        # Guarda a mensagem original para editá-la depois, dando sensação de rapidez
        msg_espera = await update.message.reply_text("🔍 Analisando a notícia... (Isso pode levar alguns segundos)")

        # Se for URL
        if texto_usuario.startswith("http://") or texto_usuario.startswith("https://"):
            payload = {"url": texto_usuario}
        else:
            # Se for texto
            if len(texto_usuario) < 40:
                await msg_espera.edit_text(
                    "❌ O texto é muito curto. Envie uma notícia com pelo menos 40 caracteres."
                )
                return
            payload = {"text": texto_usuario}

        # RESOLUÇÃO DO GARGALO: Chamada HTTP Assíncrona não bloqueia o bot
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(API_URL, json=payload)

        # Erro da API
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

        # RESOLUÇÃO DA AMBIGUIDADE: Lógica de Explicação
        if prediction == "fake":
            classificacao = "🚨 **POSSÍVEL DESINFORMAÇÃO**"
            
            # Se a IA Semântica achou fake, mas o texto é neutro/formal
            if carga_emocional < 4.0:
                explicacao = (
                    "🧠 *Por que é falso se parece formal?*\n"
                    "Apesar da linguagem neutra e sem exageros, nossa IA Semântica (BERT) "
                    "identificou fortes padrões de manipulação de contexto e desinformação nesta narrativa."
                )
            else:
                explicacao = (
                    "🧠 *Por que é falso?*\n"
                    "Nossa IA Semântica identificou desinformação, e o Raio-X abaixo "
                    "confirma o uso de gatilhos emocionais e linguagem sensacionalista."
                )
        else:
            classificacao = "✅ **PROVAVELMENTE VERDADEIRA**"
            explicacao = (
                "🧠 *Por que é verdadeira?*\n"
                "O texto possui uma estrutura consistente e nossa IA Semântica "
                "não detectou padrões associados a narrativas manipuladas."
            )

        # Formatação limpa e explicativa
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
            f"Reflita antes de compartilhar e busque fontes confiáveis!"
        )

        await msg_espera.edit_text(resposta, parse_mode="Markdown")

    except httpx.TimeoutException:
        await msg_espera.edit_text("⏳ A análise demorou demais. O servidor pode estar sobrecarregado.")

    except httpx.RequestError as e:
        print(f"Erro de conexão com a API: {e}")
        await msg_espera.edit_text("❌ Não consegui conectar à API de Inteligência Artificial.")

    except Exception as e:
        print(f"Erro interno: {e}")
        # Tratamento seguro caso a mensagem não possa ser editada (ex: foi apagada)
        try:
            await msg_espera.edit_text("❌ Ocorreu um erro interno ao analisar a notícia.")
        except:
            await update.message.reply_text("❌ Ocorreu um erro interno ao analisar a notícia.")


# Configuração do bot
app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder_mensagem))

@asynccontextmanager
async def lifespan(web_app: FastAPI):
    await app.initialize()
    await app.start()

    webhook_url = os.getenv("WEBHOOK_URL")

    if webhook_url:
        await app.bot.set_webhook(url=webhook_url)
        print(f"🤖 Bot iniciado!")
        print(f"🌐 Webhook: {webhook_url}")
    else:
        print("⚠️ WEBHOOK_URL não configurada.")

    yield

    await app.stop()
    await app.shutdown()


web_app = FastAPI(lifespan=lifespan)

@web_app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, app.bot)
    await app.process_update(update)
    return {"ok": True}

@web_app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8080))
    uvicorn.run(web_app, host="0.0.0.0", port=PORT)