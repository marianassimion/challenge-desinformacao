import nest_asyncio
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import Application, CommandHandler, MessageHandler, filters, InlineQueryHandler, ContextTypes
import logging

# Importando a lógica do modelo do arquivo refatorado
from model_utils import prever_risco_desinformacao

# Habilita a execução do bot em ambientes como Colab/Jupyter
nest_asyncio.apply()

# Configuração de Logging para monitorar o bot no terminal
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

# Token oficial do BotFather
TOKEN = "8800042993:AAGCfClabRLOkrd3JwQkuvxQb8qSHuDznuQ"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start do bot"""
    await update.message.reply_text(
        "Olá! Eu sou o Detetor de Fake News. 🔍\n\n"
        "Envie-me qualquer notícia ou manchete e eu analisarei a probabilidade de desinformação usando IA."
    )

async def responder_mensagem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa mensagens de texto enviadas diretamente ao bot"""
    texto_usuario = update.message.text
    await update.message.reply_text("🔍 Analisando a notícia no motor híbrido... Aguarde um instante.")

    # Chamada para a função real de predição (IA REAL)
    resultado = prever_risco_desinformacao(texto_usuario)

    if resultado["sucesso"]:
        risco = resultado["risco"]
        score = resultado["score_emocional"]
        adjetivos = resultado["perc_adjetivos"]

        # Formatação da resposta baseada no nível de risco
        alerta = "🔴 ALTO RISCO" if risco > 70 else "🟡 RISCO MODERADO" if risco > 40 else "🟢 BAIXO RISCO"

        resposta = (
            f"📊 **Análise de Confiança**\n"
            f"Veredito: {alerta}\n"
            f"Probabilidade de Fake News: {risco}%\n\n"
            f"⚠️ **Indicadores Técnicos:**\n"
            f"- Carga Emocional: {score}/10\n"
            f"- Densidade de Adjetivos: {adjetivos}%\n\n"
            f"💡 *Dica: Notícias com alta carga emocional e muitos adjetivos tendem a ser sensacionalistas. Reflita antes de compartilhar!*"
        )
    else:
        resposta = "❌ Desculpe, ocorreu um erro técnico ao analisar esse texto. Tente novamente mais tarde."

    await update.message.reply_text(resposta, parse_mode='Markdown')

async def inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Permite analisar notícias em qualquer chat usando @nome_do_bot texto"""
    query = update.inline_query.query
    if not query:
        return

    resultado = prever_risco_desinformacao(query)

    if resultado["sucesso"]:
        risco = resultado["risco"]
        emoji = "🔴" if risco > 70 else "🟡" if risco > 40 else "🟢"
        resultado_texto = f"{emoji} Esta notícia tem {risco}% de risco de ser Fake News. Pense criticamente antes de repassar."
    else:
        resultado_texto = "⚠️ Erro ao analisar a notícia."

    resultados = [
        InlineQueryResultArticle(
            id="fake-check-1",
            title=f"Verificar Risco: {resultado.get('risco', 'Erro')}%",
            input_message_content=InputTextMessageContent(resultado_texto)
        )
    ]
    await update.inline_query.answer(resultados)

if __name__ == "__main__":
    print("🚀 Iniciando Bot de Detecção de Fake News...")
    print("O Bot está online! Abra o Telegram e envie uma mensagem para ele.")

    app = Application.builder().token(TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder_mensagem))
    app.add_handler(InlineQueryHandler(inline_query))

    app.run_polling()
