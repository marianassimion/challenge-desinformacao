# Guia de Testes: Detector de Fake News Hibrido

Este guia fornece instrucoes passo a passo para que qualquer membro da equipa possa configurar o ambiente e testar a eficacia do modelo de deteccao de desinformacao, seja em producao ou localmente.

---

## 1. Teste Rapido em Producao (Live - Recomendado)

O sistema ja esta em producao na nuvem e pode ser testado diretamente pelo utilizador final sem necessidade de configuracoes ou instalacao de dependencias. A nossa interface principal e o Bot do Telegram hospedado no Google Cloud Run.

1. **Aceda ao Bot:** Abra o Telegram e procure pelo @ do bot oficial do projeto ou clique no link de convite.
2. **Inicie a interacao:** Envie o comando /start para o bot.
3. **Como testar:**
   - **Teste de Web Scraping:** Copie e cole a URL de uma noticia (ex: G1, UOL, CNN). O bot extraira o texto automaticamente.
   - **Teste de Texto Bruto:** Cole o texto completo de uma mensagem suspeita do WhatsApp.
4. **Analise a resposta:** O bot processara o texto usando o modelo hibrido de IA (BERT + Estilometria) e retornara se e Fake ou True, o nivel de confianca e o detalhamento das metricas.

---

## 2. Testes para Desenvolvedores (Ambiente Local)

Se precisa de correr o codigo localmente para desenvolvimento ou validar o modelo bruto, siga as instrucoes abaixo.

### Configuracao do Ambiente

```bash
# 1. Clone o repositorio
git clone <url-do-repositorio>
cd challenge-desinformacao

# 2. Instale as dependencias
pip install -r requirements.txt

# 3. Baixe o modelo de linguagem do Spacy para Portugues
python3 -m spacy download pt_core_news_sm
```

### Configuracao de Credenciais de Seguranca
Como o token foi retirado do codigo para evitar fugas de dados, precisara de um arquivo de variaveis de ambiente:
1. Na raiz do projeto, crie um arquivo chamado `.env`.
2. Adicione o seu token de testes do Telegram (gerado no BotFather):

```env
TELEGRAM_TOKEN="SEU_TOKEN_DE_TESTE_AQUI"
```
*(Nota: O arquivo .env ja esta no .gitignore e nunca deve ser commitado).*

### Metodo A: Teste via Telegram Bot (Local)
Para testar alteracoes no bot usando a sua maquina (com suporte a aceleracao Apple M4 MPS ou GPU):

```bash
python3 src/bot.py
```

### Metodo B: Teste via API (FastAPI)
Para testar via requisicoes HTTP (Swagger):
1. **Inicie o servidor:**

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 -m fakenews.api.app
```
2. **Aceda a interface visual:** Abra o navegador em `http://localhost:8000/docs` e faca um POST na rota `/predict`.

---

## 3. Como Interpretar os Resultados

| Resultado | Significado | O que observar |
| :--- | :--- | :--- |
| **Fake** | A IA detetou padroes de desinformacao. | Verifique se o score_emocional esta alto (muitas exclamacoes, caps lock). |
| **True** | A IA considerou a noticia legitima. | Verifique se o texto possui uma estrutura gramatical mais neutra e formal. |
| **Confidence** | Nivel de certeza da IA (0.0 a 1.0). | Valores abaixo de 0.7 indicam que a IA esta em duvida. |