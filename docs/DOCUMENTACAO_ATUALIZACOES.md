# Relatorio de Atualizacoes - Fake News Detector API

Este documento detalha todas as melhorias arquiteturais, correcoes de bugs e novas funcionalidades implementadas no projeto para garantir escalabilidade, manutenibilidade e estabilidade.

---

## Atualizacoes Mais Recentes (Deploy & Cloud)

### 1. Atualizacao Critica de Seguranca (Token e Credenciais)
- **Sintoma:** Conflitos e sequestro de Token publico no GitHub.
- **Acao:** O token vazado foi revogado via BotFather.
- **Solucao Implementada:** Migracao de credenciais hardcoded para Variaveis de Ambiente. A biblioteca python-dotenv foi introduzida e o codigo agora consome o token de um ficheiro invisivel (.env) bloqueado pelo .gitignore. No deploy em nuvem, o token e injetado de forma segura via Secret Manager.

### 2. Bypass de Arquitetura e Performance (Chip Apple M4)
- **Sintoma:** Falha na instalacao de dependencias C++ (libomp) no macOS restrito, inviabilizando o treino e execucao com XGBoost localmente.
- **Solucao:** Substituicao da engine preditiva para HistGradientBoostingClassifier do scikit-learn.
- **Resultado:** Execucao 100% nativa utilizando a aceleracao MPS da Neural Engine da Apple, mantendo a precisao exata em 93% e viabilizando o deploy imediato em ambientes serverless como o Google Cloud Run sem erros de compilacao C++.

### 3. Deploy em Producao (Google Cloud Run)
- O sistema foi containerizado e promovido a aplicacao Serverless rodando 24/7 na plataforma Google Cloud Run.
- O frontend principal passou a ser o Bot do Telegram, agora equipado com extracao assincrona de URLs via trafilatura.

---

## Atualizacoes Anteriores (Refatoracao de Base)

### 1. Refatoracao de Arquitetura (Clean Code)
O projeto foi migrado de um modelo de "script unico" para uma Arquitetura em Camadas, seguindo os principios de Responsabilidade Unica (SRP).
- **src/fakenews/core/model_manager.py**: Implementacao do padrao Singleton. Agora, os modelos (BERT, Algoritmo ML e Spacy) sao carregados apenas uma vez na memoria e compartilhados por toda a aplicacao, eliminando o uso de variaveis globais e reduzindo o consumo de RAM.
- **src/fakenews/core/features.py**: Criacao da classe FeatureExtractor. Toda a logica matematica de extracao de embeddings do BERT e calculos de estilometria foi movida para ca, isolando a logica de Ciencia de Dados da logica de API.
- **src/fakenews/api/app.py**: A API foi simplificada para atuar apenas como a camada de interface. Ela agora orquestra as chamadas ao ModelManager e FeatureExtractor, sem precisar conhecer os detalhes internos do modelo.

### 2. Correcoes de Bugs e Estabilidade
- **Problema de Importacao (ModuleNotFoundError)**: Erros ao importar fakenews.processing.scraper. Solucao: Implementacao de configuracao de caminho absoluto no topo do app.py, garantindo que o Python localize a raiz do projeto independentemente de como o servidor seja iniciado.
- **Erro de Processamento Assincrono (AttributeError)**: A funcao extract_text_from_url era assincrona (async), mas estava sendo chamada sem a palavra-chave await. Solucao: Implementacao correta do await na chamada do scraper, garantindo que o texto seja totalmente baixado antes de iniciar a validacao.
- **Validacao de JSON no Swagger**: Erro 422 pois o padrao JSON nao aceita quebras de linha literais dentro de strings. Solucao: Ajuste na estrutura de requisicao para suportar campos opcionais e melhoria na compatibilidade de recebimento de dados.

### 3. Novas Funcionalidades Antigas
- **Medicao de Latencia (Performance Monitoring)**: Foi implementado um Middleware de Latencia no FastAPI. O sistema agora captura o tempo exato de inicio e fim de cada requisicao e exibe no cabecalho HTTP da resposta como X-Process-Time.
- **Estabilizacao de URLs**: A opcao de analise via link foi totalmente restaurada e integrada a nova arquitetura, permitindo que o utilizador escolha entre enviar o texto bruto ou uma URL do G1/outros portais.

### 4. Guia de Execucao Atualizado (API)
Para garantir que todas as dependencias e caminhos sejam resolvidos corretamente, o backend HTTP deve ser iniciado da seguinte forma:

```bash
# 1. Definir o caminho dos modulos
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 2. Iniciar a API como modulo do Python
python3 -m fakenews.api.app
```

---
**Status Final:** Estavel | Em Producao (Live) | Hospedado no Google Cloud Run