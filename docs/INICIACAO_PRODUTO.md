# Iniciacao do Produto: Sistema Anti-Desinformacao

## 1. Definicao do Problema de Negocio
A desinformacao digital propaga-se rapidamente atraves de redes sociais e aplicativos de mensageria. Identificou-se que 73% da amostra ja compartilhou fake news.
**Objetivo:** Criar uma ferramenta de assistencia que identifique sinais de desinformacao em tempo real e forneca a explicacao do motivo da classificacao.

## 2. Traducao para Problema de Machine Learning
O problema foi modelado como uma Classificacao Binaria Supervisionada.
- **Input (X):** Caracteristicas linguisticas, semanticas (BERT) e emocionais do texto.
- **Output (y):** Classe da noticia $\rightarrow$ 0: Verdadeira ou 1: Falsa.

## 3. Justificativa das Escolhas Tecnicas

| Categoria | Escolha Tecnica | Justificativa |
| :--- | :--- | :--- |
| **Linguagem** | Python 3.14+ | Ecossistema lider para Data Science e ML, rodando nativamente. |
| **NLP Semantico** | BERTimbau (512 tokens)| Captura o sentido contextual do texto em profundidade. |
| **Algoritmo** | HistGradientBoosting | Robusto para datasets desbalanceados, nao exige dependencias C++ (libomp), permitindo execucao perfeita no chip M4 e Cloud Run. |
| **Deploy** | Google Cloud Run | Arquitetura Serverless, escalabilidade instantanea, e suporte a conteineres pesados. |
| **Persistencia** | xgb_model.joblib | Permite exportar o modelo preditivo para inferencia em milissegundos. |

## 4. Arquitetura Geral do Fluxo
URL ou Texto $\rightarrow$ Scraper (Trafilatura) $\rightarrow$ BERTimbau + SpaCy $\rightarrow$ Modelo Hibrido (HistGradientBoosting) $\rightarrow$ Telegram Bot Veredito.