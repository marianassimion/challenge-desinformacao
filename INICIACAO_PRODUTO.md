# Iniciação do Produto: Sistema Anti-Desinformação

## 1. Definição do Problema de Negócio
A desinformação digital propaga-se rapidamente através de redes sociais e aplicativos de mensageria (WhatsApp/Telegram), onde a curadoria editorial é baixa. Com base em pesquisa com stakeholders, identificou-se que:
- **73%** da amostra já compartilhou ou possivelmente compartilhou notícias falsas.
- Os usuários reconhecem sinais de fake news (como títulos sensacionalistas), mas falham em aplicar a checagem antes do compartilhamento.
- Existe uma demanda alta (**93% de aceitação**) por alertas automáticos que previnam o compartilhamento impulsivo.

**Objetivo:** Criar uma ferramenta de assistência que identifique sinais de desinformação em tempo real e forneça a explicação do motivo da classificação, promovendo a literacia digital do usuário.

## 2. Tradução para Problema de Machine Learning
O problema foi modelado como uma **Classificação Binária Supervisionada**.
- **Input (X):** Características linguísticas e emocionais do texto (features).
- **Output (y):** Classe da notícia $\rightarrow$ `0: Verdadeira` ou `1: Falsa`.

## 3. Justificativa das Escolhas Técnicas
Para a implementação do MVP, foram escolhidas as seguintes tecnologias:

| Categoria | Escolha Técnica | Justificativa |
| :--- | :--- | :--- |
| **Linguagem** | Python 3 | Ecossistema líder para Data Science e ML. |
| **NLP/Processamento** | SpaCy | Alta performance para análise de classes gramaticais em Português. |
| **Algoritmo** | Random Forest | Robusto para datasets tabulares, evita overfitting e fornece a "Importância das Features" (explicabilidade). |
| **Dataset** | Fake.br-Corpus | Dataset referenciado e robusto para notícias em português. |
| **Persistência** | Joblib | Permite exportar o modelo treinado para deploy rápido sem re-treinamento. |

## 4. Arquitetura Geral do Fluxo
`Texto da Notícia` $\rightarrow$ `Pipeline de NLP (SpaCy)` $\rightarrow$ `Cálculo de Score Emocional` $\rightarrow$ `Modelo Random Forest` $\rightarrow$ `Veredito + Explicação`.
