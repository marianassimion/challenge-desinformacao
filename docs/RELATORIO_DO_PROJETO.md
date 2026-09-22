# Relatório de Evolução do Projeto: Detector de Fake News Híbrido

## 📝 Resumo Executivo
Este documento detalha a jornada técnica de transformação de um script experimental de análise de dados em um sistema profissional de detecção de desinformação. O projeto evoluiu de uma análise estatística simples para uma arquitetura de IA híbrida, combinando processamento de linguagem natural (NLP) de última geração com análise estilométrica.

---

## 📈 1. A Jornada de Evolução do Modelo

O desenvolvimento foi guiado por experimentos rigorosos, onde cada falha gerou um insight fundamental para a arquitetura final.

### Passo 1: O Baseline Inicial (Estilometria Simples)
*   **Abordagem:** Uso do dataset `Fake.br-Corpus` analisando apenas a "forma" da escrita (maiúsculas, exclamações e classes gramaticais).
*   **Resultado:** **74% de Acurácia**.
*   **Insight:** O modelo funcionava bem em um ambiente controlado, mas era vulnerável a mudanças de estilo.

### Passo 2: O Teste de Estresse (Expansão de Dados)
*   **Abordagem:** Unificação de três bases de dados (`Fake.br`, `FakeRecogna`, `FACTCK.BR`) para testar a generalização.
*   **Resultado:** **66% de Acurácia**.
*   **Insight:** A queda na precisão provou que "estilo" não é suficiente. Notícias falsas podem ser escritas de forma profissional. Era necessário entender o **sentido** (semântica) do texto.

### Passo 3: A Solução Híbrida (Semântica + Estilo)
*   **Abordagem:** Implementação do modelo **BERTimbau** (BERT para Português) para extração de embeddings semânticos, fundidos com a análise gramatical do **Spacy**.
*   **Resultado:** **87% de Acurácia**.
*   **Insight:** A união da semântica (o "quê") com a estilometria (o "como") criou um modelo robusto e generalista.

---

## 🛠️ 2. Arquitetura Técnica Atual

O sistema foi transformado de um notebook (`.ipynb`) para uma aplicação de software modular:

### Componentes do Sistema:
1.  **`hybrid_model.py` (A Engine):** 
    *   Realiza a unificação de datasets.
    *   Processa a gramática via Spacy.
    *   Gera vetores semânticos via BERTimbau.
    *   Treina a Random Forest final e exporta o modelo.
2.  **`rf_model.joblib` (O Cérebro):** 
    *   Arquivo binário que contém a IA treinada. Permite que a predição seja instantânea sem a necessidade de retreinar o modelo a cada requisição.
3.  **`app.py` (A Interface de Serviço):** 
    *   API construída com **FastAPI**.
    *   Oferece um endpoint `/predict` que recebe o texto e devolve a classificação com o nível de confiança e métricas detalhadas.
4.  **`requirements.txt`:** 
    *   Lista de todas as dependências (`transformers`, `torch`, `fastapi`, etc.) para garantir a reprodutibilidade do ambiente.

---

## 🚀 3. Próximos Passos (Roadmap do Produto)

O projeto agora sai da fase de "Pesquisa" e entra na fase de "Produto". As próximas metas são:

### 🟢 Curto Prazo: Operacionalização
*   Implementação de um Front-end simples para interação do usuário.
*   Deploy da API em ambiente de nuvem (Cloud).

### 🔵 Médio Prazo: Memória e Aprendizado
*   **Integração de Banco de Dados:** Armazenamento de cada notícia analisada.
*   **Ciclo de Feedback:** Permitir que usuários humanos corrijam a IA, gerando dados rotulados para retreinamento.

### 🔴 Longo Prazo: Escala e Refinamento
*   Implementação de monitoramento de *drift* (quando o estilo das fake news muda com o tempo).
*   Expansão para detecção de a multiclasse (ex: "Sátira", "Enganoso", "Falso", "Verdadeiro").

---
**Responsável Técnico:** Claude Code / Arquiteto de Software
**Status Atual:** Modelo Treinado e API Implementada.
