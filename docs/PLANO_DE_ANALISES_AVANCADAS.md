# Plano de Análises Avançadas: Explorando a Inteligência do Modelo

Este documento detalha as análises complementares que podem ser realizadas para aprofundar o entendimento sobre o comportamento do modelo de detecção de Fake News e transformar a precisão estatística em inteligência de negócio.

---

## 🎯 Objetivo Geral
Mover a análise do estágio de **Validação de Acurácia** (Quanto a IA acerta?) para o estágio de **Explicabilidade e Diagnóstico** (Por que a IA acerta ou erra?).

---

## 🔬 Propostas de Análise

### 1. Análise de Erros (Error Diagnosis)
Focar nos casos onde a IA falhou para entender as "zonas cegas" do modelo.
*   **Ação:** Isolar os Falsos Positivos (Verdadeiras marcadas como Fakes) e Falsos Negativos (Fakes marcadas como Verdadeiras).
*   **Pergunta-Chave:** *"Quais padrões comuns existem nas notícias que enganam a IA?"*
*   **Valor:** Permite a criação de regras de exceção e a coleta de dados específicos para corrigir esses erros.

### 2. Agrupamento Semântico (Topic Clustering)
Utilizar os *embeddings* do BERTimbau para mapear a distribuição temática das notícias.
*   **Ação:** Aplicar algoritmos de redução de dimensionalidade (t-SNE ou UMAP) e clusterização (K-Means) para visualizar a distribuição de notícias `fake` vs `true` em um mapa 2D.
*   **Pergunta-Chave:** *"Existem temas específicos (ex: Saúde, Política) onde a IA performa pior?"*
*   **Valor:** Identifica a necessidade de especializar o modelo em certas áreas do conhecimento.

### 3. Análise de Viés por Fonte (Source Bias Analysis)
Avaliar se o modelo é verdadeiramente generalista ou se aprendeu a identificar a "assinatura" de cada dataset.
*   **Ação:** Comparar a importância das features (`score_emocional` vs `semântica`) treinando modelos separados para cada fonte (`Fake.br`, `FakeRecogna`, `FACTCK.BR`).
*   **Pergunta-Chave:** *"O padrão de desinformação é consistente entre diferentes fontes ou cada base tem seu próprio 'jeito' de mentir?"*
*   **Valor:** Valida a robustez do modelo para enfrentar notícias de fontes totalmente novas.

### 4. Análise de Evolução Temporal (Temporal Drift)
Analisar se a "estética" das fake news mudou ao longo do tempo.
*   **Ação:** Cruzar a data da notícia com as métricas estilométricas e a precisão da IA.
*   **Pergunta-Chave:** *"As notícias falsas estão se tornando menos sensacionalistas e mais sutis com o passar dos anos?"*
*   **Valor:** Define a frequência necessária de retreinamento do modelo para evitar a obsolescência.

### 5. Explicabilidade com XAI (Explainable AI)
Transformar a "caixa preta" da IA em decisões transparentes utilizando SHAP ou LIME.
*   **Ação:** Implementar a biblioteca `SHAP` para gerar gráficos de contribuição por feature para cada predição individual.
*   **Pergunta-Chave:** *"Quais palavras ou métricas gramaticais foram determinantes para a classificação desta notícia específica?"*
*   **Valor:** Permite que a aplicação final diga ao usuário: *"Classificado como Fake porque contém termos sensacionalistas e baixa densidade de verbos"*.

---

## 🚀 Matriz de Priorização

| Análise | Esforço | Impacto | Prioridade |
| :--- | :--- | :--- | :--- |
| **Análise de Erros** | Baixo | Médio | 🟢 Alta |
| **Explicabilidade (XAI)** | Médio | Altíssimo | 🟢 Alta |
| **Agrupamento Semântico** | Médio | Médio | 🟡 Média |
| **Viés por Fonte** | Baixo | Médio | 🟡 Média |
| **Análise Temporal** | Médio | Baixo | ⚪ Baixa |

---
**Status:** Planejado | **Responsável:** Time 7 / Cientista de Dados**
