# Métricas de Performance do Modelo

Este documento detalha a performance do modelo de classificação de notícias falsas implementado no projeto.

## 1. Configuração do Experimento
- **Algoritmo:** Random Forest Classifier
- **Dataset Principal:** Fake.br-Corpus
- **Divisão de Dados:** 80% Treino / 20% Teste
- **Features Utilizadas:**
  - Percentual de Verbos
  - Percentual de Adjetivos
  - Percentual de Pronomes
  - Score Emocional (Sensacionalismo, Exclamações, Maiúsculas)

## 2. Resultados de Performance (Test Set)

| Métrica | Valor | Interpretação |
| :--- | :--- | :--- |
| **Acurácia Geral** | **74%** | O modelo acerta 74% de todas as predições. |
| **Precision (Falsas)** | **77%** | Quando o modelo diz que é Fake, ele está correto em 77% das vezes. |
| **Recall (Falsas)** | **69%** | O modelo consegue identificar 69% de todas as fake news presentes no teste. |
| **F1-Score (Falsas)** | **72%** | Equilíbrio entre Precision e Recall para a classe Fake. |

## 3. Análise de Importância das Features
O modelo Random Forest permitiu identificar quais características são os maiores "indicadores" de desinformação:

1. **Score Emocional (32.9%)** $\rightarrow$ O maior indicador. Títulos sensacionalistas e uso de maiúsculas são fortes sinais de fake news.
2. **Percentual de Verbos (23.9%)**
3. **Percentual de Pronomes (23.5%)**
4. **Percentual de Adjetivos (19.5%)**

## 4. Conclusão da Modelagem
O modelo apresenta uma performance sólida para um MVP, com a **acurácia de 74%**. O destaque é a eficácia do **Score Emocional**, validando a hipótese levantada na pesquisa qualitativa de que o sensacionalismo é o principal marcador de desinformação.
