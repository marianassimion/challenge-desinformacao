# Metricas de Performance do Modelo

Este documento detalha a performance do modelo de classificacao de noticias falsas implementado no projeto, apos a refatoracao para a nova arquitetura nativa.

## 1. Configuracao do Experimento
- **Algoritmo:** HistGradientBoostingClassifier (Nativo do Scikit-Learn, escolhido para compatibilidade nativa e alta performance no chip Apple M4)
- **Dataset Principal:** Dataset Unificado (Fake.br-Corpus + FakeRecogna + FACTCK.BR)
- **Divisao de Dados:** 80% Treino / 20% Teste
- **Features Utilizadas:**
  - Embeddings Semanticos do BERTimbau (512 tokens)
  - Percentual de Verbos
  - Percentual de Adjetivos
  - Percentual de Pronomes
  - Score Emocional (Sensacionalismo, Exclamacoes, Maiusculas)

## 2. Resultados de Performance (Test Set)

| Metrica | Valor | Interpretacao |
| :--- | :--- | :--- |
| **Acuracia Geral** | **93%** | O modelo acerta 93% de todas as predicoes. |
| **Precision (Falsas)** | **93%** | Quando o modelo diz que e Fake, ele esta correto em 93% das vezes. |
| **Recall (Falsas)** | **93%** | O modelo consegue identificar 93% de todas as fake news presentes no teste. |
| **F1-Score (Falsas)** | **93%** | Equilibrio perfeito entre Precision e Recall, demonstrando altissima robustez. |

## 3. Analise de Evolucao
O salto de 74% (Baseline inicial com Random Forest) para 93% (HistGradientBoosting + BERTimbau) comprova a eficacia da abordagem hibrida. O modelo deixou de focar apenas em padroes visuais (como o Score Emocional) e passou a compreender o contexto semantico da noticia, resistindo a ataques onde fake news sao escritas com linguagem formal.

## 4. Conclusao da Modelagem
O modelo apresenta uma performance de nivel de producao. A troca de algoritmos garantiu nao apenas um ganho estatistico, mas viabilizou a execucao ultrarrapida do sistema sem depender de bibliotecas externas complexas.