# Análise Estratégica e Roadmap: Detecção de Fake News

Este documento detalha a visão de produto, a arquitetura e o plano de implementação para o sistema de detecção de notícias falsas.

## 🎯 1. Visão de Produto (SaaS de Desinformação)

O objetivo é transformar a análise de dados em uma aplicação funcional onde o usuário final possa validar notícias em tempo real.

### 🔄 O Ciclo de Valor (Data Flywheel)
O produto não será apenas um classificador, mas um sistema que aprende continuamente:
`Notícia do Usuário` $\rightarrow$ `Modelo Híbrido` $\rightarrow$ `Veredito` $\rightarrow$ `Banco de Dados` $\rightarrow$ `Retreinamento`.

Ao salvar cada análise no banco de dados, o sistema cria um ativo de dados proprietário, permitindo que o modelo evolua conforme novas formas de desinformação surgem.

---

## 📊 2. Análise de Experimentos e Validação

### 🧪 Teste de Generalização (Baseline vs. Unificado)
Realizamos um experimento comparando o uso de um único dataset (`Fake.br-Corpus`) contra a unificação de três bases (`Fake.br`, `FakeRecogna`, `FACTCK.BR`).

| Métrica | Apenas `Fake.br-Corpus` | **Dataset Unificado** | Insight |
| :--- | :--- | :--- | :--- |
| **Acurácia** | **74%** | **66%** | A queda na acurácia indica que o modelo estava "viciado" no estilo de um único dataset. |
| **Principal Feature** | `score_emocional` | `perc_verbos` | A diversidade de dados provou que o "sentimentalismo" é fácil de detectar, mas a "estrutura gramatical" é mais consistente. |

**Conclusão:** A queda na precisão validou a necessidade de evoluir da **Estilometria Simples** para a **Análise Semântica (BERT)**.

---

## 🛠️ 3. Arquitetura de Modelagem

### ✅ Decisão: Pipeline Híbrido (BERTimbau + Estilometria)
Para atingir a máxima precisão, utilizaremos a combinação de duas inteligências:

1.  **Inteligência Semântica (BERTimbau):** Extração de *embeddings* (vetores de sentido) para entender o contexto e a lógica do texto.
2.  **Inteligência Estilométrica (Spacy/Emocional):** Análise de classes gramaticais e gatilhos emocionais (Caps Lock, exclamações, palavras sensacionalistas).

**Estratégia de Treino:**
*   **Treino Geral:** Utilizar as 3 bases unificadas para aprender a essência da desinformação.
*   **Especialização:** Realizar ajuste final focando no `Fake.br-Corpus` para maximizar a precisão no dataset principal.

---

## 🚀 4. Roadmap de Implementação

O desenvolvimento está dividido em três fases focadas em transformar o código em produto:

### Fase 1: O Coração (Precisão Máxima)
*   [ ] Implementar a extração de vetores do BERTimbau.
*   [ ] Fundir vetores BERT + Métricas Gramaticais $\rightarrow$ Modelo Final.
*   [ ] Validar acurácia final (Meta: $> 85\%$).

### Fase 2: O Corpo (Interface e Entrega)
*   [ ] Desenvolver API em **FastAPI** com endpoint `/predict`.
*   [ ] Criar Front-end minimalista para envio de notícias.
*   [ ] Implementar a lógica de resposta instantânea.

### Fase 3: A Mente (Memória e Aprendizado)
*   [ ] Configurar banco de dados (**PostgreSQL/MongoDB**) para salvar predições.
*   [ ] Implementar sistema de feedback (Validação Humana).
*   [ ] Criar pipeline de retreinamento automático com novos dados coletados.

---
**Status:** Em Desenvolvimento | **Responsável:** Claude Code / Arquiteto de Software**
