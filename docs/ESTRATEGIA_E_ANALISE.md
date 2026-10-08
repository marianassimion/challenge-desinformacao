# Analise Estrategica e Roadmap: Deteccao de Fake News

Este documento detalha a visao de produto, a arquitetura e o plano de implementacao para o sistema de deteccao de noticias falsas.

## 1. Visao de Produto (SaaS de Desinformacao)

O objetivo e transformar a analise de dados numa aplicacao funcional onde o utilizador final possa validar noticias em tempo real.

### O Ciclo de Valor (Data Flywheel)
O produto nao sera apenas um classificador, mas um sistema que aprende continuamente:
Noticia do Utilizador -> Modelo Hibrido -> Veredito -> Banco de Dados -> Retreinamento.

Ao guardar cada analise no banco de dados, o sistema cria um ativo de dados proprietario, permitindo que o modelo evolua conforme novas formas de desinformacao surgem.

---

## 2. Analise de Experimentos e Validacao

### Teste de Generalizacao (Baseline vs. Unificado)
Realizamos um experimento comparando o uso de um unico dataset (Fake.br-Corpus) contra a unificacao de tres bases (Fake.br, FakeRecogna, FACTCK.BR).

| Metrica | Apenas Fake.br-Corpus | Dataset Unificado | Insight |
| :--- | :--- | :--- | :--- |
| **Acuracia** | 74% | 66% | A queda na acuracia indica que o modelo estava "viciado" no estilo de um unico dataset. |
| **Principal Feature** | score_emocional | perc_verbos | A diversidade de dados provou que o "sentimentalismo" e facil de detetar, mas a "estrutura gramatical" e mais consistente. |

**Conclusao:** A queda na precisao validou a necessidade de evoluir da Estilometria Simples para a Analise Semantica (BERTimbau), processando 512 tokens para evitar perda de contexto em textos longos. Com a nova arquitetura, a acuracia final saltou para 93%.

---

## 3. Arquitetura de Modelagem

### Decisao: Pipeline Hibrido Avancado
Para atingir a maxima precisao, utilizamos a combinacao de duas inteligencias:

1.  **Inteligencia Semantica (BERTimbau):** Extracao de embeddings (vetores de sentido) para entender o contexto e a logica do texto.
2.  **Inteligencia Estilometrica (Spacy/Emocional):** Analise de classes gramaticais e gatilhos emocionais (maiusculas, exclamacoes, palavras sensacionalistas).
3.  **Motor Preditivo:** HistGradientBoostingClassifier, implementado como um Bypass de Arquitetura para garantir suporte nativo sem gargalos de permissao C++ no Apple Silicon.

---

## 4. Roadmap de Implementacao

**Responsavel Tecnico:** Time 7
**Status Atual:** Em Producao (Live) | API e Bot Operacionais.

O desenvolvimento esta dividido em tres fases focadas em transformar o codigo em produto:

### Fase 1: O Coracao (Precisao Maxima)
*   [x] Extracao de vetores do BERTimbau com capacidade para 512 tokens.
*   [x] Fusao de vetores BERT + Metricas Gramaticais -> Modelo Final.
*   [x] Validacao da acuracia final em 93%.

### Fase 2: O Corpo (Interface e Entrega)
*   [x] Desenvolver Bot do Telegram com extracao de URLs (Trafilatura).
*   [x] Protecao de Credenciais usando Variaveis de Ambiente (.env).
*   [x] Deploy Serverless realizado via Google Cloud Run.

### Fase 3: A Mente (Memoria e Aprendizado)
*   [ ] Configurar banco de dados (PostgreSQL/MongoDB) para guardar predicoes.
*   [ ] Implementar sistema de feedback (Validacao Humana).
*   [ ] Criar pipeline de retreinamento automatico com novos dados recolhidos.