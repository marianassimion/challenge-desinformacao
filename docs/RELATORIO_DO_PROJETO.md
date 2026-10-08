# Relatorio de Evolucao do Projeto: Detector de Fake News Hibrido

## Resumo Executivo
Este documento detalha a jornada tecnica de transformacao de um script experimental de analise de dados em um sistema profissional de deteccao de desinformacao. O projeto evoluiu de uma analise estatistica simples para uma arquitetura de IA hibrida, operando na nuvem em tempo real.

---

## 1. A Jornada de Evolucao do Modelo

### Passo 1: O Baseline Inicial (Estilometria Simples)
*   **Abordagem:** Uso de RandomForest e analise da "forma" da escrita.
*   **Resultado:** 74% de Acuracia.

### Passo 2: O Teste de Estresse (Expansao de Dados)
*   **Abordagem:** Unificacao das bases (Fake.br, FakeRecogna, FACTCK.BR).
*   **Resultado:** 66% de Acuracia. Queda provou que "estilo" nao e suficiente.

### Passo 3: A Solucao Hibrida (Semantica + HistGradientBoosting)
*   **Abordagem:** Implementacao do BERTimbau (512 tokens) fundido com analise do Spacy, utilizando o poderoso HistGradientBoostingClassifier nativo.
*   **Resultado:** 93% de Acuracia.
*   **Insight:** A execucao nativa resolveu gargalos de hardware no Apple Silicon (M4), e a uniao da semantica com estilometria criou um modelo de nivel de producao.

---

## 2. Arquitetura Tecnica Atual

### Componentes do Sistema:
1.  **hybrid_model.py (A Engine):** Processa dados via Spacy, gera vetores via BERT e treina o modelo preditivo final.
2.  **xgb_model.joblib (O Cerebro):** Arquivo binario que contem o HistGradientBoosting treinado. Forcado no Git para viabilizar o deploy.
3.  **bot.py e app.py (As Interfaces):** O Bot do Telegram age como o microsservico primario de mensageria interagindo com usuarios, enquanto o FastAPI fornece uma via alternativa via HTTP.
4.  **requirements.txt:** Lista de dependencias limpas e seguras, abstraindo credenciais via python-dotenv.

---

**Responsavel Tecnico:** Time 7 / Arquiteto de Software
**Status Atual:** Em Producao (Live) | API e Bot Operacionais na Nuvem.

## 3. Roadmap do Produto

O projeto saiu da fase de "Pesquisa" e agora e um "Produto" funcional. 

### Fases Concluidas (O Coracao e O Corpo)
*   [x] Implementacao da extracao de vetores do BERTimbau e features gramaticais.
*   [x] Validacao da acuracia final (93%) utilizando HistGradientBoosting.
*   [x] Desenvolvimento da interface conversacional via Bot do Telegram com leitura de URLs.
*   [x] Deploy Serverless: Sistema hospedado com sucesso no Google Cloud Run, garantindo alta disponibilidade (24/7) e seguranca via .env.

### Proximos Passos: A Mente (Memoria e Aprendizado)
*   [ ] Integracao de Banco de Dados: Armazenamento de cada noticia analisada.
*   [ ] Ciclo de Feedback: Permitir que usuarios humanos corrijam a IA.
*   [ ] Implementacao de monitoramento de drift (quando o estilo das fake news muda com o tempo).