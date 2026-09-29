# Decisões de Arquitetura e Engenharia de Software

Este documento centraliza as decisões arquiteturais e as atualizações de infraestrutura aplicadas ao projeto de Detecção de Desinformação, visando resolver gargalos de predição e aumentar a resiliência do sistema em produção.

## 1. Resolução do Viés Estatístico (O Bug dos 99% de Falso)
**Contexto:** Durante a homologação do Bot no Telegram, detectou-se um comportamento anômalo onde o modelo classificava virtualmente 99% das entradas reais como "Falso" (Fake).
**Diagnóstico (Domain Shift):** A análise exploratória (EDA) revelou que a base majoritária (`FakeRecogna`) não possuía pontuação gramatical nem letras maiúsculas. O modelo de *Machine Learning* associou incorretamente o uso de pontuação correta (comum nas mensagens enviadas por usuários no Telegram) à classe de desinformação. Além disso, o modelo BERT estava com um gargalo, limitado a ler apenas os primeiros 128 tokens, ignorando o contexto de textos longos (média de 643 palavras).
**Decisão Arquitetural:** 
* A base `FakeRecogna` teve seu peso ajustado/isolado na extração de features estilométricas, impedindo que a ausência de pontuação corrompesse o aprendizado do modelo.
* O hiperparâmetro `MAX_LEN` do BERT foi expandido de 128 para 512 tokens (capacidade máxima), garantindo a captura semântica integral das notícias.

## 2. Substituição do Algoritmo Base (Random Forest ➡️ XGBoost)
**Contexto:** O `RandomForestClassifier` demonstrou dificuldade em generalizar as variações de estilo entre os datasets acadêmicos e as mensagens do mundo real (Telegram).
**Decisão Arquitetural:** Houve a migração para o algoritmo `XGBClassifier` (XGBoost).
**Justificativa:** O XGBoost lida de forma nativa e muito superior com dados não-padronizados e desbalanceamento em *features* tabulares (como os nossos *scores* estilométricos). Com a mudança, o modelo alcançou 96% de F1-Score geral e corrigiu a calibração de probabilidade, separando corretamente a intenção semântica do formato do texto.

## 3. Desacoplamento da Arquitetura (FastAPI vs. Telegram Bot)
**Contexto:** Originalmente, a orquestração do modelo e a interface de usuário concorriam pelos mesmos recursos, gerando risco de *downtime* (inatividade).
**Decisão Arquitetural:** Separação estrita em microsserviços lógicos. O motor de inferência (FastAPI em `app.py`) e a interface de mensageria (`bot.py`) foram desacoplados.
**Justificativa:** 
* **Resiliência e Prevenção de Falhas:** O ambiente do Telegram é caótico. Usuários enviam figurinhas (stickers), áudios, fotos e textos vazios. A separação permite que o `bot.py` possua camadas rigorosas de proteção (`try/except`) para bloquear *payloads* inválidos antes mesmo que eles atinjam o modelo de Machine Learning.
* **Escalabilidade:** Caso o volume de usuários no Telegram escale agressivamente, podemos replicar a API do modelo (`app.py`) em múltiplos servidores de forma independente da interface do bot.