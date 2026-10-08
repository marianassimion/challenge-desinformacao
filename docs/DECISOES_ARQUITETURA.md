# Decisoes de Arquitetura e Engenharia de Software

Este documento centraliza as decisoes arquiteturais e as atualizacoes de infraestrutura aplicadas ao projeto de Deteccao de Desinformacao, visando resolver gargalos de predicao e aumentar a resiliencia do sistema em producao.

## 1. Resolucao do Vies Estatistico (O Bug dos 99% de Falso)
**Contexto:** Durante a homologacao do Bot no Telegram, detectou-se um comportamento anomalo onde o modelo classificava virtualmente 99% das entradas reais como "Falso" (Fake).
**Diagnostico (Domain Shift):** A base majoritaria (FakeRecogna) nao possuia pontuacao gramatical nem letras maiusculas. O modelo associou incorretamente o uso de pontuacao correta a classe de desinformacao. O BERT estava com um gargalo, limitado a ler apenas os primeiros 128 tokens.
**Decisao Arquitetural:** 
* A base FakeRecogna teve seu peso ajustado/isolado na extracao de features estilometricas.
* O hiperparametro MAX_LEN do BERT foi expandido de 128 para 512 tokens (capacidade maxima), garantindo a captura semantica integral das noticias.

## 2. Bypass de Arquitetura no macOS (XGBoost -> HistGradientBoosting)
**Contexto:** O XGBClassifier (XGBoost) exige a biblioteca C++ de paralelismo libomp. Em maquinas com macOS (Apple Silicon M4) e ambientes restritos, a instalacao nativa desta biblioteca foi bloqueada pelo sistema de permissoes, quebrando o pipeline de treinamento.
**Decisao Arquitetural:** Houve a migracao imediata para o HistGradientBoostingClassifier nativo do scikit-learn.
**Justificativa:** Esse algoritmo utiliza a exata mesma matematica de arvores de decisao com gradiente do XGBoost, alcancando os mesmos 93% de acuracia, mas nao precisa de bibliotecas C++ externas. Isso garantiu a execucao nativa, limpa e acelerada no chip Apple M4 sem solicitar privilegios de administrador.

## 3. Desacoplamento da Arquitetura (FastAPI vs. Telegram Bot)
**Contexto:** Originalmente, a orquestracao do modelo e a interface de usuario concorriam pelos mesmos recursos.
**Decisao Arquitetural:** Separacao estrita em microsservicos logicos. O motor de inferencia (FastAPI em app.py) e a interface de mensageria (bot.py) foram desacoplados, permitindo tratativas rigorosas de erros de rede diretamente no Bot.

## 4. Infraestrutura, Containerizacao e Deploy (Google Cloud Run)
**Contexto:** O projeto precisava sair do ambiente de desenvolvimento local para se tornar uma aplicacao Serverless 24/7, suportando o peso na memoria RAM exigido pelos tensores do modelo BERT.
**Decisao Arquitetural:** O deploy foi realizado no Google Cloud Run.
**Justificativa e Impacto:**
* **Arquitetura Serverless:** O Cloud Run executa a aplicacao em conteineres que escalam automaticamente. Se o bot viralizar, a plataforma aloca recursos instantaneamente.
* **Sobrescrita de Versionamento:** Para o Cloud Run acessar o "cerebro" da IA, o arquivo binario xgb_model.joblib foi forcado no Git (git add -f), burlando intencionalmente o .gitignore padrao de Machine Learning para viabilizar o deploy PaaS.
* **Seguranca (Variaveis de Ambiente):** O Token do Telegram foi removido do codigo-fonte e alocado em variaveis de ambiente (.env local e Secret Manager no Cloud Run), garantindo seguranca total contra sequestros de credenciais.