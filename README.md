# Fake News Detector: Sistema Hibrido de Deteccao

Este projeto implementa um sistema de deteccao de noticias falsas (Fake News) utilizando uma abordagem hibrida que combina Deep Learning Semantico (BERT) e Analise Estilometrica (NLP). O resultado e um modelo capaz de entender tanto o contexto da noticia quanto a forma como ela foi escrita, atingindo **93% de acuracia** e operando em arquitetura Serverless na nuvem.

## Visao Geral

O sistema nao se baseia apenas em palavras-chave, mas em duas dimensoes de analise:
1.  **Dimensao Semantica:** Utiliza o modelo `BERTimbau` (expandido para processar 512 tokens) para extrair vetores de sentido (embeddings), identificando a logica e o contexto da noticia na sua totalidade.
2.  **Dimensao Estilometrica:** Analisa a gramatica (proporcao de verbos, adjetivos e pronomes) e o "score emocional" (uso de caps lock, exclamacoes e palavras sensacionalistas).

Essas duas dimensoes sao fundidas e classificadas por um algoritmo nativo `HistGradientBoostingClassifier`, criando um detetor robusto, generalista e totalmente compativel com processadores ARM (como o Apple M4) sem necessidade de dependencias C++ externas.

---

## Arquitetura Tecnica (Refatorada)

O projeto segue principios de Clean Code e Arquitetura em Camadas, estando preparado para execucao em Nuvem.

### Estrutura de Pastas e Componentes
- **`src/fakenews/core/`**: O "coracao" do projeto.
  - `model_manager.py`: Gere a carga de modelos via padrao Singleton.
  - `features.py`: Classe `FeatureExtractor` responsavel por transformar texto bruto em vetores numericos.
- **`src/bot.py`**: Interface principal do utilizador via Telegram (Microsservico de Mensageria).
- **`src/fakenews/api/app.py`**: Camada de interface alternativa (FastAPI).
- **`src/fakenews/processing/`**: Ferramentas de extracao, com o web scraper integrado para leitura automatica de URLs.

### Pipeline de Dados e Deploy
`Datasets Unificados` -> `Extracao de Features (BERT 512 + Spacy)` -> `HistGradientBoosting` -> `xgb_model.joblib` (forçado no Git) -> `Deploy Serverless (Google Cloud Run)` -> `Telegram Bot`.

---

## Guia de Utilizacao (Producao)

O sistema ja esta hospedado no Google Cloud Run e pode ser utilizado imediatamente sem qualquer instalacao:

1. Aceda ao nosso Bot no Telegram.
2. Envie o comando `/start`.
3. Cole um texto suspeito ou envie a URL de uma noticia (G1, UOL, etc.) para que a IA raspe o site e analise a veracidade em tempo real.

---

## Guia de Instalacao e Uso (Desenvolvedores)

Se deseja correr a infraestrutura localmente para desenvolvimento:

### 1. Pre-requisitos
Os datasets (`FakeRecogna`, `Fake.br-Corpus`, `FACTCK.BR`) sao git submodules. Depois de clonar o repositorio, execute:
```bash
git submodule update --init --recursive
```

### 2. Instalacao das Dependencias
Instale as bibliotecas e o modelo de linguagem do Spacy:
```bash
pip install -r requirements.txt
pip install -e .
python3 -m spacy download pt_core_news_sm
```

### 3. Configuracao de Seguranca (.env)
Para proteger credenciais, crie um ficheiro `.env` na raiz do projeto e adicione o seu Token do Telegram (gerado via BotFather):
```env
TELEGRAM_TOKEN="SEU_TOKEN_AQUI"
```

### 4. Executando o Bot (Interface Principal)
O modelo agora tem suporte nativo a aceleracao de hardware (MPS no Apple Silicon ou CUDA em GPUs). Para iniciar o orquestrador do Telegram:
```bash
python3 src/bot.py
```

### 5. Executando a API (Interface Secundaria)
Caso queira testar a aplicacao via HTTP (Swagger):
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 -m fakenews.api.app
```
Aceda a `http://localhost:8000/docs`.

---

## Resultados Alcancados

| Versao do Modelo | Acuracia | Base de Dados | Tecnica |
| :--- | :--- | :--- | :--- |
| Baseline | 74% | Fake.br | Estilometria Simples |
| Teste de Estresse | 66% | 3 Bases | Estilometria Simples |
| **Hibrido Final (Atual)** | **93%** | **3 Bases** | **BERT 512 + HistGradientBoosting** |

---

## Roadmap de Evolucao
- [x] **Interface Conversacional:** Bot do Telegram para utilizadores finais com leitura de URLs.
- [x] **Cloud Deploy:** Hospedagem da arquitetura de predicao no Google Cloud Run (Serverless).
- [ ] **Banco de Dados:** Armazenamento de predicoes para criar um ciclo de aprendizagem (Data Flywheel).
- [ ] **Feedback Loop:** Sistema para que humanos corrijam a IA e melhorem o modelo em tempo real.

---
**Desenvolvido por:** Time 7 