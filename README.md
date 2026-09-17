# 🛡️ Fake News Detector: Sistema Híbrido de Detecção

Este projeto implementa um sistema de detecção de notícias falsas (Fake News) utilizando uma abordagem híbrida que combina **Deep Learning Semântico (BERT)** e **Análise Estilométrica (NLP)**. O resultado é um modelo capaz de entender tanto o contexto da notícia quanto a forma como ela foi escrita, atingindo **87% de acurácia**.

## 🚀 Visão Geral

O sistema não se baseia apenas em palavras-chave, mas em duas dimensões de análise:
1.  **Dimensão Semântica:** Utiliza o modelo `BERTimbau` (BERT treinado para português) para extrair vetores de sentido (*embeddings*), identificando a lógica e o contexto da notícia.
2.  **Dimensão Estilométrica:** Analisa a gramática (proporção de verbos, adjetivos e pronomes) e o "score emocional" (uso de caps lock, exclamações e palavras sensacionalistas).

Essas duas dimensões são fundidas e classificadas por uma **Random Forest**, criando um detector robusto e generalista.

---

## 🛠️ Arquitetura Técnica

### Pipeline de Dados
`Datasets (Fake.br, FakeRecogna, FACTCK.BR)` $\rightarrow$ `Limpeza e Unificação` $\rightarrow$ `Extração de Features (BERT + Spacy)` $\rightarrow$ `Random Forest` $\rightarrow$ `Modelo Serializado (.joblib)` $\rightarrow$ `API FastAPI`.

### Componentes do Projeto
- **`hybrid_model.py`**: Script de treinamento. Responsável por processar os dados, gerar os embeddings e salvar o modelo final.
- **`app.py`**: API de produção. Carrega o modelo salvo e oferece endpoints para predição em tempo real.
- **`rf_model.joblib`**: O arquivo binário do modelo treinado.
- **`STRATEGY_AND_ANALYSIS.md`**: Documento de visão de produto e roadmap.
- **`PROJECT_REPORT.md`**: Relatório de experimentos e resultados de acurácia.

---

## 📦 Guia de Instalação e Uso

### 1. Pré-requisitos
Você precisará do Python 3.10+ instalado em sua máquina.

### 2. Instalação das Dependências
Clone o repositório e instale as bibliotecas necessárias:
```bash
pip install -r requirements.txt
python3 -m spacy download pt_core_news_sm
```

### 3. Treinando o Modelo
Para treinar o modelo do zero e gerar o arquivo `.joblib`:
```bash
python3 hybrid_model.py
```
*Este processo baixará o BERTimbau e processará ~20k notícias. Pode levar alguns minutos.*

### 4. Executando a API
Com o modelo treinado (`rf_model.joblib` presente na pasta), inicie o servidor:
```bash
python3 app.py
```
O servidor estará disponível em `http://localhost:8000`.

### 5. Testando a API
Você pode testar a IA através da interface interativa do FastAPI em:
👉 `http://localhost:8000/docs`

Ou via `curl`:
```bash
curl -X 'POST' 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"text": "URGENTE! Bomba revelada sobre a economia brasileira, compartilhe antes que apaguem!"}'
```

---

## 📊 Resultados Alcançados

| Versão do Modelo | Acurácia | Base de Dados | Técnica |
| :--- | :--- | :--- | :--- |
| Baseline | 74% | Fake.br | Estilometria Simples |
| Unificado | 66% | 3 Bases | Estilometria Simples |
| **Híbrido Final** | **87%** | **3 Bases** | **BERT + Estilometria** |

---

## 🗺️ Roadmap de Evolução
- [ ] **Front-end:** Interface web para usuários finais.
- [ ] **Banco de Dados:** Armazenamento de predições para criar um ciclo de aprendizado.
- [ ] **Feedback Loop:** Sistema para que humanos corrijam a IA e melhorem o modelo.
- [ ] **Cloud Deploy:** Hospedagem da API em AWS/GCP/Azure.

---
**Desenvolvido por:** Time 7 / Arquiteto de Software
