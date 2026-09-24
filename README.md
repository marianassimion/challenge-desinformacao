# 🛡️ Fake News Detector: Sistema Híbrido de Detecção

Este projeto implementa um sistema de detecção de notícias falsas (Fake News) utilizando uma abordagem híbrida que combina **Deep Learning Semântico (BERT)** e **Análise Estilométrica (NLP)**. O resultado é um modelo capaz de entender tanto o contexto da notícia quanto a forma como ela foi escrita, atingindo **87% de acurácia**.

## 🚀 Visão Geral

O sistema não se baseia apenas em palavras-chave, mas em duas dimensões de análise:
1.  **Dimensão Semântica:** Utiliza o modelo `BERTimbau` (BERT treinado para português) para extrair vetores de sentido (*embeddings*), identificando a lógica e o contexto da notícia.
2.  **Dimensão Estilométrica:** Analisa a gramática (proporção de verbos, adjetivos e pronomes) e o "score emocional" (uso de caps lock, exclamações e palavras sensacionalistas).

Essas duas dimensões são fundidas e classificadas por uma **Random Forest**, criando um detector robusto e generalista.

---

## 🛠️ Arquitetura Técnica (Refatorada)

O projeto segue princípios de **Clean Code** e **Arquitetura em Camadas** para garantir manutenibilidade e escalabilidade.

### 📂 Estrutura de Pastas
- **`src/fakenews/core/`**: O "coração" do projeto.
    - `model_manager.py`: Gerencia a carga de modelos via padrão **Singleton** (carrega modelos apenas uma vez na memória).
    - `features.py`: Classe `FeatureExtractor` responsável por transformar texto bruto em vetores numéricos.
- **`src/fakenews/api/`**: Camada de interface.
    - `app.py`: Servidor FastAPI que expõe a predição para o mundo externo.
- **`src/fakenews/processing/`**: Ferramentas de extração, como o `scraper.py` para leitura de URLs.

### Pipeline de Dados
`Datasets (Fake.br, FakeRecogna, FACTCK.BR)` $\rightarrow$ `Limpeza e Unificação` $\rightarrow$ `Extração de Features (BERT + Spacy)` $\rightarrow$ `Random Forest` $\rightarrow$ `Modelo Serializado (.joblib)` $\rightarrow$ `API FastAPI`.

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
Para rodar o servidor corretamente e evitar erros de importação de módulos, utilize o comando abaixo na raiz do projeto:

```bash
# 1. Define o caminho dos módulos para o Python
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 2. Inicia a API como módulo
python3 -m fakenews.api.app
```
O servidor estará disponível em `http://localhost:8000`.

### 5. Testando a API
Você pode testar a IA através da interface interativa do FastAPI em:
👉 `http://localhost:8000/docs`

Ou via `curl` (Enviando Texto):
```bash
curl -X 'POST' 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"text": "URGENTE! Bomba revelada sobre a economia brasileira, compartilhe antes que apaguem!"}'
```

Ou via `curl` (Enviando URL):
```bash
curl -X 'POST' 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://g1.globo.com/exemplo-de-noticia"}'
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
