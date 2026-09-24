# 🧪 Guia de Testes: Detector de Fake News Híbrido

Este guia fornece instruções passo a passo para que qualquer membro da equipe possa configurar o ambiente e testar a eficácia do modelo de detecção de desinformação.

## 📋 Pré-requisitos

Antes de começar, certifique-se de ter instalado:
- **Python 3.10+**
- **Git**

---

## 🚀 1. Configuração do Ambiente

Siga estes passos para preparar sua máquina:

### Clonagem e Instalação
```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd challenge-desinformacao

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Baixe o modelo de linguagem do Spacy para Português
python3 -m spacy download pt_core_news_sm
```

### Recuperação de Dados e Modelos (DVC)
Como os arquivos pesados não ficam no Git, use o DVC para baixá-los:
```bash
# Instale o DVC
pip install dvc

# Baixe os dados e o modelo do storage remoto
dvc pull
```

---

## 🧪 2. Formas de Testar o Modelo

Existem duas maneiras principais de testar a IA: via **API Interativa** (mais fácil) ou via **Script de Treino/Validação**.

### Método A: Teste via API (Recomendado)
Este método testa o modelo em "tempo real", simulando o uso final do produto.

1. **Inicie o servidor (Caminho Profissional):**
   ```bash
   # Define o caminho dos módulos para evitar ModuleNotFoundError
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   
   # Inicia a API como módulo do Python
   python3 -m fakenews.api.app
   ```
2. **Acesse a interface visual (Swagger):**
   Abra o navegador em: 👉 `http://localhost:8000/docs`
3. **Como testar:**
   - Clique no botão **POST `/predict`**.
   - Clique em **"Try it out"**.
   - Você pode preencher o campo `text` (texto bruto) ou o campo `url` (link da notícia).
   - Clique em **"Execute"**.
   - **Analise a resposta:** Verifique a `prediction` (fake/true), a `confidence` e as métricas estilométricas.
   - **Verifique a Latência:** Olhe para o terminal onde o servidor está rodando; você verá o tempo exato de processamento da requisição (ex: `Request to /predict took 0.45s`).

### Método B: Teste de Validação (Métricas)
Se você quiser testar a acurácia do modelo em todo o dataset de teste:

1. **Execute o script de treinamento/validação:**
   ```bash
   python3 src/training/hybrid_model.py
   ```
2. **O que observar:**
   Ao final da execução, o script imprimirá o **Classification Report**. Foque nestas métricas:
   - **Accuracy:** Percentual total de acertos.
   - **F1-Score:** Equilíbrio entre Precisão e Recall.

---

## 📊 3. Como Interpretar os Resultados

| Resultado | Significado | O que observar |
| :--- | :--- | :--- |
| **Fake** | A IA detectou padrões de desinformação. | Verifique se o `score_emocional` está alto (muitas exclamações, caps lock). |
| **True** | A IA considerou a notícia legítima. | Verifique se o texto possui uma estrutura gramatical mais neutra e formal. |
| **Confidence** | Nível de certeza da IA (0.0 a 1.0). | Valores abaixo de 0.7 indicam que a IA está em dúvida. |

---

## 🚩 Reportando Erros
Se você encontrar um caso onde a IA errou feio, anote:
1. O texto ou link da notícia.
2. O resultado esperado vs. o resultado da IA.
3. O tempo de resposta (latência).

Isso nos ajudará a otimizar o modelo e a performance do sistema!
