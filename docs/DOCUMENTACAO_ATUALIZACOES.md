# 📄 Relatório de Atualizações - Fake News Detector API

Data: 24/09/2026

Este documento detalha todas as melhorias arquiteturais, correções de bugs e novas funcionalidades implementadas no projeto para garantir escalabilidade, manutenibilidade e estabilidade.

---

## 🚀 1. Refatoração de Arquitetura (Clean Code)

O projeto foi migrado de um modelo de "script único" para uma **Arquitetura em Camadas**, seguindo os princípios de Responsabilidade Única (SRP).

### 🏗️ Mudanças Estruturais
- **`src/fakenews/core/model_manager.py`**: Implementação do padrão **Singleton**. Agora, os modelos (BERT, Random Forest e Spacy) são carregados apenas uma vez na memória e compartilhados por toda a aplicação, eliminando o uso de variáveis globais e reduzindo o consumo de RAM.
- **`src/fakenews/core/features.py`**: Criação da classe `FeatureExtractor`. Toda a lógica matemática de extração de embeddings do BERT e cálculos de estilometria foi movida para cá, isolando a lógica de Ciência de Dados da lógica de API.
- **`src/fakenews/api/app.py`**: A API foi simplificada para atuar apenas como a camada de interface. Ela agora orquestra as chamadas ao `ModelManager` e `FeatureExtractor`, sem precisar conhecer os detalhes internos do modelo.

---

## 🐞 2. Correções de Bugs e Estabilidade

### 🛠️ Problema de Importação (`ModuleNotFoundError`)
- **Sintoma**: Erros ao importar `fakenews.processing.scraper`.
- **Causa**: Problemas na resolução do caminho (`PYTHONPATH`) ao rodar o servidor a partir de subpastas ou worktrees.
- **Solução**: Implementação de configuração de caminho absoluto no topo do `app.py`, garantindo que o Python localize a raiz do projeto independentemente de como o servidor seja iniciado.

### 🛠️ Erro de Processamento Assíncrono (`AttributeError`)
- **Sintoma**: Erro `coroutine object has no attribute 'strip'` ao enviar links.
- **Causa**: A função `extract_text_from_url` era assíncrona (`async`), mas estava sendo chamada sem a palavra-chave `await`.
- **Solução**: Implementação correta do `await` na chamada do scraper, garantindo que o texto seja totalmente baixadp antes de iniciar a validação e predição.

### 🛠️ Validação de JSON no Swagger
- **Sintoma**: Erro `422 Unprocessable Content` ao colar textos com quebras de linha.
- **Causa**: O padrão JSON não aceita quebras de linha literais dentro de strings.
- **Solução**: Ajuste na estrutura de requisição para suportar campos opcionais e melhoria na compatibilidade de recebimento de dados.

---

## ✨ 3. Novas Funcionalidades

### ⏱️ Medição de Latência (Performance Monitoring)
Foi implementada um **Middleware de Latência** no FastAPI. 
- **Funcionalidade**: O sistema agora captura o tempo exato de início e fim de cada requisição.
- **Resultado**: O tempo de resposta é impresso no log do servidor (ex: `Request to /predict took 0.4521s`) e enviado no cabeçalho HTTP da resposta como `X-Process-Time`.

### 🔗 Estabilização de URLs
A opção de análise via link foi totalmente restaurada e integrada à nova arquitetura, permitindo que o usuário escolha entre enviar o texto bruto ou uma URL do G1/outros portais.

---

## 🛠️ 4. Guia de Execução Atualizado

Para garantir que todas as dependências e caminhos sejam resolvidos corretamente, o projeto deve ser iniciado da seguinte forma:

```bash
# 1. Definir o caminho dos módulos
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 2. Iniciar a API como módulo do Python
python3 -m fakenews.api.app
```

---

**Status Final:** ✅ Estável | 🚀 Refatorado | ⏱️ Monitorado
