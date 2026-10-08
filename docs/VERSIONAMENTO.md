# Guia de Versionamento: Detector de Fake News Hibrido

Este documento explica como funciona o sistema de controlo de versao deste projeto. Diferente de projetos de software comuns, projetos de Machine Learning (ML) exigem o versionamento de tres pilares: Codigo, Dados e Modelos.

## 1. A Arquitetura de Versionamento

Utilizamos uma abordagem de Versionamento Tripartite, com uma adaptacao especifica para suportar o nosso deploy na nuvem:

### A. Codigo -> Git
O Git e utilizado exclusivamente para versionar a "receita" do projeto.
- **O que e versionado:** Arquivos .py, .md, .json e requirements.txt.
- **O que e ignorado:** Pastas de dados (data/), e arquivos de variaveis de ambiente (.env) por questoes de seguranca (protecao de Token).

### B. Dados e Modelos -> DVC + Excecao Estrategica
A regra padrao do projeto e utilizar o DVC (Data Version Control) para os dados pesados e cache interno, enviando o arquivo real para um storage remoto (ex: ficheiros .csv de treino).
- **A Excecao para o Deploy (Google Cloud Run):** Para viabilizar o deploy direto (Serverless), o servidor da nuvem precisa de ter acesso direto ao "cerebro" treinado. Como o nosso novo modelo preditivo (xgb_model.joblib) ficou extremamente leve (aprox. 1MB), aplicamos um bypass intencional ao .gitignore:

```bash
git add -f models/xgb_model.joblib
```

Isso permite que a nuvem puxe a arquitetura 100% pronta sem etapas complexas de DVC no servidor de producao.

### C. Experimentos -> MLflow
Enquanto o Git versiona o "como fazer" e o DVC versiona o "resultado" dos dados brutos, o MLflow versiona o "porque".
- **O que ele rastreia:** Hiperparametros (MAX_LEN do BERTimbau), Metricas (Acuracia, Precision, F1-Score) e Artefatos.
- **Utilidade:** Se mudarmos a base de dados ou o algoritmo e a acuracia oscilar, o MLflow permite-nos comparar os dois treinos e entender exatamente o que causou a alteracao.

---

## 2. Guia Rapido para Colaboradores

Se acabou de clonar o projeto, siga estes passos para sincronizar tudo:

### Sincronizar Codigo
```bash
git pull origin main
```

### Sincronizar Dados Base (DVC)
```bash
# Instale o DVC se nao tiver
pip install dvc

# Baixe a versao correta dos dados
dvc pull
```

### Visualizar Experimentos
Para ver a comparacao de todos os treinos realizados pela equipa:
```bash
# Inicie o servidor do MLflow
python3 -m mlflow ui
```
Aceda a `http://localhost:5000` no seu navegador.

---

## 3. Fluxo de Trabalho Recomendado (Clean Code Workflow)

Para manter o projeto organizado e evitar conflitos, siga este fluxo:

1. **Sempre trabalhe em branches separadas:** `git checkout -b feat/nova-funcionalidade` ou `git checkout -b fix/correcao-bug`.
2. **Sincronize a branch de testes:** Antes de finalizar, faca o merge das suas alteracoes na branch `testes` para validacao.
3. **Execucao da API ou Bot:** Devido a estrutura de pacotes do projeto, utilize sempre os comandos configurando o caminho raiz:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   python3 -m fakenews.api.app
   ```
4. **Merge Final:** Apos a validacao na branch `testes`, a alteracao deve ser fundida na `main`.

---
**Desenvolvido por:** Time 7 / Arquiteto de Software