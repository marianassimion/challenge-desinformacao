# 📦 Guia de Versionamento: Detector de Fake News Híbrido

Este documento explica como funciona o sistema de controle de versão deste projeto. Diferente de projetos de software comuns, projetos de Machine Learning (ML) exigem o versionamento de três pilares: **Código, Dados e Modelos**.

## 🏗️ A Arquitetura de Versionamento

Para evitar que o repositório fique lento e pesado, utilizamos uma abordagem de **Versionamento Tripartite**:

### 1. Código -> Git
O **Git** é utilizado exclusivamente para versionar a "receita" do projeto.
- **O que é versionado:** Arquivos `.py`, `.md`, `.json` e `requirements.txt`.
- **O que é ignorado:** Pastas de dados (`data/`) e arquivos binários de modelos (`models/*.joblib`).
- **Por que?** Arquivos binários grandes causam "inchaço" no Git, tornando o `clone` e o `pull` extremamente lentos.

### 2. Dados e Modelos -> DVC (Data Version Control)
Utilizamos o **DVC**, que funciona como um "Git para dados". O DVC não salva o arquivo pesado no GitHub, mas sim um pequeno arquivo de texto (ex: `rf_model.joblib.dvc`) que serve como um ponteiro.

- **Como funciona:** 
    - O arquivo pesado (`.joblib` ou `.csv`) é movido para um cache interno do DVC.
    - O arquivo `.dvc` (o ponteiro) é commitado no Git.
    - O arquivo real é enviado para um storage remoto (S3, Google Drive, etc.).
- **Vantagem:** Você pode trocar de branch no Git e, ao dar um `dvc pull`, o DVC baixa automaticamente a versão exata do modelo que corresponde àquele commit de código.

### 3. Experimentos -> MLflow
Enquanto o Git versiona o "como fazer" e o DVC versiona o "resultado", o **MLflow** versiona o **"porquê"**.

- **O que ele rastreia:** 
    - **Hiperparâmetros:** Versão do BERT utilizada, número de árvores da Random Forest.
    - **Métricas:** Acurácia, Precisão, Recall e F1-Score de cada treino.
    - **Artefatos:** O modelo final treinado.
- **Utilidade:** Se mudarmos a base de dados e a acurácia cair de 87% para 80%, o MLflow nos permite comparar os dois treinos e entender exatamente o que causou a queda.

---

## 🛠️ Guia Rápido para Colaboradores

Se você acabou de clonar o projeto, siga estes passos para sincronizar tudo:

### 1. Sincronizar Código
```bash
git pull origin main
```

### 2. Sincronizar Dados e Modelos
```bash
# Instale o DVC se não tiver
pip install dvc

# Baixe a versão correta do modelo e dos dados
dvc pull
```

### 3. Visualizar Experimentos
Para ver a comparação de todos os treinos realizados pela equipe:
```bash
# Inicie o servidor do MLflow
python3 -m mlflow ui
```
Acesse `http://localhost:5000` no seu navegador.

---

## 🔄 Fluxo de Trabalho Recomendado (Clean3 Code Workflow)

Para manter o projeto organizado e evitar conflitos, siga este fluxo:

1. **Sempre trabalhe em branches separadas:** `git checkout -b feat/nova-funcionalidade` ou `git checkout -b fix/correcao-bug`.
2. **Sincronize a branch de testes:** Antes de finalizar, faça o merge de suas alterações na branch `testes` para validação.
3. **Execução da API:** Devido à estrutura de pacotes do projeto, utilize sempre o comando de módulo:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   python3 -m fakenews.api.app
   ```
4. **Merge Final:** Após a validação na branch `testes`, a alteração deve ser fundida na `main`.

---
**Desenvolvido por:** Time 7 / Arquiteto de Software
