# 📊 Auditoria e Análise do Dataset: Detecção de Fake News

Este documento apresenta a análise técnica e a auditoria de dados do corpus unificado utilizado para o treinamento e validação do modelo de detecção de desinformação. O objetivo é garantir a transparência sobre a qualidade dos dados e a robustez do modelo.

---

## 1. Contexto e Origem
O projeto utiliza um **Dataset Unificado**, combinando três fontes distintas para mitigar o vício de estilo (*overfitting* de dataset) e aumentar a capacidade de generalização da IA.

*   **Fake.br-Corpus:** Base acadêmica focada em notícias brasileiras, publicada na *Expert Systems with Applications* (2020). Oferece pares balanceados de notícias verdadeiras e falsas.
*   **FACTCK.BR:** Base de dados proveniente de agências de checagem profissionais (Aos Fatos, Lupa, Truco) via esquema `ClaimReview`. Foca em alegações específicas e vereditos técnicos. Licença MIT.
*   **FakeRecogna:** Base de dados adicional integrada para expandir a volumetria de exemplos e diversificar os padrões de escrita.

**População:** Notícias e alegações em língua portuguesa circulando em ambiente web.
**Viés de Seleção:** Há um viés intrínseco, pois os dados provêm de fontes que já foram identificadas como "fake" ou "true" por curadores ou agências. Notícias "cinzentas" (sem veredito claro) são excluídas, o que pode tornar o modelo excessivamente confiante em casos claros, mas vulnerável a nuances.

---

## 2. Estrutura e Volumetria
O volume total do dataset unificado é de **19.103 registros**.

### Composição Detalhada:
| Fonte | Registros | Colunas Chave | Tipo de Dado | Chave Única |
| :--- | :--- | :--- | :--- | :--- |
| **FakeRecogna** | 11.903 | `text`, `label`, `title` | Misto | N/A |
| **Fake.br-Corpus**| 7.200 | `text`, `label` | Texto | ID do Arquivo |
| **FACTCK.BR** | 1.313 | `text`, `label`, `URL` | Misto | URL |

**Representação da Unidade:** Cada registro representa um documento textual único (notícia completa ou alegação + revisão).
**Variáveis Unificadas:**
*   `text` (String): Conteúdo textual processado.
*   `label` (Categorical): Binário (`fake` ou `true`).
*   `source` (Categorical): Identificador da base de origem.

---

## 3. Análise de Qualidade e Ruído
A auditoria de qualidade revelou pontos críticos que impactam diretamente a modelagem:

### 📉 Valores Nulos e Completude
*   **FakeRecogna:** Apresenta a maior fragilidade em metadados. A coluna `subtitle` possui **~53% de nulidade**, tornando-a irrelevante para a modelagem.
*   **FACTCK.BR:** Alta completude nas colunas de veredito, mas a construção do campo `text` exige a concatenação de `claim` e `review` para evitar perda de contexto.

### 📏 Distribuição de Tamanho de Texto (Sinal vs Ruído)
A variância no comprimento dos textos é extrema, o que representa um desafio para modelos de NLP:
*   **Média:** 1.833 caracteres.
*   **Mediana:** 687 caracteres.
*   **Mínimo:** 5 caracteres (Ruído/Outlier).
*   **Máximo:** 46.084 caracteres (Documentos longos).

**Impacto Técnico:** Textos extremamente curtos (ex: 5 chars) podem ser classificados erroneamente por falta de contexto. Textos excessivamente longos são truncados pelo BERT (limite de 512 tokens), o que pode causar a perda de informações cruciais localizadas no final do texto.

---

## 4. Distribuição e Relações
### Balanceamento de Classes
O dataset final apresenta um **balanceamento perfeito (50% Fake / 50% True)**. Isso é resultado da composição equilibrada entre as bases `Fake.br` e `FakeRecogna`, que compensam o desbalanceamento natural do `FACTCK.BR` (onde a maioria dos registros são desmentidos).

### Correlação Semântica vs Estilométrica
*   **Estilometria:** Funciona bem em bases controladas (ex: `Fake.br`), onde o "estilo" de mentir é consistente.
*   **Semântica:** Torna-se o diferencial ao unificar as bases, java a IA deixa de buscar "palavras-chave" e passa a analisar a estrutura lógica e o contexto do texto.

---

## 5. Definições de Pesquisa
**Objetivo:** Desenvolver um classificador robusto capaz de detectar desinformação independentemente da fonte, autor ou estilo de escrita.

**Perguntas Respondidas:**
*   A estrutura gramatical (densidade de verbos e adjetivos) é um indicador confiável? **Sim.**
*   O modelo consegue generalizar entre bases de dados diferentes? **Sim, via abordagem híbrida.**

**Limitações dos Dados:**
*   O dataset não informa a **intenção** do autor (Sátira vs. Má-fé).
*   Não há dados sobre a **viralização** da notícia, impedindo a análise de impacto social.

---

## 6. Ética e Governança de Dados
**Privacidade:** O dataset contém URLs e nomes de autores públicos. Embora sejam dados abertos, o sistema deve evitar a criação de "listas negras" de autores para evitar vieses punitivos.
**Risco de Viés:** O modelo pode correlacionar temas específicos (ex: política ou vacinas) automaticamente a `fake` apenas porque esses temas são predominantes nas amostras de desinformação.
**Mitigação:** A recomendação é que o modelo seja utilizado como uma **ferramenta de apoio à decisão**, e nunca como um juiz final e automatizado de verdade.
