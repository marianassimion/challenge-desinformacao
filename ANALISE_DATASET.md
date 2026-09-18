# Análise Detalhada do Dataset: Detecção de Fake News

Este documento fornece uma análise exaustiva dos dados utilizados no projeto, respondendo a critérios de contexto, estrutura, qualidade, distribuição, objetivos de pesquisa e ética.

---

## 1. Contexto e Origem
O projeto utiliza um **Dataset Unificado**, composto por três fontes distintas para garantir a generalização do modelo:

*   **Fake.br-Corpus:** Coletado de websites, focado em notícias em português. Foi apresentado em conferências como PROPOR 2018 e publicado na *Expert Systems with Applications* (2020).
*   **FACTCK.BR:** Coletado via esquema `ClaimReview` de agências de checagem profissionais (Aos Fatos, Lupa e Truco). Foca em alegações específicas e seus respectivos vereditos. Licença MIT.
*   **FakeRecogna:** Base de dados adicional integrada para expandir o volume de exemplos de desinformação.

**População/Amostra:** Notícias e alegações em língua portuguesa circulando na internet.
**Viés de Seleção:** Os dados vêm de fontes que já foram identificadas como "fake" ou "true" por curadores ou agências. Notícias "cinzentas" (sem veredito claro) podem estar sub-representadas.
**Restrições:** O FACTCK.BR segue a licença MIT.

---

## 2. Estrutura
O volume total de dados unificados é de aproximadamente **20.416 registros**.

### Composição por Fonte:
| Fonte | Registros | Colunas Originais | Tipo de Dado | Chave Única |
| :--- | :--- | :--- | :--- | :--- |
| **FakeRecogna** | 11.903 | 8 | Misto (Texto, Data, Categórico) | Não explícita |
| **Fake.br-Corpus**| 7.200 | 2 | Texto | Nome do arquivo |
| **FACTCK.BR** | 1.313 | 9 | Misto (URL, Texto, Data, Rating) | URL |

**Representação da Linha:** Cada linha representa uma notícia ou uma alegação (claim) individual.
**Variáveis Principais (Unificadas):**
*   `text`: O conteúdo textual da notícia ou a junção da alegação com a revisão (Texto).
*   `label`: a classificação final: `fake` ou `true` (Categórica).
*   `source`: a origem do dado (Categórica).

---

## 3. Qualidade
**Valores Nulos:**
*   **FakeRecogna:** Apresenta nulidade significativa na coluna `Subtitulo` (~53% nulos) e pontuais em `Titulo` e `Data`.
*   **FACTCK.BR:** Nulidades baixas em `claimReviewed` e `reviewBody`.
*   **Fake.br-Corpus:** Sem nulos (baseada em arquivos `.txt`).

**Consistência:** 
Os formatos de data e labels variam entre as fontes (ex: `0/1` no Recogna, `Falso/Verdadeiro` no FactCK), exigindo a etapa de normalização implementada no `hybrid_model.py`.

**Outliers:** 
Não foram detectados valores impossíveis, mas existem textos extremamente curtos (especialmente no FactCK.BR) que podem atuar como ruído para modelos de NLP.

---

## 4. Distribuição e Relações
**Distribuição de Classes:**
*   **FakeRecogna:** Perfeitamente balanceado (50% fake / 50% true).
*   **Fake.br-Corpus:** Perfeitamente balanceado (50% fake / 50% true).
*   **FACTCK.BR:** Fortemente desbalanceado para a classe `fake` (predomínio de desmentidos).

**Relações:**
Observou-se que a "forma" da notícia (estilometria) é um forte indicador em datasets menores, mas a "semântica" (sentido) torna-se crucial quando unificamos as bases, pois o estilo de "mentir" varia entre as fontes.

---

## 5. Perguntas de Negócio e Pesquisa
**Problema a Resolver:** Como identificar automaticamente se um texto é desinformação, independentemente da fonte ou do estilo do autor?

**Perguntas que os dados respondem:**
*   Existe um padrão gramatical comum em notícias falsas? (Sim, via estilometria).
*   A IA consegue generalizar entre diferentes bases de dados? (Sim, via modelo híbrido).

**Perguntas que os dados NÃO respondem:**
*   Qual a intenção do autor ao criar a fake news?
*   Qual o impacto real (estatístico) da notícia na população?

**Variável Alvo:** `label` (Prever se a notícia é `fake` ou `true`).

---

## 6. Ética e Uso Responsável
**Dados Sensíveis:** O dataset contém nomes de autores e URLs. Embora sejam dados públicos, a divulgação de listas de "autores de fake news" pode gerar implicações legais ou éticas.
**Vieses e Discriminação:** O modelo pode aprender a associar certas palavras-chave ou temas (ex: política, religião) automaticamente a `fake` apenas porque a amostra de treino era concentrada nesses temas, gerando falsos positivos em notícias verdadeiras sobre os mesmos assuntos.
**Impacto:** Conclusões erradas podem levar à censura de informações verdadeiras ou à propagação de mentiras caso a confiança no modelo seja absoluta sem revisão humana.
