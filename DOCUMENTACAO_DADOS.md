# Documentação de Dados e Engenharia de Features

Este documento detalha o pipeline de dados, desde a ingestão até a geração das features utilizadas para o treinamento do modelo de detecção de desinformação.

## 1. Pipeline de Dados (Fluxo)
O pipeline de dados segue a sequência:
`Ingestão de Datasets` $\rightarrow$ `Padronização de Esquema` $\rightarrow$ `Limpeza de Texto` $\rightarrow$ `Extração de Features Linguísticas` $\rightarrow$ `Cálculo de Score Emocional` $\rightarrow$ `Dataset Final`.

## 2. Fontes de Dados
Foram utilizados dois datasets principais para compor a base de conhecimento:
- **Fake.br-Corpus:** Dataset principal, fornecendo a massa de textos rotulados como `fake` ou `true`.
- **FakeRecogna:** Dataset complementar utilizado para validação e padronização de metadados (título, subtítulo, autor, link).

## 3. Regras de Transformação e Limpeza
Para garantir a qualidade dos dados, foram aplicadas as seguintes regras:
- **Padronização de Rótulos:** Conversão de labels numéricos para categóricos (`0` $\rightarrow$ `true`, `1` $\rightarrow$ `fake`).
- **Tratamento de Texto:** Remoção de espaços extras e normalização de caracteres via encoding `utf-8` com tratamento de erros (`ignore`) para evitar quebras no processamento de arquivos `.txt`.
- **Limitação de Escopo:** Processamento limitado aos primeiros 5.000 caracteres de cada notícia para otimização de memória RAM e tempo de processamento via SpaCy.

## 4. Engenharia de Features (Feature Engineering)
As features foram criadas para capturar a "estética" e a "estrutura" de uma notícia falsa, indo além do significado das palavras.

### A. Features Linguísticas (Via SpaCy)
Utilizou-se o modelo `pt_core_news_sm` para extrair a densidade gramatical. O cálculo é feito dividindo a contagem da classe pelo tamanho total do documento:
- **`perc_verbos`**: Percentual de verbos. Notícias falsas tendem a ter estruturas verbais diferentes de reportagens factuais.
- **`perc_adjetivos`**: Percentual de adjetivos. Alta densidade de adjetivos geralmente indica textos opinativos ou sensacionalistas.
- **`perc_pronomes`**: Percentual de pronomes. Ajuda a identificar a subjetividade do texto.

### B. Feature de Score Emocional (Heurística)
Foi criada uma métrica customizada para medir o "grau de sensacionalismo" do texto.
**Fórmula de cálculo:**
$Score = \frac{(\text{Exclamações} \times 1.5) + (\text{Palavras Sensacionalistas} \times 3) + (\text{Palavras em MAIÚSCULAS})}{\text{Total de Palavras}} \times 100$

- **Palavras Sensacionalistas:** Lista de gatilhos como *"urgente"*, *"bomba"*, *"escândalo"*, *"não vão acreditar"*.
- **Normalização:** O score é limitado ao teto de **10** para evitar que textos curtíssimos com uma única exclamação distorçam a métrica.

## 5. Feature Store (Simplificado)
O resultado final do pipeline é consolidado em um DataFrame do Pandas, que serve como nossa "Feature Store" temporária, exportada para o modelo de Random Forest.
