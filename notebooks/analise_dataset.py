import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def analyze():
    print("Carregando datasets...")
    # Load Recogna
    try:
        arquivo_fakerecogna = glob.glob("data/FakeRecogna/**/*.xlsx", recursive=True)[0]
        df_recogna = pd.read_excel(arquivo_fakerecogna)
        df_recogna = df_recogna.rename(columns={"Titulo": "title", "Subtitulo": "subtitle", "Noticia": "text", "Classe": "label"})
        df_recogna["label"] = df_recogna["label"].map({0: "fake", 1: "true"})
        df_recogna["source"] = "FakeRecogna"
    except Exception as e:
        df_recogna = pd.DataFrame()

    # Load Fake.br
    try:
        base_fakebr = "data/Fake.br-Corpus/full_texts"
        registros_fakebr = []
        for label in ["fake", "true"]:
            pasta = os.path.join(base_fakebr, label)
            for arquivo in glob.glob(os.path.join(pasta, "*.txt")):
                with open(arquivo, "r", encoding="utf-8", errors="ignore") as f:
                    registros_fakebr.append({"text": f.read().strip(), "label": label, "source": "Fake.br-Corpus"})
        df_fakebr = pd.DataFrame(registros_fakebr)
    except Exception as e:
        df_fakebr = pd.DataFrame()

    # Load FactCK
    try:
        df_factck = pd.read_csv("data/FACTCK.BR/FACTCKBR.tsv", sep="\t")
        df_factck["label"] = df_factck["alternativeName"].apply(lambda x: "fake" if str(x).lower() == "falso" else ("true" if str(x).lower() == "verdadeiro" else None))
        df_factck = df_factck.dropna(subset=["label"])
        df_factck["text"] = df_factck["review"].fillna("") + " " + df_factck["claim"].fillna("")
        df_factck["source"] = "FACTCK.BR"
    except Exception as e:
        df_factck = pd.DataFrame()

    # Consolidar Dataset
    df_all = pd.concat([df_recogna, df_fakebr, df_factck], ignore_index=True)
    df_all = df_all.dropna(subset=["text"])
    df_all["tamanho_texto"] = df_all["text"].str.len()
    df_all["quantidade_palavras"] = df_all["text"].apply(lambda x: len(str(x).split()))

    print("Gerando gráficos...")
    sns.set_theme(style="whitegrid")

    # Gráfico 1: Desbalanceamento de Estilo (Distribuição por Fonte)
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df_all, x="source", hue="label", palette="viridis")
    plt.title("Distribuição de Classes por Base de Dados", fontsize=14)
    plt.xlabel("Fonte do Dataset")
    plt.ylabel("Quantidade de Registros")
    plt.savefig("docs/assets/distribuicao_fontes.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico 2: Dispersão (Boxplot) do Tamanho dos Textos
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_all, x="source", y="quantidade_palavras", palette="pastel")
    plt.title("Dispersão do Número de Palavras por Fonte", fontsize=14)
    plt.xlabel("Fonte do Dataset")
    plt.ylabel("Número de Palavras (Escala Logarítmica)")
    plt.yscale("log") # Escala logarítmica ajuda a visualizar os outliers
    plt.savefig("docs/assets/dispersao_tamanho.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Estatísticas Descritivas:")
    print(f"Total de Linhas: {len(df_all)}")
    print(f"Média de Palavras: {df_all['quantidade_palavras'].mean():.0f}")
    print(f"Mediana de Palavras: {df_all['quantidade_palavras'].median():.0f}")
    
    print("Análise concluída! Imagens salvas em docs/assets/")

if __name__ == "__main__":
    analyze()