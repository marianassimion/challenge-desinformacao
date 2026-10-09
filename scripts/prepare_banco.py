"""
Prepara o banco_fake_news.xlsx para o treino.

Gera em data/extra/:
  - banco_treino.csv   -> entra no treino (via dataloader)
  - banco_holdout.csv  -> NUNCA entra no treino; serve só para medir recall em fake inédita

Uso (na raiz do repositório):
  python scripts/prepare_banco.py --xlsx banco_fake_news.xlsx
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="banco_fake_news.xlsx")
    ap.add_argument("--out_dir", default="data/extra")
    ap.add_argument("--min_chars", type=int, default=300)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx)
    df = df.rename(columns={
        "Título": "title",
        "Corpo da notícia": "text",
        "Fonte / Portal": "portal",
    })
    print(f"Linhas lidas: {len(df)}")

    df["text"] = df["text"].astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() >= args.min_chars]
    df = df.drop_duplicates(subset="text")
    print(f"Após filtro (>= {args.min_chars} chars) e dedup: {len(df)}")

    out = pd.DataFrame({
        "title": df["title"],
        "subtitle": "",
        "text": df["text"],
        "label": "fake",
        "source": "BancoProprio",
        "portal": df["portal"],
    })

    treino, holdout = train_test_split(out, test_size=args.holdout, random_state=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    treino.to_csv(out_dir / "banco_treino.csv", index=False, encoding="utf-8")
    holdout.to_csv(out_dir / "banco_holdout.csv", index=False, encoding="utf-8")
    print(f"Treino: {len(treino)} | Holdout: {len(holdout)} -> {out_dir}/")


if __name__ == "__main__":
    main()
