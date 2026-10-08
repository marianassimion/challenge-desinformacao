import spacy
from duckduckgo_search import DDGS

# Carrega o modelo Spacy (o mesmo já usado na estilometria)
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "pt_core_news_sm"])
    nlp = spacy.load("pt_core_news_sm")

def extrair_palavras_chave(texto):
    """
    Extrai apenas o 'sumo' do texto para a busca ser precisa.
    Prioriza Nomes, Organizações e Locais.
    """
    doc = nlp(texto[:3000]) # Limita o tamanho para performance
    palavras = []
    
    # 1. Tenta pegar Entidades (ex: "Bahia", "OAB")
    for ent in doc.ents:
        if ent.text.lower() not in palavras:
            palavras.append(ent.text)
            
    # 2. Complementa com Substantivos e Verbos principais
    if len(palavras) < 4:
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop and len(token.text) > 2:
                if token.text.lower() not in palavras:
                    palavras.append(token.text)
    
    # Retorna até 6 termos fortes para não confundir o buscador
    return " ".join(palavras[:6])

def buscar_fontes_confiaveis(texto_suspeito):
    # Usa o Spacy para criar uma query inteligente
    palavras_chave = extrair_palavras_chave(texto_suspeito)
    print(f"🔎 Termos de busca gerados pelo Spacy: '{palavras_chave}'")

    # Define os seus 5 sites de confiança
    sites = "site:g1.globo.com OR site:uol.com.br OR site:cnnbrasil.com.br OR site:estadao.com.br OR site:bbc.com"
    
    # Monta a query avançada
    query = f"{palavras_chave} {sites}"
    resultados_encontrados = []
    
    try:
        with DDGS() as ddgs:
            # Busca focada no Brasil
            resultados = ddgs.text(query, region='br-tz', max_results=3)
            
            for r in resultados:
                resultados_encontrados.append({
                    "titulo": r["title"],
                    "link": r["href"]
                })
        return resultados_encontrados
    except Exception as e:
        print(f"Erro na busca: {e}")
        return []

# --- ÁREA DE TESTE LOCAL ---
if __name__ == "__main__":
    noticia_teste = (
        "Gente, acabei de receber no grupo da família! "
        "A advogada tenta manipular a IA em julgamento de habeas corpus na Bahia. "
        "O mundo está perdido!"
    )
    
    print("Iniciando varredura...\n")
    resultados = buscar_fontes_confiaveis(noticia_teste)
    
    print("\nResultados encontrados:")
    if resultados:
        for res in resultados:
            print(f"- {res['titulo']}\n  {res['link']}")
    else:
        print("Nenhuma notícia encontrada nos portais confiáveis.")