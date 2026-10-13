import spacy
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import sys

try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    import subprocess
    # sys.executable garante que usa o Python correto do ambiente virtual (Mac/Linux/Win)
    subprocess.run([sys.executable, "-m", "spacy", "download", "pt_core_news_sm"])
    nlp = spacy.load("pt_core_news_sm")

def extrair_palavras_chave(texto):
    doc = nlp(texto[:3000])
    palavras = []
    
    # 1. Pega Entidades (Locais, Pessoas, Organizações)
    for ent in doc.ents:
        if ent.label_ in ["LOC", "ORG", "PER"] and ent.text not in palavras:
            palavras.append(ent.text)
                
    lixo_whatsapp = ["gente", "grupo", "família", "whatsapp", "zap", "notícia", "mundo", "verdade", "mentira", "recebi"]
    
    # 2. Se tiver poucas entidades, complementa com substantivos e verbos fortes
    if len(palavras) < 4:
        for token in doc:
            if token.pos_ in ["PROPN", "NOUN", "VERB"] and len(token.text) > 3 and not token.is_stop:
                if token.text.lower() not in lixo_whatsapp and token.text not in palavras:
                    palavras.append(token.text)
    
    # Limita a 5 termos para a pesquisa ser focada
    return " ".join(palavras[:5])

def buscar_fontes_confiaveis(texto_suspeito):
    palavras_chave = extrair_palavras_chave(texto_suspeito)
    
    # Query avançada forçando os 5 principais portais
    query = f"{palavras_chave} site:g1.globo.com OR site:uol.com.br OR site:cnnbrasil.com.br OR site:estadao.com.br OR site:bbc.com"
    query_codificada = urllib.parse.quote(query)
    
    # Usando o Endpoint RSS Oficial do Google News Brasil (Imune a bloqueios anti-bot)
    url = f"https://news.google.com/rss/search?q={query_codificada}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    
    try:
        resposta = requests.get(url, timeout=10)
        resultados_encontrados = []
        
        if resposta.status_code == 200:
            # O Google devolve um XML, vamos navegá-lo para extrair os links
            root = ET.fromstring(resposta.text)
            
            # Pega as 3 primeiras notícias encontradas
            for item in root.findall('.//item')[:3]:
                titulo = item.find('title').text
                link = item.find('link').text
                resultados_encontrados.append({
                    "titulo": titulo,
                    "link": link
                })
                
        return resultados_encontrados
    except Exception as e:
        print(f"Erro no Google News RSS: {e}")
        return []