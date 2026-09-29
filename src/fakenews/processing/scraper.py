from newspaper import Article
from bs4 import BeautifulSoup
import requests
import re
import asyncio
from playwright.async_api import async_playwright

async def extract_text_with_playwright(url: str) -> str:
    """
    Usa um navegador real (headless) para extrair o texto de sites dinâmicos.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            # Navega até a URL e espera o conteúdo carregar
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Tenta pegar o título da página
            title = await page.title()

            # Extrai todo o texto do corpo da página
            # Tentamos focar em tags comuns de conteúdo principal para evitar menus
            content_elements = await page.query_selector_all("p")
            texts = [await el.inner_text() for el in content_elements]

            # Filtramos frases curtas e repetitivas
            filtered_texts = [t for t in texts if len(t) > 40]
            body_text = "\n".join(filtered_texts)

            await browser.close()
            return f"{title}\n{body_text}"
        except Exception as e:
            await browser.close()
            raise e

def clean_extracted_text(text: str) -> str:
    """
    Remove ruídos comuns de sites de notícias para evitar que poluam a análise.
    """
    if not text:
        return ""

    noise_patterns = [
        r"Clique e siga o canal .* no WhatsApp",
        r"Inscreva-se e receba a newsletter",
        r"Entrar com Conta Globo",
        r"Leia mais:",
        r"Veja também",
        r"Mais lidas",
        r"© Copyright .*",
        r"Todos os direitos reservados",
        r"Siga-nos em",
        r" Compartilhe no",
        r"Pular para o conteúdo"
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    lines = text.split('\n')
    filtered_lines = [line.strip() for line in lines if len(line.strip()) > 40]

    return "\n".join(filtered_lines).strip()

async def extract_text_from_url(url: str) -> str:
    """
    Extrai o título e o corpo do texto de uma URL de notícia.
    Ordem de tentativa: Newspaper3k -> BeautifulSoup -> Playwright.
    """
    try:
        # 1. Tentativa com newspaper3k (Rápido)
        article = Article(url)
        article.download()
        article.parse()

        title = article.title
        text = article.text

        if text and len(text) > 300:
            cleaned = clean_extracted_text(text)
            if len(cleaned) > 100:
                return f"{title}\n{cleaned}"

        # 2. Fallback para BeautifulSoup (Se o newspaper3k falhou)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        body_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40]
        text_bs = "\n".join(body_parts)

        if len(text_bs) > 300:
            cleaned = clean_extracted_text(text_bs)
            if len(cleaned) > 100:
                return f"{soup.title.string if soup.title else 'Sem Título'}\n{cleaned}"

        # 3. Fallback para Playwright (Para sites com renderização via JS - CSR)
        print(f"Falling back to Playwright for URL: {url}")
        text_pw = await extract_text_with_playwright(url)
        cleaned_pw = clean_extracted_text(text_pw)

        if len(cleaned_pw) > 100:
            return cleaned_pw

        raise ValueError("Conteúdo extraído insuficiente mesmo com Playwright.")

    except Exception as e:
        raise Exception(f"Erro ao extrair conteúdo da URL {url}: {str(e)}")
