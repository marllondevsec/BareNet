# barenet/inspector.py
"""Utilities leves para extrair e sanitizar conteúdo HTML sem depender de libs externas.

- extract_title(html) -> título da página (se existir)
- html_to_text(html, max_len=4096) -> versão de texto segura (remove script/style/iframe/form)
"""

from html.parser import HTMLParser
import html as _html
from typing import List

class _TitleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._in_title = False
        self._parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self._parts.append(data)

def extract_title(html_text: str) -> str:
    """Extrai o conteúdo da tag <title> de forma segura."""
    if not html_text:
        return ""
    p = _TitleParser()
    try:
        p.feed(html_text)
        title = "".join(p._parts).strip()
        return _html.unescape(title)
    except Exception:
        return ""

class _TextExtractor(HTMLParser):
    """Extrai texto ignorando script, style, iframe e form; preserva quebras simples."""
    def __init__(self):
        super().__init__()
        self._skip_stack: List[str] = []
        self._out: List[str] = []

    def handle_starttag(self, tag, attrs):
        tag_l = tag.lower()
        if tag_l in ("script", "style", "iframe", "form"):
            self._skip_stack.append(tag_l)
        elif tag_l in ("p", "br", "div", "li", "tr"):
            # separadores simples
            self._out.append("\n")

    def handle_endtag(self, tag):
        tag_l = tag.lower()
        if self._skip_stack and tag_l == self._skip_stack[-1]:
            self._skip_stack.pop()
        elif tag_l in ("p", "div", "li", "tr"):
            self._out.append("\n")

    def handle_data(self, data):
        if not self._skip_stack:
            self._out.append(data)

def html_to_text(html_text: str, max_len: int = 4096) -> str:
    """Converte HTML para texto plano simples, removendo scripts/iframes/forms e limitando tamanho."""
    if not html_text:
        return ""
    try:
        ex = _TextExtractor()
        ex.feed(html_text)
        txt = "".join(ex._out)
        # normaliza espaços e decodifica entidades HTML
        txt = _html.unescape(txt)
        # colapsa múltiplas quebras/espaços
        lines = [line.strip() for line in txt.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        if len(cleaned) > max_len:
            return cleaned[:max_len] + "\n...[truncated]"
        return cleaned
    except Exception:
        # fallback bem simples: remover tags via replace (ultima linha de defesa)
        import re
        s = re.sub(r"<script.*?>.*?</script>", "", html_text, flags=re.S | re.I)
        s = re.sub(r"<[^>]+>", " ", s)
        s = _html.unescape(s)
        out = " ".join(s.split())
        return out[:max_len] + ("...[truncated]" if len(out) > max_len else out)

