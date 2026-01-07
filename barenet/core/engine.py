# barenet/core/engine.py
from .http import PassiveHTTPClient
from datetime import datetime

class BareNetEngine:
    """
    Engine mínima compatível com o CLI.
    open_url(url) -> dict com chaves:
      url, status, content_type, size, snippet, fetched_at, context_date, findings
    """

    def __init__(self, filters=None):
        self.http = PassiveHTTPClient()
        self.context_date = None
        self.filters = filters or []

    def set_date(self, date_obj):
        self.context_date = date_obj

    def open_url(self, url: str) -> dict:
        # fetch do conteúdo (pode lançar)
        resp = self.http.fetch(url)

        page = {
            "url": resp.url,
            "status": resp.status,
            "content_type": resp.content_type,
            "size": resp.size,
            "snippet": resp.body[:1024].decode(errors="replace"),
            "fetched_at": datetime.utcnow(),
            "context_date": self.context_date,
        }

        # aplicar filtros plug-and-play (cada filtro deve expor analyze(page)->dict|None)
        findings = []
        for flt in self.filters:
            try:
                r = flt.analyze(page)
                if r:
                    findings.append(r)
            except Exception:
                # um filtro quebrado não para o fluxo
                pass
        page["findings"] = findings

        return page
