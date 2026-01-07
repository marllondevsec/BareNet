# barenet/plugins/signals/header_check.py
from .base import BaseSignal

class HeaderCheck(BaseSignal):
    name = "header_check"
    description = "Verifica ausência de headers de segurança (X-Frame-Options, HSTS)"

    def analyze(self, page: dict):
        """Retorna um finding dict se algo estiver faltando."""
        headers = page.get("headers") or {}
        lower_keys = {k.lower() for k in headers.keys()}
        findings = []

        # X-Frame-Options
        if "x-frame-options" not in lower_keys:
            findings.append("X-Frame-Options ausente")

        # HSTS (aplica-se apenas a https)
        if page.get("url", "").startswith("https://") and "strict-transport-security" not in lower_keys:
            findings.append("Strict-Transport-Security (HSTS) ausente")

        if findings:
            return {"name": self.name, "description": "; ".join(findings)}
        return None
