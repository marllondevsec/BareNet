# barenet/core/engine.py
from .http import PassiveHTTPClient
from datetime import datetime
import pkgutil
import importlib
from typing import List, Dict
from barenet.plugins.signals.base import BaseSignal
import barenet.config as config

def discover_signals() -> List[BaseSignal]:
    """Descobre e instancia automaticamente filtros dentro de barenet.plugins.signals"""
    signals = []
    try:
        import barenet.plugins.signals as signals_pkg
    except Exception:
        return signals

    for finder, name, ispkg in pkgutil.iter_modules(signals_pkg.__path__):
        if name.startswith("_"):
            continue
        mod_name = f"barenet.plugins.signals.{name}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and issubclass(obj, BaseSignal) and obj is not BaseSignal:
                try:
                    signals.append(obj())
                except Exception:
                    # não falhar a descoberta por conta de inicialização de um filtro
                    pass
    return signals

class BareNetEngine:
    def __init__(self):
        self.http = PassiveHTTPClient()
        self.context_date = None
        # Discover all filters (instances)
        self.all_signals: List[BaseSignal] = discover_signals()
        # Load enabled list from config
        self.cfg = config.load()
        enabled = set(self.cfg.get("enabled_filters", []))
        # active filters are instances whose .name is in enabled
        self.filters: List[BaseSignal] = [s for s in self.all_signals if s.name in enabled]

    def set_date(self, date_obj):
        self.context_date = date_obj

    def list_available_signals(self) -> List[Dict]:
        """Retorna lista de sinais com status (enabled/disabled)."""
        enabled = {s.name for s in self.filters}
        out = []
        for s in self.all_signals:
            out.append({"name": s.name, "description": s.description, "enabled": s.name in enabled})
        return out

    def toggle_signal(self, name: str) -> bool:
        """Ativa/desativa um filtro pelo nome. Retorna novo estado (True=enabled)."""
        found = next((s for s in self.all_signals if s.name == name), None)
        if not found:
            raise KeyError(f"Signal not found: {name}")

        enabled_names = set(self.cfg.get("enabled_filters", []))
        if name in enabled_names:
            enabled_names.remove(name)
            # remove from active list
            self.filters = [s for s in self.filters if s.name != name]
            new_state = False
        else:
            enabled_names.add(name)
            self.filters.append(found)
            new_state = True

        self.cfg["enabled_filters"] = list(enabled_names)
        config.save(self.cfg)
        return new_state

    def open_url(self, url: str) -> Dict:
        resp = self.http.fetch(url)
        page = {
            "url": resp.url,
            "status": resp.status,
            "content_type": resp.content_type,
            "size": resp.size,
            "headers": getattr(resp, "headers", {}) or {},
            "snippet": resp.body[:1024].decode(errors="replace"),
            "fetched_at": datetime.utcnow(),
            "context_date": self.context_date,
        }

        findings = []
        for flt in self.filters:
            try:
                r = flt.analyze(page)
                if r:
                    findings.append(r)
            except Exception:
                # um filtro defeituoso não para a execução
                pass
        page["findings"] = findings
        return page
