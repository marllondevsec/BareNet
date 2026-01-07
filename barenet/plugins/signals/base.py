# barenet/plugins/signals/base.py
from typing import Optional, Dict

class BaseSignal:
    """
    Contrato mínimo para um detector / filtro.
    - name: identificador único (str)
    - description: texto curto (str)
    - analyze(page: Dict) -> Optional[Dict]: retorna dict com finding ou None
    """
    name = "base"
    description = "Base signal - no-op"

    def analyze(self, page: Dict) -> Optional[Dict]:
        raise NotImplementedError
