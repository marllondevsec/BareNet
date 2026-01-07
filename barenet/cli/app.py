# barenet/cli/app.py
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.box import ROUNDED
from datetime import datetime

ASCII_LOGO = r"""
 ____  _    _  ____  _   _  _   _
|  _ \| |  | |/ __ \| \ | || \ | |
| |_) | |  | | |  | |  \| ||  \| |
|  _ <| |  | | |  | | . ` || . ` |
| |_) | |__| | |__| | |\  || |\  |
|____/ \____/ \____/|_| \_||_| \_|
"""

class BareNetCLI:
    """
    CLI visual usando rich.
    Construa com: cli = BareNetCLI(engine)
    engine precisa expor: open_url(url: str) -> dict (contrato abaixo)
    """

    def __init__(self, engine):
        self.engine = engine
        self.console = Console()
        self.version = "BareNet 0.1"

    def _render_header(self):
        logo = Text(ASCII_LOGO, style="bold red")
        info = Text(f"{self.version} — navegador passivo", style="bold white")
        header = Panel(
            Align.center(logo + Text("\n") + info),
            box=ROUNDED,
            padding=(1, 2),
            title="BareNet",
            border_style="red",
        )
        self.console.print(header)

    def _render_footer(self):
        hint = Text("Digite uma URL (https://...) ou um termo de busca. 'q' para sair.", style="dim")
        self.console.print(Panel(hint, padding=(0,1), box=ROUNDED))

    def _show_result(self, result: dict):
        if not result:
            self.console.print(Panel("Sem resultado.", style="yellow"))
            return

        md = Text()
        md.append(f"URL: {result.get('url')}\n", style="bold")
        md.append(f"Status: {result.get('status')}\n")
        md.append(f"Content-Type: {result.get('content_type')}\n")
        md.append(f"Tamanho: {result.get('size')} bytes\n")
        if result.get("context_date"):
            # context_date pode ser datetime ou string, lidamos com ambos
            cd = result["context_date"]
            try:
                md.append(f"Data de contexto: {cd.date()}\n")
            except Exception:
                md.append(f"Data de contexto: {cd}\n")
        md.append(f"Fetchedo em: {result.get('fetched_at')}\n")
        md.append("\nTrecho (primeiros 1 KiB):\n")
        md.append(result.get("snippet", "")[:1024])
        if result.get("findings"):
            md.append("\n\nAchados:\n", style="bold red")
            for f in result["findings"]:
                # espera-se que cada finding seja um dict com name/description (convenção)
                name = f.get("name", "<filter>")
                desc = f.get("description", "")
                md.append(f"- {name}: {desc}\n")
        self.console.print(Panel(md, title="Resultado", box=ROUNDED))

    def run(self):
        while True:
            self.console.clear()
            self._render_header()
            self._render_footer()

            try:
                raw = self.console.input("[bold white on red]  › [/bold white on red] ")
            except (KeyboardInterrupt, EOFError):
                self.console.print("\nSaindo.")
                return

            cmd = raw.strip()
            if not cmd:
                continue
            if cmd.lower() in ("q", "quit", "exit"):
                self.console.print("Saindo.")
                return

            # se parece URL -> abrir (engine)
            if cmd.startswith("http://") or cmd.startswith("https://"):
                try:
                    result = self.engine.open_url(cmd)
                except Exception as e:
                    self.console.print(Panel(f"[ERRO] {e}", style="red"))
                    self.console.input("Enter para continuar...")
                    continue

                self._show_result(result)
                self.console.input("Enter para continuar...")
                continue

            # caso contrário, tratar como pesquisa (stub por enquanto)
            self.console.print(Panel(f"Pesquisa por: [bold]{cmd}[/bold]\n( ainda não implementada )", box=ROUNDED))
            self.console.input("Enter para continuar...")
