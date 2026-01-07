# barenet/__main__.py
"""Entry point for python -m barenet"""

from barenet.cli.app import BareNetCLI
from barenet.core.engine import BareNetEngine

def main():
    engine = BareNetEngine()
    cli = BareNetCLI(engine)
    try:
        cli.run()
    except (KeyboardInterrupt, EOFError):
        print("\nSaindo (interrompido).")

if __name__ == "__main__":
    main()
