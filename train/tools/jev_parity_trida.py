"""Compatibility entry point; use decision_parity_trida.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("decision_parity_trida.py")), run_name="__main__")
