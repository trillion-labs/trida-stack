"""Compatibility entry point; use decision_fsdp_check.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("decision_fsdp_check.py")), run_name="__main__")
