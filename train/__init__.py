# Optional: load a local .env if python-dotenv is installed. Only a missing dotenv is
# tolerated, so importing `train` (e.g. `train.py --help`) never fails for that reason.
import contextlib

with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    load_dotenv()
