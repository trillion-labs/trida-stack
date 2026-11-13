"""A package exception applies only to the advisory and aliases it names."""
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("check_advisories", Path(__file__).with_name("check_advisories.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_one_exception_does_not_cover_another_advisory(tmp_path, monkeypatch, capsys):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"dependencies": [{"name": "example", "vulns": [
        {"id": "old", "aliases": ["old-alias"]}, {"id": "new", "aliases": []}]}]}))
    monkeypatch.setattr(gate, "load_exceptions", lambda: ({"old-alias": {}}, []))
    monkeypatch.setattr(sys, "argv", ["check_advisories", "--pip-audit", str(audit)])
    assert gate.main() == 1
    assert "example: new" in capsys.readouterr().out


def test_alias_exception_covers_same_advisory(tmp_path, monkeypatch):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"dependencies": [{"name": "example", "vulns": [
        {"id": "primary", "aliases": ["alias"]}]}]}))
    monkeypatch.setattr(gate, "load_exceptions", lambda: ({"alias": {}}, []))
    monkeypatch.setattr(sys, "argv", ["check_advisories", "--pip-audit", str(audit)])
    assert gate.main() == 0
