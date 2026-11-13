"""COMPLIANCE.md section 4b against Hugging Face metadata — offline (the live fetch runs in model-card.yml)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("check_model_card", Path(__file__).parent / "check_model_card.py")
cmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmc)

TABLE = """
| | component | licence | access | verified |
|---|---|---|---|---|
| base | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | **Apache-2.0** | public | HF model metadata |
| derived | [trillionlabs/Zoro-1](https://huggingface.co/trillionlabs/Zoro-1) | **Apache-2.0** | gated | HF model metadata |
| derived | [trillionlabs/Zoro](https://huggingface.co/trillionlabs/Zoro) (was Trida2.0-4B) | **Apache-2.0** | public | HF |
"""


def hf(licence="apache-2.0", gated=False, private=False):
    return {"cardData": {"license": licence}, "gated": gated, "private": private}


def test_rows_are_read_per_model():
    rows = cmc.model_rows(TABLE)
    assert rows["trillionlabs/Zoro-1"] == {"licence": "Apache-2.0", "access": "gated"}
    assert rows["trillionlabs/Zoro"] == {"licence": "Apache-2.0", "access": "public"}
    assert rows["Qwen/Qwen3.5-4B"]["access"] == "public"


def test_matching_metadata_passes():
    rows = cmc.model_rows(TABLE)
    assert cmc.check("trillionlabs/Zoro-1", hf(gated="auto"), rows) == []
    assert cmc.check("trillionlabs/Zoro", hf(), rows) == []


def test_each_kind_of_drift_is_reported():
    rows = cmc.model_rows(TABLE)
    assert any("access" in p for p in cmc.check("trillionlabs/Zoro", hf(gated="manual"), rows))
    assert any("access" in p for p in cmc.check("trillionlabs/Zoro-1", hf(gated=False), rows))
    assert any("access" in p for p in cmc.check("trillionlabs/Zoro-1", hf(private=True), rows))
    assert any("licence" in p for p in cmc.check("trillionlabs/Zoro-1", hf(licence="other", gated="auto"), rows))
    assert any("no row" in p for p in cmc.check("trillionlabs/Unlisted", hf(), rows))


def test_only_section_4b_is_read():
    doc = ("## 4. Datasets\n| derived | [trillionlabs/Zoro-1](x) | **MIT** | public | elsewhere |\n"
           "## 4b. Model weights\n" + TABLE + "\n## 5. License summary\n"
           "| derived | [trillionlabs/Zoro](x) | **MIT** | private | later |\n")
    rows = cmc.model_rows(cmc.section_4b(doc))
    assert rows["trillionlabs/Zoro-1"]["access"] == "gated" and rows["trillionlabs/Zoro"]["access"] == "public"


def test_compliance_md_lists_every_watched_model():
    rows = cmc.model_rows(cmc.section_4b(cmc.COMPLIANCE.read_text()))
    assert set(cmc.MODELS) <= set(rows), "add a section 4b row for each model in MODELS"


def test_every_derived_row_is_watched():
    # a derived checkpoint listed in section 4b but missing from MODELS is never re-checked weekly
    import re
    section = cmc.section_4b(cmc.COMPLIANCE.read_text())
    derived = set(re.findall(r"^\|\s*derived\s*\|\s*\[([^\]]+)\]", section, re.M))
    assert derived <= set(cmc.MODELS), f"add to MODELS: {sorted(derived - set(cmc.MODELS))}"


def test_mit_models_match_their_row():
    rows = cmc.model_rows("| derived | [trillionlabs/GravityOCR](x) | **MIT** | public | HF |")
    assert cmc.check("trillionlabs/GravityOCR", hf(licence="mit"), rows) == []
