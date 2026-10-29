# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""jev.encode_rows against golden token rows from the training stack's encoder at commit 951c21c
(fixtures/jev_golden/PROVENANCE.md). Needs transformers 5.x (the tokenizer config at the pinned revision does not load
under 4.57.6; checked with 5.12.1) and the pinned tokenizer in the local Hugging Face cache. It never downloads (no
network in tests), so it is not in CI. The model repo is gated: accept its terms on Hugging Face, log in, and fetch
the tokenizer at the pinned revision once:

    hf auth login
    hf download trillionlabs/Trida2.1-4B --revision a64a0737a79921991a769a99b06872b883f455d3 --include "tokenizer*"

then run: pytest -q inference/zoro-1/test_jev_golden.py
"""
import json
from pathlib import Path

import pytest

import jev

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "jev_golden" / "golden.json").read_text(encoding="utf-8"))
CASES = GOLDEN["cases"]
TOKENIZER_REPO = "trillionlabs/Trida2.1-4B"
TOKENIZER_REVISION = "a64a0737a79921991a769a99b06872b883f455d3"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION, local_files_only=True)
    except OSError as e:
        raise RuntimeError(f"{TOKENIZER_REPO} tokenizer at revision {TOKENIZER_REVISION} is not in the local Hugging "
                           f"Face cache; see this file's docstring for the download command") from e


def test_fixtures_name_the_pinned_tokenizer():
    assert GOLDEN["provenance"]["tokenizer"] == f"{TOKENIZER_REPO} at revision {TOKENIZER_REVISION}"


def test_fixtures_cover_the_catalog():
    # Every question type, a soft target, images, the state cut and the branch budget, at both the default limits
    # and the demo's training_context(7552).
    assert sorted(CASES) == sorted([
        "korean_text", "emoji", "nested_objects", "list_state", "special_token_text", "long_option_list", "soft_target",
        "korean_text_default_limits", "emoji_default_limits", "nested_objects_default_limits",
        "list_state_default_limits", "special_token_text_default_limits", "long_option_list_default_limits",
        "soft_target_default_limits",
        "state_text_383_tokens", "state_text_384_tokens",
        "image_4_text_20_tokens", "image_64_text_317_tokens", "image_64_text_318_tokens", "image_380_text_5_tokens",
        "image_381_text_5_tokens", "image_256_korean_demo_limits",
        "branch_budget_plus_0", "branch_budget_plus_1"])
    qtypes = {q["type"] for c in CASES.values() for q in c["request"]["questions"].values()}
    assert qtypes == {"noul", "choice", "score"}


@pytest.mark.parametrize("name", sorted(CASES))
def test_rows_match_golden(tok, name):
    case = CASES[name]
    rec = jev.materialize(case["request"])
    if "raises" in case:
        with pytest.raises(jev.ContextOverflow) as e:
            jev.encode_rows(tok, rec, **case["kwargs"])
        assert str(e.value) == case["message"]
    else:
        assert jev.encode_rows(tok, rec, **case["kwargs"]) == case["rows"]
