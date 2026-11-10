"""Run with .upstream/.venv/bin/python; no model, GPU or network required."""
import json
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".upstream"))

import httpx  # noqa: E402
from core.pricing import cost_usd  # noqa: E402
from core.providers import configuration, make_client, resolve_model  # noqa: E402
from core.question_types import Choice, Noul, Score, response_record  # noqa: E402


class TridaProviderTest(unittest.TestCase):
    def setUp(self):
        self.config = configuration("trida:zoro-1-q8", base_url="http://trida.test:30001")
        self.client = make_client(self.config)
        self.addCleanup(self.client.close)
        self.questions = {
            "gate": Noul(instructions="Act?"),
            "route": Choice(instructions="Where?", criteria={"a": "Team A", "b": "Team B"}),
            "level": Score(instructions="Urgency?", criteria=["Low", "High"]),
        }
        self.answers = {
            "gate": {"type": "noul", "noul": 0.8},
            "route": {"type": "choice", "choice": "b", "confidence": 0.4,
                      "probabilities": {"a": 0.3, "b": 0.7}},
            "level": {"type": "score", "score": 0.7, "confidence": 0.4,
                      "legend": ["Low", "High"], "probabilities": {"0": 0.3, "1": 0.7}},
        }

    def respond(self, status=200, payload=None):
        def handler(request):
            self.request = request
            return httpx.Response(status, json=payload if payload is not None else {"answers": self.answers})
        self.client.client.close()
        self.client.client = httpx.Client(
            base_url=self.config.base_url + "/", transport=httpx.MockTransport(handler)
        )

    def ask(self):
        return self.client.system_one(model=resolve_model(self.config), state={"ticket": "Help"},
                                      questions=self.questions)

    def test_request_and_all_answer_types(self):
        self.respond()
        response = self.ask()
        self.assertEqual(self.request.url.path, "/v1/decide")
        body = json.loads(self.request.content)
        self.assertFalse(body["reasoning"])
        self.assertEqual(body["request"]["state"], {"ticket": "Help"})
        self.assertEqual(body["request"]["questions"]["route"],
                         {"type": "choice", "instructions": "Where?",
                          "criteria": {"a": "Team A", "b": "Team B"}})
        self.assertEqual(response.answers["route"].probabilities, {"a": 0.3, "b": 0.7})
        self.assertEqual(response.answers["gate"].noul, 0.8)
        self.assertEqual(response.answers["level"].probabilities, {0: 0.3, 1: 0.7})
        self.assertEqual(response.answers["level"].legend, {0: "Low", 1: "High"})
        self.assertIsNone(response.usage.input_tokens)
        self.assertIsNone(response.usage.output_tokens)

    def test_identity_and_unknown_cost(self):
        self.assertEqual(self.config.identity.name, "trida:zoro-1-q8")
        self.assertEqual(self.config.identity.lab, "other")
        self.assertEqual(self.config.question_mode, "all")
        self.assertEqual(self.config.thinking, "off")
        self.assertIsNone(cost_usd({}, self.config.identity.name, 100, 0))

    def test_missing_endpoint_or_reasoning_rejected(self):
        with self.assertRaises(ValueError):
            configuration("trida:zoro-1")
        with self.assertRaises(ValueError):
            configuration("trida:zoro-1", thinking="high", base_url=self.config.base_url)

    def test_server_failure_includes_reason(self):
        self.respond(400, {"error": "context exceeded"})
        with self.assertRaisesRegex(httpx.HTTPStatusError, "context exceeded"):
            self.ask()

    def test_missing_question_rejected(self):
        del self.answers["gate"]
        self.respond()
        with self.assertRaisesRegex(ValueError, "question IDs"):
            self.ask()

    def test_invalid_probabilities_rejected(self):
        for probabilities in ({"a": 0.2, "b": 0.2}, {"a": -0.1, "b": 1.1},
                              {"a": True, "b": 0.0}, {"a": 0.3, "c": 0.7}):
            with self.subTest(probabilities=probabilities):
                self.answers["route"]["probabilities"] = probabilities
                self.respond()
                with self.assertRaises(ValueError):
                    self.ask()

    def test_answer_type_mismatch_rejected(self):
        self.answers["gate"]["type"] = "choice"
        self.respond()
        with self.assertRaisesRegex(ValueError, "Answer type"):
            self.ask()

    def test_reasoning_request_metadata_and_trace(self):
        config = configuration("trida:zoro-1-q8", base_url=self.config.base_url,
                               trida_reasoning=True, max_output_tokens=2048)
        self.client.close()
        self.client = make_client(config)
        self.addCleanup(self.client.close)
        self.respond(payload={"answers": self.answers, "reasoning": "Check the evidence.",
                              "reasoning_stats": {"finish_reason": "stop"},
                              "reasoning_ms": 123.0})
        response = self.ask()
        body = json.loads(self.request.content)
        self.assertTrue(body["reasoning"])
        self.assertEqual(body["max_reasoning_tokens"], 2048)
        self.assertEqual(config.identity.effort, "reasoning")
        self.assertEqual(self.config.identity.effort, "off")
        self.assertEqual(response_record(response)["decision_details"]["reasoning"],
                         "Check the evidence.")
        self.assertEqual(response.answers["route"].choice, "b")

    def test_reasoning_fallback_is_not_scored_as_reasoned(self):
        self.client.reasoning = True
        for extra in ({}, {"reasoning": "Unverified"},
                      {"reasoning": "Unfinished", "cut": "length",
                            "reasoning_stats": {"finish_reason": "length"}},
                      {"reasoning": "Unfinished", "reasoning_stats": {"finish_reason": "length"}}):
            with self.subTest(extra=extra):
                self.respond(payload={"answers": self.answers, **extra})
                with self.assertRaisesRegex(ValueError, "Reasoning did not complete"):
                    self.ask()

    def test_reasoning_budget_defaults_and_ceiling(self):
        config = configuration("trida:zoro-1", base_url=self.config.base_url, trida_reasoning=True)
        self.assertEqual(config.max_output_tokens, 2048)
        self.assertEqual(configuration("openai:model").max_output_tokens, 64000)
        with self.assertRaisesRegex(ValueError, "4096"):
            configuration("trida:zoro-1", base_url=self.config.base_url,
                          trida_reasoning=True, max_output_tokens=64000)

    def test_reasoning_rejected_for_other_providers(self):
        with self.assertRaisesRegex(ValueError, "Trida only"):
            configuration("typesafe:jev-1.13.0", trida_reasoning=True)

    def test_setup_ignores_git_diff_display_settings(self):
        settings = {"diff.noprefix": "true", "diff.mnemonicPrefix": "true",
                    "core.abbrev": "5", "diff.external": "false"}
        env = {**os.environ, "GIT_CONFIG_COUNT": str(len(settings))}
        for i, (key, value) in enumerate(settings.items()):
            env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"] = key, value
        result = subprocess.run([str(HERE / "setup.sh"), "--check"], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_setup_rejects_untracked_files(self):
        with tempfile.NamedTemporaryFile(dir=HERE / ".upstream", prefix="unexpected-"):
            result = subprocess.run([str(HERE / "setup.sh"), "--check"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("untracked files", result.stderr)

    def test_pinned_repositories_match_upstream(self):
        from core.dataset import DATASETS
        pins = json.loads((HERE / "dataset-revisions.json").read_text())
        self.assertEqual({key: value["repo"] for key, value in pins.items()}, DATASETS)

    def test_unlicensed_datasets_are_blocked_before_download(self):
        for workflow in ["customer_service", "security_incidents", "agent_trace_observability"]:
            with self.subTest(workflow=workflow):
                result = subprocess.run([str(HERE / "run.sh"), workflow,
                    "--model", "trida:zoro-1", "--base-url", "http://unused", "--dry-run"],
                    capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("License not verified", result.stderr)

    def test_patch_only_changes_connectivity(self):
        paths = [line.split()[2][2:] for line in (HERE / "patches/trida-provider.patch").read_text().splitlines()
                 if line.startswith("diff --git ")]
        self.assertEqual(paths, ["core/providers.py", "core/trida_client.py", "run.py"])


if __name__ == "__main__":
    unittest.main()
