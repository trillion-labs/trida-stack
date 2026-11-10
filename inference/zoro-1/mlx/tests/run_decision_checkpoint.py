"""Opt-in trained-checkpoint E2E: raw MLX HTTP, q8 conversion/HTTP, optional CUDA reference.

Run with the MLX environment's Python, passing an accessible local decision checkpoint.
No model downloads or credentials are handled by this runner.
"""

import argparse
import base64
import io
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from compare_decisions import differences


def cases():
    image = io.BytesIO()
    Image.new("RGB", (96, 96), (255, 0, 0)).save(image, format="PNG")
    return {
        "text": {
            "request": {
                "state": "Two plus two equals four.",
                "questions": {
                    "correct": {"type": "noul", "instructions": "Is the statement correct?"},
                    "result": {
                        "type": "choice",
                        "instructions": "What is two plus two?",
                        "criteria": {"four": "4", "five": "5", "six": "6"},
                    },
                    "certainty": {
                        "type": "score",
                        "instructions": "How certain is the statement?",
                        "criteria": ["false", "uncertain", "certain"],
                    },
                },
            }
        },
        "image": {
            "request": {
                "state": "Look at this picture.",
                "image": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode(),
                "questions": {
                    "color": {
                        "type": "choice",
                        "instructions": "What is the dominant color?",
                        "criteria": {"red": "red", "blue": "blue", "green": "green"},
                    }
                },
            }
        },
    }


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def post(root, route, body):
    request = urllib.request.Request(
        root.rstrip("/") + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def run_server(model, output, name):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = f"http://127.0.0.1:{port}"
    with (output / f"{name}-server.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "trida_mlx.server",
                "--model",
                str(model),
                "--port",
                str(port),
                "--context-length",
                "16384",
            ],
            stdout=log,
            stderr=log,
        )
        try:
            for _ in range(1800):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} server exited; see {log.name}")
                try:
                    card = get(root + "/v1/models")["data"][0]
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            else:
                raise TimeoutError(f"{name} server did not start")
            assert card["decision"] and card["supports_vision"], card
            result = {"model": str(model), "card": card, "decisions": {}}
            for label, body in cases().items():
                response = post(root, "/v1/decide", body)
                expected_ids = body["request"]["questions"].keys()
                assert response["answers"].keys() == expected_ids, response
                result["decisions"][label] = response
                print(f"{name}: {label} request passed", flush=True)
            body = {**cases()["text"], "reasoning": True, "max_reasoning_tokens": 768}
            result["reasoning"] = post(root, "/v1/decide", body)
            assert isinstance(result["reasoning"]["reasoning"], str)
            assert "reasoning_ms" in result["reasoning"]
            result["reasoning_stats"] = get(root + "/stats")["last"]
            assert result["reasoning"]["reasoning"].strip()
            assert result["reasoning_stats"]["new_tokens"] < 768, (
                "reasoning exhausted its token budget"
            )
            again = post(root, "/v1/decide", cases()["text"])
            assert not differences(result["decisions"]["text"]["answers"], again["answers"], 0.0002)
            result["chat"] = post(
                root,
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 32,
                    "enable_thinking": False,
                    "temperature": 0,
                    "trida_mode": "causal",
                },
            )
            assert result["chat"]["usage"]["completion_tokens"] > 0
            assert result["chat"]["choices"][0]["message"]["content"].strip()
            assert result["chat"]["choices"][0]["finish_reason"] == "stop"
            print(f"{name}: reasoning, repeat decision and chat passed", flush=True)
            return result
        finally:
            process.terminate()
            process.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", help="optional SGLang server root URL")
    parser.add_argument(
        "--atol", type=float, default=0.01, help="absolute q8/CUDA answer tolerance"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "atol": args.atol}
    try:
        report["raw"] = run_server(args.model, args.output, "raw")
        q8 = args.output / "q8"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "trida_mlx.convert",
                "--model",
                str(args.model),
                "--out",
                str(q8),
                "--bits",
                "8",
            ],
            check=True,
        )
        report["q8"] = run_server(q8, args.output, "q8")
        report["comparisons"] = {}
        for label, body in cases().items():
            raw = report["raw"]["decisions"][label]["answers"]
            quantized = report["q8"]["decisions"][label]["answers"]
            report["comparisons"][f"q8/{label}"] = differences(raw, quantized, args.atol)
            if args.reference:
                reference = post(args.reference, "/v1/decide", body)["answers"]
                report["comparisons"][f"cuda/{label}"] = differences(reference, raw, args.atol)
        report["cuda_parity_tested"] = bool(args.reference)
        report["status"] = "failed" if any(report["comparisons"].values()) else "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json")}))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
