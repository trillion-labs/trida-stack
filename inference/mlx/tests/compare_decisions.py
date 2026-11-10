"""Compare SGLang and MLX /v1/decide answers on identical JSONL request bodies.

Run once against bf16 MLX, then against q8 to measure drift. This client is standard
library only and can run separately from either serving environment.
"""

import argparse
import json
import math
import urllib.request
from pathlib import Path


def differences(expected, actual, atol, path="answers"):
    """Report contract mismatches and numerical errors beyond the chosen absolute tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            return [f"{path}: different keys"]
        return [
            e for k in expected for e in differences(expected[k], actual[k], atol, f"{path}.{k}")
        ]
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [f"{path}: different lengths"]
        return [
            e
            for i, (x, y) in enumerate(zip(expected, actual))
            for e in differences(x, y, atol, f"{path}[{i}]")
        ]
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isfinite(actual)
            or not math.isfinite(expected)
            or abs(expected - actual) > atol
        ):
            return [f"{path}: {expected!r} != {actual!r} (atol={atol})"]
        return []
    return (
        []
        if type(expected) is type(actual) and expected == actual
        else [f"{path}: {expected!r} != {actual!r}"]
    )


def post(url, body):
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/decide",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="SGLang server root URL")
    parser.add_argument("--candidate", required=True, help="MLX server root URL")
    parser.add_argument(
        "--requests", type=Path, required=True, help="JSONL with serve_decide.py request bodies"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.005,
        help="absolute answer tolerance, including rounded probabilities",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and non-negative")
    count, failed = 0, 0
    for line in args.requests.read_text().splitlines():
        if not line.strip():
            continue
        body = json.loads(line)
        if body.get("reasoning"):
            parser.error("parity requests must disable reasoning (generation is stochastic)")
        ref, candidate = post(args.reference, body), post(args.candidate, body)
        errors = differences(ref["answers"], candidate["answers"], args.atol)
        count += 1
        failed += bool(errors)
        print(json.dumps({"request": count, "passed": not errors, "differences": errors}))
    if count == 0:
        parser.error("--requests must contain at least one request")
    print(json.dumps({"requests": count, "failed": failed, "atol": args.atol}))
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
