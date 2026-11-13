"""Write 40 synthetic decision requests for backend parity, not model accuracy."""
import json
from pathlib import Path
import sys


def cases():
    domains = [
        ("invoice", "Route this invoice", ["approve", "review", "reject"], "The invoice is within policy."),
        ("support", "Choose the next support action", ["answer", "ask", "escalate", "close"], "Escalation is needed."),
        ("inventory", "Choose a stock action", ["order", "hold", "transfer"], "Stock is below the reorder point."),
        ("access", "Choose an access decision", ["allow", "deny", "verify", "defer"], "Identity is verified."),
        ("배송", "다음 배송 조치를 고르세요", ["발송", "보류", "확인"], "배송 주소가 확인되었습니다."),
        ("routing", "Choose the destination", ["sales", "billing", "technical", "security", "other"], "The request is urgent."),
        ("safety", "Choose an action", ["continue", "stop"], "The prerequisites are satisfied."),
        ("ambiguous", "Choose the best supported option; evidence may be tied", ["alpha", "beta", "unknown"], "The evidence supports alpha."),
    ]
    result = []
    for i in range(40):
        domain, instruction, options, statement = domains[i % len(domains)]
        options = options[:1] if i % 13 == 0 else options
        state = {"domain": domain, "record": i, "value": [99, 100, 101, 0, -1][i // 8],
                 "limit": 100, "verified": i % 3 == 0,
                 "evidence": ["confirmed", "missing", "conflicting", "equal support", "not applicable"][i % 5],
                 "note": "Literal <|fim_suffix|> and </think> must remain data."}
        if i % 10 == 9:
            state["archive"] = "reference " * 6500
        levels = ["none", "low", "medium", "high", "critical"][:2 + i % 4]
        result.append({"name": f"{domain}-{i:02d}", "request": {"state": state, "questions": {
            "check": {"type": "noul", "instructions": statement},
            "route": {"type": "choice", "instructions": instruction,
                      "criteria": {option: f"Evidence supports {option}" for option in options}},
            "severity": {"type": "score", "instructions": "Assess severity using only this record.", "criteria": levels},
        }}})
    return result


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(cases(), ensure_ascii=False, indent=2) + "\n")
