from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib

SIGNALS = [
    r"ignore .*previous",
    r"forget .*instructions",
    r"(?:show|reveal).*prompt",
    r"new (?:task|instructions?)",
]


class PromptInjectionGuard:
    def __init__(self, model_path: str | Path | None = None, threshold: float = 0.5):
        self.model_path = Path(model_path or Path(__file__).with_name("prompt_guard.joblib"))
        self.model = joblib.load(self.model_path)
        self.threshold = threshold

    def scan(self, text: str) -> dict:
        probability = float(self.model.predict_proba([text])[0, 1])
        matched = [pattern for pattern in SIGNALS if re.search(pattern, text, re.I)]
        return {
            "label": "prompt_injection" if probability >= self.threshold else "benign",
            "probability": round(probability, 6),
            "threshold": self.threshold,
            "matched_rule_signals": matched,
        }


def main():
    parser = argparse.ArgumentParser(description="Scan text for likely LLM prompt injection")
    parser.add_argument("text", nargs="?", help="Text to scan; reads stdin when omitted")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("Provide text as an argument or via stdin")
    print(json.dumps(PromptInjectionGuard(threshold=args.threshold).scan(text), indent=2))


if __name__ == "__main__":
    main()
