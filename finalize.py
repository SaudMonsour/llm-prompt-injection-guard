from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


metrics = json.loads((ROOT / "metrics.json").read_text())
source = json.loads((ROOT / "source_verification.json").read_text())
verification = json.loads((ROOT / "verification.json").read_text())

selection = {
    "protocol": "Five-fold stratified cross-validation on the official training split; model family selected by mean Macro-F1. The official test split was not used for selection.",
    "seed": 42,
    "selected_model": metrics["selected_model"],
    "primary_metric": metrics["primary_metric"],
    "candidates": {
        name: {
            "cv_macro_f1_mean": result["cv_macro_f1_mean"],
            "cv_macro_f1_sd": result["cv_macro_f1_sd"],
        }
        for name, result in metrics["models"].items()
    },
    "decision_threshold": 0.5,
}
(ROOT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

execution = {
    "status": "completed",
    "date": "2026-09-30",
    "entry_point": "study.py",
    "verification_entry_point": "verify.py",
    "framework": "scikit-learn",
    "paid_llm_calls": 0,
    "human_review": "pending",
    "canonical_parent_commit": "07fc1aa040de8e8cd0aab89e48594bc968676b43",
    "canonical_agent_result": "no_project",
    "reproducibility": {
        "source_hashes_verified": True,
        "snapshot_rows_checked": verification["snapshot_rows_checked"],
        "holdout_predictions_recalculated": verification["prediction_rows_recalculated"],
        "prediction_max_absolute_difference": verification["prediction_max_absolute_difference"],
    },
}
(ROOT / "execution.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")

exclude = {"project.json", "__pycache__"}
artifacts = {}
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path.name in exclude or "__pycache__" in path.parts:
        continue
    artifacts[str(path.relative_to(ROOT))] = sha256(path)

project = {
    "schema_version": 1,
    "status": "complete",
    "created_at": "2026-09-30T08:04:11+03:00",
    "published_at": "2026-09-30T08:19:19+03:00",
    "date": "2026-09-30",
    "dataset_id": "deepset-prompt-injections",
    "dataset_family": "prompt-injection",
    "title": "Prompt Injection Guard for LLM Applications",
    "task": "classification",
    "folder": "2026-09-30-prompt-injection-guard",
    "data_sha256": "b67c5329fd2d36c6d6ee3931e1b49c45f1a388e180d2fea3b4e95b006d3dd82d",
    "primary_result": "macro_f1: 0.8699",
    "framework": "scikit-learn",
    "selected_model": metrics["selected_model"],
    "dataset_spec": {
        "source": "https://huggingface.co/datasets/deepset/prompt-injections",
        "license": "Apache-2.0",
        "rows": 662,
        "official_train_rows": source["train"]["rows"],
        "official_test_rows": source["test"]["rows"],
        "question": "Can a lightweight local classifier detect prompt-injection attempts before untrusted text reaches an LLM?",
        "caveat": "Small English benchmark with recognizable attack phrases; the detector is one defense layer and does not establish universal prompt-injection robustness.",
    },
    "config": {
        "seed": 42,
        "cv_folds": 5,
        "selection_metric": "macro_f1",
        "decision_threshold": 0.5,
        "test_policy": "Official test split evaluated after cross-validation selection",
    },
    "runner": "study.py",
    "validation_runner": "verify.py",
    "tool_entry_point": "prompt_guard.py",
    "authorship": "Automated by OpenAI Codex; human review pending",
    "ai_review": "disabled",
    "verification": "All 662 snapshot rows and source hashes checked; all 116 holdout predictions independently recalculated; six figures inspected.",
    "artifact_sha256": artifacts,
}
(ROOT / "project.json").write_text(json.dumps(project, indent=2), encoding="utf-8")
print(json.dumps({"artifacts": len(artifacts), "project": project["folder"]}, indent=2))
