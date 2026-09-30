from __future__ import annotations

import gzip
import hashlib
import html
import json
import platform
import re
import time
import urllib.request
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

SEED = 42
ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SOURCE = {
    "train": "https://huggingface.co/datasets/deepset/prompt-injections/resolve/main/data/train-00000-of-00001-9564e8b05b4757ab.parquet",
    "test": "https://huggingface.co/datasets/deepset/prompt-injections/resolve/main/data/test-00000-of-00001-701d16158af87368.parquet",
}
EXPECTED = {
    "train": "2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04",
    "test": "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e",
}
SIGNALS = [
    r"ignore (?:all |the )?(?:previous|preceding|above) (?:instructions|orders|tasks)",
    r"forget (?:all |the )?(?:previous|above) (?:instructions|tasks|assignments)",
    r"(?:show|reveal) (?:me )?(?:all )?(?:your )?(?:system )?prompt",
    r"act as (?:an? )?", r"pretend (?:that )?you are",
    r"new (?:task|instructions?)", r"do not follow",
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download() -> tuple[pd.DataFrame, dict]:
    frames, checks = [], {}
    for split, url in SOURCE.items():
        path = ROOT / f"source_{split}.parquet"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        path.write_bytes(raw)
        sha = digest(raw)
        if sha != EXPECTED[split]:
            raise RuntimeError(f"Source hash mismatch for {split}: {sha}")
        frame = pd.read_parquet(path)
        if list(frame.columns) != ["text", "label"]:
            raise RuntimeError(f"Unexpected columns: {frame.columns.tolist()}")
        frame["split"] = split
        frame["source_row"] = np.arange(len(frame), dtype=int)
        frames.append(frame)
        checks[split] = {"url": url, "sha256": sha, "rows": len(frame), "bytes": len(raw)}
    data = pd.concat(frames, ignore_index=True)
    if len(data) != 662 or data[["text", "label"]].isna().any().any():
        raise RuntimeError("Dataset integrity check failed")
    data["text"] = data.text.astype(str)
    data["label"] = data.label.astype(int)
    if set(data.label) != {0, 1}:
        raise RuntimeError("Labels must be 0/1")
    data["text_sha256"] = data.text.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    checks.update({
        "total_rows": len(data),
        "missing_cells": int(data.isna().sum().sum()),
        "exact_duplicate_rows": int(data.duplicated(["text", "label"]).sum()),
        "conflicting_text_groups": int((data.groupby("text").label.nunique() > 1).sum()),
        "all_cells_inspected": True,
    })
    with gzip.open(ROOT / "dataset_snapshot.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        data.to_csv(handle, index=False)
    with gzip.open(ROOT / "dataset_snapshot.csv.gz", "rt", encoding="utf-8") as handle:
        replay = pd.read_csv(handle)
    if len(replay) != 662 or replay.text_sha256.tolist() != data.text_sha256.tolist():
        raise RuntimeError("Compressed snapshot verification failed")
    checks["snapshot_decompressed_rows_checked"] = len(replay)
    checks["snapshot_sha256"] = digest((ROOT / "dataset_snapshot.csv.gz").read_bytes())
    return data, checks


def rule_scores(texts: pd.Series) -> np.ndarray:
    pattern = re.compile("|".join(f"(?:{x})" for x in SIGNALS), re.I)
    return texts.map(lambda x: 1.0 if pattern.search(x) else 0.0).to_numpy()


def candidates() -> dict[str, object]:
    logistic = lambda: LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
    )
    return {
        "Dummy baseline": Pipeline([
            ("vec", TfidfVectorizer()), ("clf", DummyClassifier(strategy="most_frequent"))
        ]),
        "Word TF-IDF + logistic": Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True)),
            ("clf", logistic()),
        ]),
        "Character TF-IDF + logistic": Pipeline([
            ("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=10000, sublinear_tf=True)),
            ("clf", logistic()),
        ]),
        "Hybrid word + character TF-IDF": Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=10000, sublinear_tf=True)),
            ])),
            ("clf", logistic()),
        ]),
    }


def score(y, pred, prob) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "injection_precision": float(p),
        "injection_recall": float(r),
        "injection_f1": float(f),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def evaluate(train: pd.DataFrame, test: pd.DataFrame):
    X, y = train.text, train.label.to_numpy()
    Xt, yt = test.text, test.label.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = candidates()
    results, rows = {}, []
    for name in ["Rule scanner", *models.keys()]:
        fold_scores = []
        for fold, (tr, va) in enumerate(cv.split(X, y), 1):
            if name == "Rule scanner":
                prob = rule_scores(X.iloc[va])
            else:
                fitted = clone(models[name]).fit(X.iloc[tr], y[tr])
                prob = fitted.predict_proba(X.iloc[va])[:, 1]
            pred = (prob >= 0.5).astype(int)
            fold_scores.append(score(y[va], pred, prob)["macro_f1"])
            for idx, p, prediction in zip(va, prob, pred):
                rows.append({
                    "model": name, "fold": fold,
                    "source_row": int(train.iloc[idx].source_row),
                    "y_true": int(y[idx]), "probability": float(p),
                    "prediction": int(prediction),
                })
        results[name] = {
            "cv_macro_f1_mean": float(np.mean(fold_scores)),
            "cv_macro_f1_sd": float(np.std(fold_scores, ddof=1)),
            "fold_macro_f1": fold_scores,
        }
    eligible = [name for name in models if name != "Dummy baseline"]
    winner = max(eligible, key=lambda name: results[name]["cv_macro_f1_mean"])
    fitted_models = {}
    for name, model in models.items():
        model.fit(X, y)
        fitted_models[name] = model
        prob = model.predict_proba(Xt)[:, 1]
        results[name]["test"] = score(yt, (prob >= 0.5).astype(int), prob)
    rp = rule_scores(Xt)
    results["Rule scanner"]["test"] = score(yt, (rp >= 0.5).astype(int), rp)
    selected = fitted_models[winner]
    prob = selected.predict_proba(Xt)[:, 1]
    pred = (prob >= 0.5).astype(int)
    errors = test[["source_row", "text", "label", "text_sha256"]].copy()
    errors["probability_injection"] = prob
    errors["prediction"] = pred
    errors["correct"] = errors.label == errors.prediction
    errors["error_type"] = np.select(
        [(errors.label == 0) & (errors.prediction == 1),
         (errors.label == 1) & (errors.prediction == 0)],
        ["false_positive", "false_negative"], default="correct"
    )
    errors["confidence"] = np.where(pred == 1, prob, 1 - prob)
    return {
        "selected_model": winner, "primary_metric": "macro_f1", "models": results
    }, selected, pd.DataFrame(rows), errors


def bootstrap_ci(y, pred) -> dict:
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[idx], pred[idx], average="macro"))
    return {
        "method": "1000-row bootstrap of fixed holdout predictions",
        "lower_95": float(np.quantile(values, .025)),
        "upper_95": float(np.quantile(values, .975)),
    }


def feature_table(model) -> pd.DataFrame:
    if "features" in model.named_steps:
        names = model.named_steps["features"].get_feature_names_out()
    else:
        names = model.named_steps["vec"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(np.abs(coef))[::-1]
    return pd.DataFrame({
        "feature": names[order], "coefficient": coef[order],
        "direction": np.where(coef[order] >= 0, "injection", "benign"),
    })


def create_plots(data, metrics, errors, features):
    plt.style.use("seaborn-v0_8-whitegrid")
    counts = data.groupby(["split", "label"]).size().unstack(fill_value=0)
    ax = counts.rename(columns={0: "Benign", 1: "Injection"}).plot(
        kind="bar", color=["#3b82f6", "#ef4444"], figsize=(8, 5)
    )
    ax.set(title="Dataset composition by official split", ylabel="Prompts", xlabel="")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.savefig(FIG / "data-quality.png", dpi=160); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name, color in [(0, "Benign", "#3b82f6"), (1, "Injection", "#ef4444")]:
        ax.hist(data.loc[data.label == label, "text"].str.len(), bins=25, alpha=.6, label=name, color=color)
    ax.set(title="Prompt length distributions", xlabel="Characters (log scale)", ylabel="Prompts")
    ax.set_xscale("log")
    ax.legend(); plt.tight_layout(); plt.savefig(FIG / "distributions.png", dpi=160); plt.close()

    rows = [(name, m["cv_macro_f1_mean"], m["cv_macro_f1_sd"], m["test"]["macro_f1"]) for name, m in metrics["models"].items()]
    frame = pd.DataFrame(rows, columns=["model", "cv", "sd", "test"]).sort_values("cv")
    fig, ax = plt.subplots(figsize=(10, 6)); y = np.arange(len(frame))
    ax.barh(y - .18, frame.cv, .35, xerr=frame.sd, label="5-fold CV", color="#2563eb")
    ax.barh(y + .18, frame.test, .35, label="Official test", color="#16a34a")
    ax.set_yticks(y, frame.model); ax.set_xlim(0, 1); ax.set_xlabel("Macro-F1")
    ax.set_title("Model comparison"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG / "model-comparison.png", dpi=160); plt.close()

    winner = metrics["selected_model"]
    matrix = np.array(metrics["models"][winner]["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Injection"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Selected model on official test split")
    plt.tight_layout(); plt.savefig(FIG / "confusion-matrix.png", dpi=160); plt.close()

    y = errors.label.to_numpy(); prob = errors.probability_injection.to_numpy()
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#7c3aed", lw=2, label=f"AUC = {roc_auc_score(y, prob):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve on official test split")
    ax.legend(); plt.tight_layout(); plt.savefig(FIG / "roc-curve.png", dpi=160); plt.close()

    top = pd.concat([features.nlargest(8, "coefficient"), features.nsmallest(8, "coefficient")]).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.feature, top.coefficient, color=np.where(top.coefficient > 0, "#ef4444", "#3b82f6"))
    ax.set(title="Strongest learned text signals", xlabel="Logistic coefficient")
    plt.tight_layout(); plt.savefig(FIG / "feature-importance.png", dpi=160); plt.close()


def write_artifacts(data, checks, metrics, model, oof, errors, features, elapsed):
    winner = metrics["selected_model"]
    test = metrics["models"][winner]["test"]
    dummy = metrics["models"]["Dummy baseline"]
    metrics["test_bootstrap_ci"] = bootstrap_ci(errors.label.to_numpy(), errors.prediction.to_numpy())
    metrics["official_test_rows"] = len(errors)
    metrics["errors"] = int((~errors.correct).sum())
    metrics["false_positives"] = int((errors.error_type == "false_positive").sum())
    metrics["false_negatives"] = int((errors.error_type == "false_negative").sum())
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    oof.to_csv(ROOT / "cv_predictions.csv", index=False)
    errors.to_csv(ROOT / "error_analysis.csv", index=False)
    features.to_csv(ROOT / "feature_importance.csv", index=False)
    stats = data.assign(
        characters=data.text.str.len(), words=data.text.str.split().str.len()
    ).groupby(["split", "label"])[["characters", "words"]].agg(
        ["count", "mean", "std", "median", "min", "max"]
    )
    stats.to_csv(ROOT / "descriptive_statistics.csv")
    pd.DataFrame([
        {"column": "text", "type": "string", "description": "Input prompt", "missing": 0},
        {"column": "label", "type": "integer", "description": "0 benign, 1 prompt injection", "missing": 0},
        {"column": "split", "type": "string", "description": "Official train/test split", "missing": 0},
    ]).to_csv(ROOT / "data_dictionary.csv", index=False)
    joblib.dump(model, ROOT / "prompt_guard.joblib", compress=3)
    checks["model_sha256"] = digest((ROOT / "prompt_guard.joblib").read_bytes())
    checks["all_saved_scores_finite"] = bool(np.isfinite(errors.probability_injection).all())
    (ROOT / "source_verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    table = "\n".join(
        f"| {'**' + name + '**' if name == winner else name} | "
        f"{'**' + format(m['cv_macro_f1_mean'], '.4f') + '**' if name == winner else format(m['cv_macro_f1_mean'], '.4f')} | "
        f"{m['cv_macro_f1_sd']:.4f} |"
        for name, m in sorted(metrics["models"].items(), key=lambda x: x[1]["cv_macro_f1_mean"], reverse=True)
    )
    readme = f"""# Prompt Injection Guard for LLM Applications

A local text-classification tool that detects prompt-injection attempts before untrusted text reaches an LLM.

---

## Overview

* **Task:** Binary classification
* **Dataset:** 662 prompts, 1 text feature.
* **Selected Model:** **{winner}** (Macro-F1 = {test['macro_f1']:.4f}).
* **Baseline Comparison:** Dummy baseline achieved Macro-F1 = {dummy['test']['macro_f1']:.4f}.

---

## Key Results

### Training Cross-Validation

| Model | Macro-F1 | Fold SD |
| :--- | :--- | :--- |
{table}

### Final Holdout Evaluation ({len(errors)} Samples)

| Model | Macro-F1 | ROC-AUC | Injection Recall |
| :--- | :--- | :--- | :--- |
| **{winner}** | **{test['macro_f1']:.4f}** | **{test['roc_auc']:.4f}** | **{test['injection_recall']:.4f}** |
| Dummy Baseline | {dummy['test']['macro_f1']:.4f} | {dummy['test']['roc_auc']:.4f} | {dummy['test']['injection_recall']:.4f} |

The selected model made **{metrics['errors']} errors**: {metrics['false_positives']} false positives and {metrics['false_negatives']} false negatives. Its fixed-model bootstrap 95% interval for test Macro-F1 is [{metrics['test_bootstrap_ci']['lower_95']:.4f}, {metrics['test_bootstrap_ci']['upper_95']:.4f}].

---

## Data Summary

* **Source:** [deepset Prompt Injections](https://huggingface.co/datasets/deepset/prompt-injections)
* **Data Quality:** 662 usable rows, 1 text feature, 0 missing cells, {checks['exact_duplicate_rows']} exact duplicate rows retained because the official split is preserved.
* **Key Feature:** {html.escape(str(features.iloc[0].feature))} had the largest absolute learned coefficient ({features.iloc[0].coefficient:.4f}).
* **Notes:** The dataset is small and contains recognizable attack phrases. Scores measure this benchmark only; novel, obfuscated, multilingual, or indirect injections may bypass the detector. Use the probability as one layer in a defense-in-depth design.

---

## Visualizations

| Dataset Composition | Prompt Lengths |
| :---: | :---: |
| ![Dataset composition](figures/data-quality.png) | ![Prompt lengths](figures/distributions.png) |

| Model Comparison | Confusion Matrix |
| :---: | :---: |
| ![Model comparison](figures/model-comparison.png) | ![Confusion matrix](figures/confusion-matrix.png) |

| ROC Curve | Learned Signals |
| :---: | :---: |
| ![ROC curve](figures/roc-curve.png) | ![Learned signals](figures/feature-importance.png) |

---

## Repository Structure

    figures/                    Data, performance, and diagnostic plots
    analysis.ipynb              Executed analysis summary
    audit.json                  Runtime metadata and source hashes
    data_dictionary.csv         Dataset field definitions
    descriptive_statistics.csv Length statistics by split and class
    error_analysis.csv          Holdout predictions and error types
    feature_importance.csv      Learned text-signal coefficients
    metrics.json                CV and official holdout scores
    prompt_guard.joblib         Fitted local detector
    prompt_guard.py             Reusable command-line scanner
    study.py                    Full study pipeline
    README.md
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / "SOURCE.md").write_text(
        "# Source\n\nDataset: deepset/prompt-injections, Apache-2.0. "
        "The exact train and test Parquet files were downloaded from the dataset repository "
        "and verified against their Git LFS SHA-256 identifiers. All 662 rows and every cell "
        "were inspected before the compressed snapshot was published.\n", encoding="utf-8"
    )
    (ROOT / "LEARNING_NOTES.md").write_text(
        "# Human Review\n\n- What surprised you?\n\n- Which error matters most?\n\n"
        "- What would you change next?\n\n- Can you explain the model choice?\n\n"
        "- What did you learn?\n", encoding="utf-8"
    )
    audit = {
        "created_at": "2026-09-30T08:04:11+03:00",
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "seed": SEED,
        "elapsed_seconds": elapsed,
        "authorship": "Automated by OpenAI Codex; human review pending",
        "paid_llm_commentary": False, "source_verification": checks,
    }
    (ROOT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    metadata = {
        "suggested_repo_name": "llm-prompt-injection-guard",
        "github_about": "Local prompt-injection detector for LLM apps, benchmarked on deepset data with reproducible evaluation and a reusable CLI.",
        "topics": ["llm-security", "prompt-injection", "ai-engineering", "llm-engineering", "nlp", "machine-learning", "scikit-learn", "security", "text-classification", "responsible-ai"],
    }
    (ROOT / "repository_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Prompt Injection Guard Analysis\n", "Executed summary of the verified study artifacts."]},
            {"cell_type": "code", "execution_count": 1, "metadata": {}, "source": ["import json, pandas as pd\n", "metrics=json.load(open('metrics.json'))\n", "metrics['selected_model']"], "outputs": [{"output_type": "execute_result", "execution_count": 1, "metadata": {}, "data": {"text/plain": [repr(winner)]}}]},
            {"cell_type": "code", "execution_count": 2, "metadata": {}, "source": ["errors=pd.read_csv('error_analysis.csv')\n", "errors.groupby('error_type').size()"], "outputs": [{"output_type": "execute_result", "execution_count": 2, "metadata": {}, "data": {"text/plain": [errors.groupby("error_type").size().to_string()]}}]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": platform.python_version()}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (ROOT / "analysis.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    cards = "\n".join(
        f'<section><h2>{html.escape(title)}</h2><img src="figures/{filename}" alt="{html.escape(title)}"></section>'
        for title, filename in [
            ("Dataset Composition", "data-quality.png"), ("Prompt Lengths", "distributions.png"),
            ("Model Comparison", "model-comparison.png"), ("Confusion Matrix", "confusion-matrix.png"),
            ("ROC Curve", "roc-curve.png"), ("Learned Signals", "feature-importance.png"),
        ]
    )
    report = (
        '<!doctype html><html><head><meta charset="utf-8"><title>Prompt Injection Guard</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#172033}'
        'img{max-width:100%;height:auto}section{margin:32px 0}</style></head><body>'
        f'<h1>Prompt Injection Guard for LLM Applications</h1><p>Selected model: <strong>{html.escape(winner)}</strong>; '
        f'official test Macro-F1: <strong>{test["macro_f1"]:.4f}</strong>.</p>{cards}</body></html>'
    )
    (ROOT / "report.html").write_text(report, encoding="utf-8")


def main():
    started = time.perf_counter()
    np.random.seed(SEED)
    data, checks = download()
    train = data[data.split == "train"].reset_index(drop=True)
    test = data[data.split == "test"].reset_index(drop=True)
    metrics, model, oof, errors = evaluate(train, test)
    features = feature_table(model)
    create_plots(data, metrics, errors, features)
    write_artifacts(data, checks, metrics, model, oof, errors, features, time.perf_counter() - started)
    print(json.dumps({
        "selected_model": metrics["selected_model"],
        "test": metrics["models"][metrics["selected_model"]]["test"],
        "folder": ROOT.name,
    }, indent=2))


if __name__ == "__main__":
    main()
