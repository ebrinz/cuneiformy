"""
Ridge alpha sweep on the whitened-gloss Gemma target.

One-shot diagnostic: does adjusting ridge regularization get us the
remaining 0.5pp to clear the +3pp phase-A gate, or is 19.85% at
alpha=100 already near the ceiling for this target space?

Reuses helpers from align_09 to keep the comparison identical to the
main 09b run in every way except the alpha value.

See: docs/EXPERIMENT_JOURNAL.md
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from languages.sumerian.scripts.align_09 import (
    build_training_data,
    train_ridge,
    evaluate_alignment,
)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

FUSED_PATH = MODELS_DIR / "fused_embeddings_1536d.npz"
ENGLISH_GEMMA_PATH = MODELS_DIR / "english_gemma_whitened_768d.npz"
ANCHOR_PATH = DATA_PROCESSED / "english_anchors.json"
RESULTS_OUT_PATH = RESULTS_DIR / "ridge_alpha_sweep.json"
GLOVE_BASELINE_PATH = RESULTS_DIR / "alignment_results.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


def main():
    if not ENGLISH_GEMMA_PATH.exists():
        print(f"ERROR: whitened Gemma cache not found at {ENGLISH_GEMMA_PATH}", file=sys.stderr)
        print("Run: python scripts/whiten_gemma.py", file=sys.stderr)
        sys.exit(1)

    print("Loading inputs...")
    fused = np.load(str(FUSED_PATH))
    sum_vectors = fused["vectors"]
    sum_vocab_list = [str(w) for w in fused["vocab"]]
    sum_vocab = {w: i for i, w in enumerate(sum_vocab_list)}

    gemma = np.load(str(ENGLISH_GEMMA_PATH))
    eng_vectors = gemma["vectors"]
    eng_vocab_list = [str(w) for w in gemma["vocab"]]
    eng_vocab = {w: i for i, w in enumerate(eng_vocab_list)}

    with open(ANCHOR_PATH) as f:
        anchors = json.load(f)

    X, Y, valid_anchors = build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors
    )
    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X, Y, valid_anchors, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    test_english = [a["english"] for a in anchors_test]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    baseline = None
    if GLOVE_BASELINE_PATH.exists():
        with open(GLOVE_BASELINE_PATH) as f:
            baseline = json.load(f).get("accuracy", {})
    baseline_top1 = baseline["top1"] if baseline else None

    print(f"\n=== Ridge alpha sweep (whitened-gloss target) ===")
    print(f"{'alpha':>10s}  {'top1':>7s}  {'top5':>7s}  {'top10':>7s}  {'delta top1':>11s}")
    print("-" * 56)

    sweep_results = []
    for alpha in ALPHAS:
        model = train_ridge(X_train, Y_train, alpha=alpha)
        Y_pred = model.predict(X_test)
        results = evaluate_alignment(Y_pred, test_english, eng_vocab_list, eng_vectors)
        delta_str = (
            f"{results['top1'] - baseline_top1:+6.2f}pp"
            if baseline_top1 is not None else "      —"
        )
        print(
            f"{alpha:>10.3g}  {results['top1']:6.2f}%  {results['top5']:6.2f}%  "
            f"{results['top10']:6.2f}%  {delta_str:>11s}"
        )
        sweep_results.append({
            "alpha": alpha,
            "accuracy": results,
            "delta_top1_vs_glove": (
                results["top1"] - baseline_top1 if baseline_top1 is not None else None
            ),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT_PATH, "w") as f:
        json.dump({
            "target": "english_gemma_whitened_768d",
            "baseline_glove": baseline,
            "sweep": sweep_results,
        }, f, indent=2)
    print(f"\nSweep saved to: {RESULTS_OUT_PATH}")


if __name__ == "__main__":
    main()
