"""
Phase A: Ridge alignment of Sumerian FastText into EmbeddingGemma 768d.

Mirrors 09_align_and_evaluate.py but targets EmbeddingGemma-encoded
English vectors instead of GloVe. Reuses helpers from align_09 to
keep the comparison apples-to-apples.

See: docs/superpowers/specs/2026-04-16-gemma-embed-alignment-design.md
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from scripts.align_09 import (
    build_training_data,
    train_ridge,
    evaluate_alignment,
)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

FUSED_PATH = MODELS_DIR / "fused_embeddings_1536d.npz"
ENGLISH_GEMMA_PATH = MODELS_DIR / "english_gemma_768d.npz"
ANCHOR_PATH = DATA_PROCESSED / "english_anchors.json"
RIDGE_OUT_PATH = MODELS_DIR / "ridge_weights_gemma.npz"
RESULTS_OUT_PATH = RESULTS_DIR / "alignment_results_gemma.json"
GLOVE_BASELINE_PATH = RESULTS_DIR / "alignment_results.json"

RIDGE_ALPHA = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42
EXPECTED_TARGET_DIM = 768


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ENGLISH_GEMMA_PATH.exists():
        print(f"ERROR: English Gemma cache not found at {ENGLISH_GEMMA_PATH}", file=sys.stderr)
        print("Run: python scripts/embed_english_gemma.py", file=sys.stderr)
        sys.exit(1)

    print(f"Loading fused Sumerian vectors from {FUSED_PATH}")
    fused = np.load(str(FUSED_PATH))
    sum_vectors = fused["vectors"]
    sum_vocab_list = [str(w) for w in fused["vocab"]]
    sum_vocab = {w: i for i, w in enumerate(sum_vocab_list)}
    print(f"Sumerian vocab: {len(sum_vocab)} words, {sum_vectors.shape[1]}d")

    print(f"Loading Gemma English vectors from {ENGLISH_GEMMA_PATH}")
    gemma = np.load(str(ENGLISH_GEMMA_PATH))
    eng_vectors = gemma["vectors"]
    eng_vocab_list = [str(w) for w in gemma["vocab"]]
    eng_vocab = {w: i for i, w in enumerate(eng_vocab_list)}
    gloss_hit_rate = float(gemma["gloss_hit_rate"]) if "gloss_hit_rate" in gemma.files else None
    gemma_model = str(gemma["gemma_model"]) if "gemma_model" in gemma.files else None
    print(f"English vocab: {len(eng_vocab)} words, {eng_vectors.shape[1]}d")

    assert eng_vectors.shape[1] == EXPECTED_TARGET_DIM, (
        f"English target dim is {eng_vectors.shape[1]}, expected {EXPECTED_TARGET_DIM}. "
        "Regenerate the Gemma cache with scripts/embed_english_gemma.py."
    )

    with open(ANCHOR_PATH) as f:
        anchors = json.load(f)
    print(f"Loaded {len(anchors)} anchors")

    X, Y, valid_anchors = build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors
    )
    print(
        f"Valid anchors: {len(valid_anchors)} / {len(anchors)} "
        f"({len(valid_anchors)/len(anchors)*100:.1f}%)"
    )

    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X, Y, valid_anchors, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print(f"Training Ridge (alpha={RIDGE_ALPHA})...")
    model = train_ridge(X_train, Y_train, alpha=RIDGE_ALPHA)

    Y_pred = model.predict(X_test)
    test_english = [a["english"] for a in anchors_test]
    results = evaluate_alignment(Y_pred, test_english, eng_vocab_list, eng_vectors)

    baseline = None
    if GLOVE_BASELINE_PATH.exists():
        with open(GLOVE_BASELINE_PATH) as f:
            baseline = json.load(f).get("accuracy", {})

    print(f"\n=== RESULTS (Gemma target) ===")
    for k_str in ("top1", "top5", "top10"):
        gemma_val = results[k_str]
        if baseline and k_str in baseline:
            delta = gemma_val - baseline[k_str]
            print(
                f"{k_str.upper():<6} Gemma {gemma_val:6.2f}%  "
                f"GloVe {baseline[k_str]:6.2f}%  "
                f"delta {delta:+.2f}pp"
            )
        else:
            print(f"{k_str.upper():<6} Gemma {gemma_val:6.2f}%")

    full_results = {
        "accuracy": results,
        "baseline_glove": baseline,
        "deltas_vs_glove": (
            {k: results[k] - baseline[k] for k in results if k in baseline}
            if baseline
            else None
        ),
        "config": {
            "alignment": "Ridge",
            "alpha": RIDGE_ALPHA,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_size": len(X_train),
            "test_size_count": len(X_test),
            "valid_anchors": len(valid_anchors),
            "total_anchors": len(anchors),
            "sumerian_vocab": len(sum_vocab),
            "english_vocab": len(eng_vocab),
            "fused_dim": int(sum_vectors.shape[1]),
            "target_dim": int(eng_vectors.shape[1]),
            "gemma_model": gemma_model,
            "gloss_hit_rate": gloss_hit_rate,
        },
    }

    with open(RESULTS_OUT_PATH, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_OUT_PATH}")

    np.savez_compressed(
        str(RIDGE_OUT_PATH),
        coef=model.coef_,
        intercept=model.intercept_,
    )
    print(f"Ridge weights saved to: {RIDGE_OUT_PATH}")


if __name__ == "__main__":
    main()
