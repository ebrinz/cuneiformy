"""
Production Export: Project all Sumerian vectors into GloVe space and package.

Uses pickle for vocab serialization (locally-generated data only, matching
heiroglyphy's established pattern for numpy/vocab packaging).
"""
import json
import pickle
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FINAL_OUTPUT = Path(__file__).parent.parent / "final_output"


def project_all_vectors(
    sum_vectors: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    """Project all Sumerian vectors into GloVe space using learned Ridge weights."""
    projected = sum_vectors @ coef.T + intercept
    return projected.astype(np.float16)


def main():
    FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    fused_data = np.load(str(MODELS_DIR / "fused_embeddings_1536d.npz"), allow_pickle=True)
    sum_vectors = fused_data["vectors"]
    sum_vocab = list(fused_data["vocab"])
    print(f"Sumerian vectors: {sum_vectors.shape}")

    ridge_data = np.load(str(MODELS_DIR / "ridge_weights.npz"))
    coef = ridge_data["coef"]
    intercept = ridge_data["intercept"]
    print(f"Ridge coef: {coef.shape}")

    aligned = project_all_vectors(sum_vectors, coef, intercept)
    print(f"Aligned vectors: {aligned.shape}, dtype: {aligned.dtype}")

    np.savez_compressed(
        str(FINAL_OUTPUT / "sumerian_aligned_vectors.npz"),
        vectors=aligned,
    )
    with open(FINAL_OUTPUT / "sumerian_aligned_vocab.pkl", "wb") as f:
        pickle.dump(sum_vocab, f)

    results_path = RESULTS_DIR / "alignment_results.json"
    with open(results_path) as f:
        results = json.load(f)

    metadata = {
        "methodology": "Cuneiformy SOTA (1536d fused -> 300d GloVe)",
        "text_embeddings": "768d FastText (min_count=5, window=10)",
        "visual_embeddings": "768d zero-padding (regularization)",
        "alignment": "Ridge Regression (alpha=0.001)",
        "accuracy": results["accuracy"],
        "vocab_size": len(sum_vocab),
        "vector_dim": 300,
        "config": results["config"],
    }

    with open(FINAL_OUTPUT / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProduction files saved to {FINAL_OUTPUT}/")


if __name__ == "__main__":
    main()
