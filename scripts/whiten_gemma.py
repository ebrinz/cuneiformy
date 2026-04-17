"""
Apply BERT-whitening style centering + decorrelation to the cached
EmbeddingGemma English vectors, then save a whitened cache.

Contextual embeddings are anisotropic — they cluster in a narrow cone,
so most of any vector is a shared bias direction and cosine similarity
between unrelated words is inflated. Centering removes the cone;
whitening scales every direction to unit variance so no single direction
dominates the alignment.

Default source: models/english_gemma_768d.npz (gloss run).
Use --source bare to whiten the bare-word cache instead.

Output: models/english_gemma_whitened_768d.npz (or
models/english_gemma_bare_whitened_768d.npz with --source bare).

See: docs/EXPERIMENT_JOURNAL.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

SOURCES = {
    "gloss": MODELS_DIR / "english_gemma_768d.npz",
    "bare": MODELS_DIR / "english_gemma_bare_768d.npz",
}
OUTPUTS = {
    "gloss": MODELS_DIR / "english_gemma_whitened_768d.npz",
    "bare": MODELS_DIR / "english_gemma_bare_whitened_768d.npz",
}
TRANSFORM_OUTPUTS = {
    "gloss": MODELS_DIR / "gemma_whitening_transform.npz",
    "bare": MODELS_DIR / "gemma_bare_whitening_transform.npz",
}

EPS = 1e-6


def compute_whitening(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, W) such that (X - mean) @ W has zero mean and identity covariance."""
    mu = X.mean(axis=0).astype(np.float64)
    X_c = X.astype(np.float64) - mu
    n = X_c.shape[0]
    cov = (X_c.T @ X_c) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_clipped = np.maximum(eigvals, EPS)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals_clipped))
    W = eigvecs @ D_inv_sqrt @ eigvecs.T
    return mu.astype(np.float32), W.astype(np.float32)


def whiten(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return ((X.astype(np.float32) - mu) @ W).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["gloss", "bare"], default="gloss")
    args = parser.parse_args()

    src_path = SOURCES[args.source]
    dst_path = OUTPUTS[args.source]
    xform_path = TRANSFORM_OUTPUTS[args.source]

    if not src_path.exists():
        print(f"ERROR: source cache not found at {src_path}", file=sys.stderr)
        suffix = " --bare" if args.source == "bare" else ""
        print(f"Run: python scripts/embed_english_gemma.py{suffix}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.source} Gemma cache from {src_path}")
    d = np.load(str(src_path))
    X = d["vectors"].astype(np.float32)
    vocab = d["vocab"]
    print(f"Shape: {X.shape}")
    print(f"Pre-whitening mean norm:  {np.linalg.norm(X.mean(axis=0)):.4f}")
    print(f"Pre-whitening mean cosine (1000-sample estimate): "
          f"{np.mean(X[:1000] @ X[:1000].T):.4f}")

    print("Computing whitening transform...")
    mu, W = compute_whitening(X)

    print("Applying transform to all vectors...")
    X_w = whiten(X, mu, W)

    new_mean_norm = float(np.linalg.norm(X_w.mean(axis=0)))
    sample_cov = (X_w[:1000].T @ X_w[:1000]) / (1000 - 1)
    sample_trace = float(np.trace(sample_cov))
    print(f"Post-whitening mean norm: {new_mean_norm:.6f} (expect ~0)")
    print(f"Post-whitening cov trace (sample of 1000): {sample_trace:.2f} (expect ~{X_w.shape[1]})")

    np.savez_compressed(
        str(dst_path),
        vocab=vocab,
        vectors=X_w,
        mode=np.array(f"{args.source}_whitened"),
        source_path=np.array(str(src_path.name)),
    )
    np.savez_compressed(
        str(xform_path),
        mean=mu,
        transform=W,
    )
    print(f"Whitened cache saved to {dst_path}")
    print(f"Whitening transform saved to {xform_path}")


if __name__ == "__main__":
    main()
