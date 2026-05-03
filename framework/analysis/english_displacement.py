"""
English-displacement measurement.

Measures how far a Sumerian-projected vector lands from the English seed's
native vector in the same target space. Reported per concept in §6 of each
deep dive.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations

import numpy as np


def english_displacement(
    lookup,
    sumerian_token: str,
    english_seed: str,
    space: str = "gemma",
) -> dict:
    """Return cosine between Sumerian-projected and English-native vectors.

    Both vectors are pre-L2-normalized in the SumerianLookup, so the dot
    product equals cosine similarity.
    """
    s = lookup._spaces[space]

    # Sumerian side
    idx_map = {t: i for i, t in enumerate(lookup.vocab)}
    if sumerian_token not in idx_map:
        raise KeyError(f"unknown Sumerian token: {sumerian_token!r}")
    sum_vec = s["sum_norm"][idx_map[sumerian_token]]

    # English side
    eng_lower = english_seed.lower()
    if eng_lower not in s["eng_vocab_map"]:
        raise KeyError(f"unknown English seed in {space!r} vocab: {english_seed!r}")
    eng_vec = s["eng_norm"][s["eng_vocab_map"][eng_lower]]

    cos_sim = float(np.clip(np.dot(sum_vec, eng_vec), -1.0, 1.0))
    return {
        "sumerian_token": sumerian_token,
        "english_seed": english_seed,
        "space": space,
        "cosine_similarity": cos_sim,
        "cosine_distance": 1.0 - cos_sim,
    }
