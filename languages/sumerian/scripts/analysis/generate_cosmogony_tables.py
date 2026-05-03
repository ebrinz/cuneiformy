"""
Generate cosmogony_tables.json from the final concept slate.

For each concept, compute:
  - Top-10 nearest Sumerian neighbors in both spaces (dual-view)
  - 2-3 analogy probe results
  - English displacement numbers in both spaces
  - 1-2 ETCSL passage excerpts

All tables committed to docs/cosmogony_tables.json — this is the canonical
source the document prose references.

Usage:
    cd /Users/crashy/Development/cuneiformy
    python scripts/analysis/generate_cosmogony_tables.py

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
_LANG_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from languages.sumerian.final_output.sumerian_lookup import SumerianLookup
from languages.sumerian.scripts.analysis.cosmogony_concepts import PRIMARY_CONCEPTS, ANUNNAKI_VOCABULARY
from framework.analysis.english_displacement import english_displacement
from languages.sumerian.scripts.analysis.etcsl_passage_finder import find_passages

ROOT = _LANG_ROOT
TABLES_PATH = ROOT / "docs" / "cosmogony_tables.json"

# Analogy probes — curated per concept. Each is (a, b, c, space) such that
# find_analogy(a, b, c, space=space) tests a specific interpretive claim.
ANALOGY_PROBES = {
    "abzu":   [("ocean", "water", "deep", "gemma")],
    "zi":     [("breath", "air", "life", "gemma")],
    "nam":    [("essence", "name", "being", "gemma")],
    "namtar": [("fate", "name", "decree", "gemma")],
    "me":     [("decree", "order", "essence", "gemma")],
}


def _load_lookup():
    print("Loading GloVe (~1 min)...")
    glove_vocab, glove_vectors = [], []
    with open(ROOT / "data/processed/glove.6B.300d.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            glove_vocab.append(parts[0])
            glove_vectors.append([float(x) for x in parts[1:]])
    glove_vectors = np.array(glove_vectors, dtype=np.float32)

    return SumerianLookup(
        gemma_vectors_path=str(ROOT / "final_output/sumerian_aligned_gemma_vectors.npz"),
        glove_vectors_path=str(ROOT / "final_output/sumerian_aligned_vectors.npz"),
        vocab_path=str(ROOT / "final_output/sumerian_aligned_vocab.pkl"),
        gemma_english_path=str(ROOT / "models/english_gemma_whitened_768d.npz"),
        glove_english_vectors=glove_vectors,
        glove_english_vocab=glove_vocab,
    )


def _top10_dual_view(lookup, english_seed: str) -> dict:
    both = lookup.find_both(english_seed, top_k=10)
    return {
        "gemma": [{"sumerian": w, "similarity": round(float(s), 4)} for w, s in both["gemma"]],
        "glove": [{"sumerian": w, "similarity": round(float(s), 4)} for w, s in both["glove"]],
    }


def _analogy_probes_for(lookup, concept_tag: str) -> list:
    probes = ANALOGY_PROBES.get(concept_tag, [])
    out = []
    for a, b, c, space in probes:
        result = lookup.find_analogy(a, b, c, top_k=5, space=space)
        out.append({
            "query": f"{a} : {b} :: {c} : ?",
            "space": space,
            "results": [{"sumerian": w, "similarity": round(float(s), 4)} for w, s in result],
        })
    return out


def main() -> int:
    lookup = _load_lookup()

    with open(ROOT / "data/raw/etcsl_texts.json") as f:
        etcsl = json.load(f)

    concepts_out = {}
    for concept in PRIMARY_CONCEPTS:
        sum_tok = concept["sumerian"]
        eng_seed = concept["english"]

        top10 = _top10_dual_view(lookup, eng_seed)

        displacement = {
            "gemma": english_displacement(lookup, sum_tok, eng_seed, space="gemma"),
            "glove": english_displacement(lookup, sum_tok, eng_seed, space="glove"),
        }

        analogies = _analogy_probes_for(lookup, sum_tok)

        passages = find_passages(sum_tok, etcsl, max_passages=2, context_lines=1)

        concepts_out[sum_tok] = {
            "concept": concept,
            "top10_dual_view": top10,
            "english_displacement": displacement,
            "analogy_probes": analogies,
            "etcsl_passages": passages,
        }

    tables = {
        "schema_version": 1,
        "concept_slate": PRIMARY_CONCEPTS,
        "anunnaki_vocabulary": ANUNNAKI_VOCABULARY,
        "concepts": concepts_out,
    }

    TABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLES_PATH, "w") as f:
        json.dump(tables, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote: {TABLES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
