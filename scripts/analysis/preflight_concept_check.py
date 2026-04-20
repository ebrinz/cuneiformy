"""
Pre-flight check for the Sumerian cosmogony document's concept slate.

For each candidate concept, reports:
  - Is the Sumerian token in the fused vocab?
  - Is the English seed in Gemma and GloVe vocabs?
  - Does find_both return non-degenerate top-5 matches?
  - How many ETCSL passages contain the token?

Produces a status (pass | fail) per concept, with failure reasons and
warnings. Output JSON is consulted before the generators and document
prose are produced.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations

import datetime as _dt

PREFLIGHT_SCHEMA_VERSION = 1
DEGENERATE_LEN_THRESHOLD = 2  # tokens of length <= 2 are flagged as degenerate
DEGENERATE_TOP5_MIN_FRACTION = 0.4  # <=40% multi-char tokens in top 5 -> flagged


def preflight_check(
    lookup,
    candidate_concepts: list[dict],
    etcsl_texts: list[dict],
) -> dict:
    """Validate each concept against the current lookup + ETCSL corpus."""
    verdicts = []
    for concept in candidate_concepts:
        sum_tok = concept["sumerian"]
        eng_seed = concept["english"]

        sum_in_vocab = sum_tok in lookup.vocab

        # English-vocab check via find() returning non-empty.
        eng_in_gemma = bool(lookup.find(eng_seed, top_k=1, space="gemma"))
        eng_in_glove = bool(lookup.find(eng_seed, top_k=1, space="glove"))

        # Top-5 quality check.
        top5 = lookup.find_both(eng_seed, top_k=5) if eng_in_gemma or eng_in_glove else {"gemma": [], "glove": []}
        all_top5 = list(top5.get("gemma", [])) + list(top5.get("glove", []))
        if all_top5:
            multi_char_count = sum(1 for w, _ in all_top5 if len(w) > DEGENERATE_LEN_THRESHOLD)
            degenerate_fraction = 1.0 - (multi_char_count / len(all_top5))
        else:
            degenerate_fraction = 1.0

        # ETCSL passage count.
        etcsl_count = 0
        for text in etcsl_texts:
            for line in text.get("lines", []):
                if sum_tok in (line.get("transliteration") or "").split():
                    etcsl_count += 1

        failure_reasons = []
        warnings = []

        if not sum_in_vocab:
            failure_reasons.append("sumerian_vocab_miss")
        if not eng_in_gemma and not eng_in_glove:
            failure_reasons.append("english_missing_both_spaces")
        if etcsl_count == 0:
            failure_reasons.append("zero_etcsl_passages")
        if degenerate_fraction > 0.5:
            warnings.append("degenerate_top5")
        if not eng_in_gemma:
            warnings.append("english_missing_gemma")
        if not eng_in_glove:
            warnings.append("english_missing_glove")

        verdicts.append({
            "concept": concept,
            "status": "fail" if failure_reasons else "pass",
            "sumerian_in_vocab": sum_in_vocab,
            "english_in_gemma": eng_in_gemma,
            "english_in_glove": eng_in_glove,
            "etcsl_passages": etcsl_count,
            "degenerate_fraction_top5": round(degenerate_fraction, 3),
            "failure_reasons": failure_reasons,
            "warnings": warnings,
            "top5_gemma": top5.get("gemma", []),
            "top5_glove": top5.get("glove", []),
        })

    return {
        "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
        "preflight_date": _dt.date.today().isoformat(),
        "concepts": verdicts,
    }
