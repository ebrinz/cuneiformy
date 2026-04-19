import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# --- Loader tests ----------------------------------------------------------


def test_load_corpus_frequency_counts_tokens():
    from scripts.coverage_diagnostic import _load_corpus_frequency

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cleaned.txt"
        path.write_text("lugal dingir lugal\nlugal mu-dug\ndingir\n", encoding="utf-8")
        freq = _load_corpus_frequency(path)
        assert freq["lugal"] == 3
        assert freq["dingir"] == 2
        assert freq["mu-dug"] == 1


def test_load_corpus_frequency_empty_file():
    from scripts.coverage_diagnostic import _load_corpus_frequency

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.txt"
        path.write_text("", encoding="utf-8")
        freq = _load_corpus_frequency(path)
        assert len(freq) == 0


def test_load_lemma_surface_map_builds_citation_to_surfaces():
    from scripts.coverage_diagnostic import _load_lemma_surface_map

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lemmas.json"
        lemmas = [
            {"cf": "lugal", "form": "lugal", "gw": "king"},
            {"cf": "lugal", "form": "lugale", "gw": "king"},
            {"cf": "lugal", "form": "lugalanene", "gw": "king"},
            {"cf": "dingir", "form": "dingir", "gw": "god"},
        ]
        path.write_text(json.dumps(lemmas), encoding="utf-8")
        mapping = _load_lemma_surface_map(path)
        assert mapping["lugal"] == {"lugal", "lugale", "lugalanene"}
        assert mapping["dingir"] == {"dingir"}


def test_load_lemma_surface_map_normalizes_via_oracc_to_atf():
    from scripts.coverage_diagnostic import _load_lemma_surface_map

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lemmas.json"
        lemmas = [
            {"cf": "šeš", "form": "šeš", "gw": "brother"},
            {"cf": "šeš", "form": "šešeš", "gw": "brother"},
        ]
        path.write_text(json.dumps(lemmas, ensure_ascii=False), encoding="utf-8")
        mapping = _load_lemma_surface_map(path)
        # ORACC 'š' -> ATF 'sz' normalization applied
        assert "szesz" in mapping
        assert "szesz" in mapping["szesz"]
        assert "szeszesz" in mapping["szesz"]


def test_load_lemma_surface_map_skips_empty_cf_or_form():
    from scripts.coverage_diagnostic import _load_lemma_surface_map

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lemmas.json"
        lemmas = [
            {"cf": "lugal", "form": "lugal", "gw": "king"},
            {"cf": "", "form": "lugal", "gw": "king"},
            {"cf": "lugal", "form": "", "gw": "king"},
        ]
        path.write_text(json.dumps(lemmas), encoding="utf-8")
        mapping = _load_lemma_surface_map(path)
        assert mapping == {"lugal": {"lugal"}}


def test_diagnostic_context_is_frozen():
    from scripts.coverage_diagnostic import DiagnosticContext

    ctx = _make_tiny_context()
    with pytest.raises((AttributeError, Exception)):
        ctx.fused_vocab = frozenset()


def _make_tiny_context():
    """Build a DiagnosticContext over tiny synthetic inputs for downstream tests."""
    from scripts.coverage_diagnostic import DiagnosticContext

    return DiagnosticContext(
        fused_vocab=frozenset({"lugal", "dingir", "nar", "ta", "narta"}),
        glove_vocab=frozenset({"king", "god"}),
        gemma_vocab=frozenset({"king", "god"}),
        corpus_frequency={"lugal": 10, "dingir": 8, "narta": 2, "mu": 1},
        lemma_surface_map={"lugal": frozenset({"lugal", "lugale"})},
        fasttext_model=None,
        gemma_english_vocab=["king", "god"],
        gemma_english_vectors=np.eye(2, 768, dtype=np.float32),
        ridge_gemma_coef=np.zeros((768, 1536), dtype=np.float32),
        ridge_gemma_intercept=np.zeros(768, dtype=np.float32),
    )


def test_diagnostic_context_accepts_all_required_fields():
    ctx = _make_tiny_context()
    assert "lugal" in ctx.fused_vocab
    assert ctx.corpus_frequency["dingir"] == 8
    assert "lugal" in ctx.lemma_surface_map
    assert ctx.ridge_gemma_coef.shape == (768, 1536)


# --- Classifier tests ------------------------------------------------------


PRIMARY_CAUSE_ORDER = (
    "normalization_recoverable",
    "in_corpus_below_min_count",
    "oracc_lemma_surface_recoverable",
    "morpheme_composition_recoverable",
    "subword_inference_recoverable",
    "genuinely_missing",
)


def _classifier_ctx(**overrides):
    """Extend _make_tiny_context with test-specific overrides."""
    from scripts.coverage_diagnostic import DiagnosticContext

    base = dict(
        fused_vocab=frozenset({"lugal", "dingir", "nar", "ta", "narta", "za3sze3la2"}),
        glove_vocab=frozenset({"king", "god"}),
        gemma_vocab=frozenset({"king", "god"}),
        corpus_frequency={"lugal": 10, "dingir": 8, "narta": 2, "rare1": 1, "rare3": 3},
        lemma_surface_map={"kasan": frozenset({"kasan", "kasane"})},
        fasttext_model=None,
        gemma_english_vocab=["king", "god"],
        gemma_english_vectors=np.eye(2, 768, dtype=np.float32),
        ridge_gemma_coef=np.zeros((768, 1536), dtype=np.float32),
        ridge_gemma_intercept=np.zeros(768, dtype=np.float32),
    )
    base.update(overrides)
    return DiagnosticContext(**base)


def test_normalize_anchor_form_handles_subscripts_braces_and_oracc():
    from scripts.coverage_diagnostic import normalize_anchor_form

    assert normalize_anchor_form("{tug₂}mug") == "tug2mug"
    assert normalize_anchor_form("za₃-sze₃-la₂") == "za3sze3la2"
    assert normalize_anchor_form("šeš") == "szesz"  # š->sz, e->e, š->sz
    assert normalize_anchor_form("mu-du₃-sze₃") == "mudu3sze3"


def test_classify_miss_normalization_recoverable_wins():
    from scripts.coverage_diagnostic import classify_miss

    ctx = _classifier_ctx()
    # za₃-sze₃-la₂ normalizes to za3sze3la2 which is in fused_vocab.
    bucket = classify_miss(
        {"sumerian": "za₃-sze₃-la₂", "english": "cosmic force", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket == "normalization_recoverable"


def test_classify_miss_in_corpus_below_min_count():
    from scripts.coverage_diagnostic import classify_miss

    ctx = _classifier_ctx()
    # "rare3" is in corpus_frequency with count 3 but NOT in fused_vocab (min_count=5).
    bucket = classify_miss(
        {"sumerian": "rare3", "english": "uncommon", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket == "in_corpus_below_min_count"


def test_classify_miss_oracc_lemma_surface_recoverable():
    from scripts.coverage_diagnostic import classify_miss

    # "kasan" is in lemma_surface_map with "kasane" among its surfaces.
    # Give fused_vocab that includes "kasane" but NOT "kasan".
    ctx = _classifier_ctx(
        fused_vocab=frozenset({"lugal", "kasane"}),
        lemma_surface_map={"kasan": frozenset({"kasan", "kasane"})},
    )
    bucket = classify_miss(
        {"sumerian": "kasan", "english": "noblewoman", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket == "oracc_lemma_surface_recoverable"


def test_classify_miss_morpheme_composition_requires_all_morphemes_in_vocab():
    from scripts.coverage_diagnostic import classify_miss

    # corpus_frequency must NOT contain "narta" (normalized form of "nar-ta"),
    # otherwise bucket 2 fires before bucket 4.
    ctx = _classifier_ctx(
        fused_vocab=frozenset({"nar", "ta"}),
        corpus_frequency={"lugal": 10, "dingir": 8},
    )
    # "nar-ta" hyphenates into ["nar", "ta"], both in vocab.
    bucket = classify_miss(
        {"sumerian": "nar-ta", "english": "musician", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket == "morpheme_composition_recoverable"

    # "nar-missing" has one morpheme out of vocab -> NOT recoverable by morpheme comp.
    # Must fall through to subword or genuinely_missing. With no trained ngrams, genuinely_missing.
    bucket2 = classify_miss(
        {"sumerian": "nar-missing", "english": "something", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket2 == "genuinely_missing"


def test_classify_miss_subword_inference_respects_threshold():
    from scripts.coverage_diagnostic import classify_miss, _ngrams

    ctx = _classifier_ctx()
    # Build a trained_ngrams set that covers >= 50% of "xyznew"'s n-grams.
    trained = _ngrams("xyznewz", 3, 6) | _ngrams("xyznew", 3, 6)
    bucket = classify_miss(
        {"sumerian": "xyznew", "english": "something", "confidence": 0.9},
        ctx,
        trained_ngrams=trained,
    )
    assert bucket == "subword_inference_recoverable"


def test_classify_miss_subword_below_threshold_falls_to_missing():
    from scripts.coverage_diagnostic import classify_miss

    ctx = _classifier_ctx()
    # Empty trained set -> overlap is 0%. Below threshold 0.5.
    bucket = classify_miss(
        {"sumerian": "completelyunknown", "english": "absent", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket == "genuinely_missing"


def test_classify_all_misses_sums_to_total():
    from scripts.coverage_diagnostic import classify_all_misses, _ngrams

    # fused_vocab must NOT contain "narta" (normalized "nar-ta") and corpus_frequency
    # must NOT contain "narta" so bucket 4 fires for the morpheme case.
    # "rare3" stays for bucket 2; "za3sze3la2" stays for bucket 1.
    ctx = _classifier_ctx(
        fused_vocab=frozenset({"lugal", "dingir", "nar", "ta", "za3sze3la2"}),
        corpus_frequency={"lugal": 10, "dingir": 8, "rare3": 3},
    )
    misses = [
        {"sumerian": "za₃-sze₃-la₂", "english": "force", "confidence": 0.9},  # normalization
        {"sumerian": "rare3", "english": "uncommon", "confidence": 0.9},        # below_min_count
        {"sumerian": "nar-ta", "english": "musician", "confidence": 0.9},       # morpheme
        {"sumerian": "completelyunknown", "english": "absent", "confidence": 0.9},  # missing
    ]
    trained = _ngrams("", 3, 6)  # empty set
    result = classify_all_misses(misses, ctx, trained)

    total = sum(b["count"] for b in result["primary_causes"].values())
    assert total == len(misses) == 4
    assert result["primary_causes"]["normalization_recoverable"]["count"] == 1
    assert result["primary_causes"]["in_corpus_below_min_count"]["count"] == 1
    assert result["primary_causes"]["morpheme_composition_recoverable"]["count"] == 1
    assert result["primary_causes"]["genuinely_missing"]["count"] == 1
    assert result["total_misses"] == 4
    # All priority buckets present even if zero
    assert set(result["primary_causes"].keys()) == set(PRIMARY_CAUSE_ORDER)


def test_classify_all_misses_empty_input():
    from scripts.coverage_diagnostic import classify_all_misses

    ctx = _classifier_ctx()
    result = classify_all_misses([], ctx, frozenset())
    assert result["total_misses"] == 0
    for bucket in result["primary_causes"].values():
        assert bucket["count"] == 0


def test_ngrams_respects_min_n_max_n():
    from scripts.coverage_diagnostic import _ngrams

    result = _ngrams("abc", 3, 3)
    # padded "<abc>" -> 3-grams: "<ab", "abc", "bc>"
    assert result == {"<ab", "abc", "bc>"}


def test_trained_ngrams_unions_vocab():
    from scripts.coverage_diagnostic import _trained_ngrams

    trained = _trained_ngrams(["ab", "cd"], 2, 2)
    assert "<a" in trained
    assert "ab" in trained
    assert "b>" in trained
    assert "<c" in trained
    assert "cd" in trained
    assert "d>" in trained


# --- Exact-match simulator tests -------------------------------------------


def test_simulate_ascii_normalize_counts_exact_hits():
    from scripts.coverage_diagnostic import simulate_ascii_normalize

    ctx = _classifier_ctx(fused_vocab=frozenset({"tug2mug", "za3sze3la2", "lugal"}))
    misses = [
        {"sumerian": "{tug₂}mug", "english": "garment", "confidence": 0.9},        # hits
        {"sumerian": "za₃-sze₃-la₂", "english": "force", "confidence": 0.9},       # hits
        {"sumerian": "completelyunknown", "english": "absent", "confidence": 0.9}, # misses
    ]
    result = simulate_ascii_normalize(misses, ctx)
    assert result["anchors_newly_resolvable"] == 2
    assert result["trustworthiness"] == "exact"


def test_simulate_ascii_normalize_ignores_anchors_already_normalized():
    from scripts.coverage_diagnostic import simulate_ascii_normalize

    ctx = _classifier_ctx(fused_vocab=frozenset({"dingir"}))
    # This anchor's normalized form is NOT in vocab -> not recovered by normalization.
    misses = [{"sumerian": "missing", "english": "absent", "confidence": 0.9}]
    result = simulate_ascii_normalize(misses, ctx)
    assert result["anchors_newly_resolvable"] == 0


def test_simulate_lower_min_count_per_threshold_is_monotone_nonincreasing():
    from scripts.coverage_diagnostic import simulate_lower_min_count

    ctx = _classifier_ctx(
        corpus_frequency={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        fused_vocab=frozenset({"e", "f"}),  # only >=5 made the vocab cut
    )
    misses = [
        {"sumerian": s, "english": "x", "confidence": 0.9}
        for s in ("a", "b", "c", "d", "e", "f")  # e, f already in vocab so skipped
    ]
    result = simulate_lower_min_count(misses, ctx)
    per = result["per_threshold"]
    # Anchors newly resolvable at each threshold (only miss anchors counted).
    # misses in corpus with freq >= threshold and not in vocab:
    #   t=1: a(1), b(2), c(3), d(4) -> 4
    #   t=2: b(2), c(3), d(4) -> 3
    #   t=3: c(3), d(4) -> 2
    #   t=4: d(4) -> 1
    assert per["1"]["anchors_newly_resolvable"] == 4
    assert per["2"]["anchors_newly_resolvable"] == 3
    assert per["3"]["anchors_newly_resolvable"] == 2
    assert per["4"]["anchors_newly_resolvable"] == 1
    # Monotone non-increasing across thresholds.
    counts = [per[str(t)]["anchors_newly_resolvable"] for t in (1, 2, 3, 4)]
    assert counts == sorted(counts, reverse=True)


def test_simulate_oracc_lemma_expansion_counts_unique_anchors():
    from scripts.coverage_diagnostic import simulate_oracc_lemma_expansion

    # Citation form "kasan" has three surface variants; "kasane" is in vocab.
    ctx = _classifier_ctx(
        fused_vocab=frozenset({"kasane"}),
        lemma_surface_map={"kasan": frozenset({"kasan", "kasane", "kasanene"})},
    )
    misses = [
        {"sumerian": "kasan", "english": "noblewoman", "confidence": 0.9},
        {"sumerian": "completelyunknown", "english": "absent", "confidence": 0.9},
    ]
    result = simulate_oracc_lemma_expansion(misses, ctx)
    assert result["anchors_newly_resolvable"] == 1
    # surface_forms_added_to_vocab reports total unique surface forms recovered.
    assert result["surface_forms_added_to_vocab"] >= 1


def test_simulate_oracc_lemma_expansion_requires_in_vocab_surface():
    from scripts.coverage_diagnostic import simulate_oracc_lemma_expansion

    # Citation has 2 surfaces, neither in vocab.
    ctx = _classifier_ctx(
        fused_vocab=frozenset({"unrelated"}),
        lemma_surface_map={"kasan": frozenset({"kasanx", "kasany"})},
    )
    misses = [{"sumerian": "kasan", "english": "noblewoman", "confidence": 0.9}]
    result = simulate_oracc_lemma_expansion(misses, ctx)
    assert result["anchors_newly_resolvable"] == 0
