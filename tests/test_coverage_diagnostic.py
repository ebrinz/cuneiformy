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


# --- Inference simulator + Tier-2 tests ------------------------------------


def _tier2_ctx_with_exact_identity():
    """A ctx where the ridge is identity-like (on first 768 dims) and Gemma
    English rows are canonical basis vectors, so a Sumerian input that's a
    scaled basis vector lands on the corresponding English word."""
    from scripts.coverage_diagnostic import DiagnosticContext

    # 5 English words: king, god, cow, river, mountain
    eng_vocab = ["king", "god", "cow", "river", "mountain"]
    eng_vectors = np.zeros((5, 768), dtype=np.float32)
    for i in range(5):
        eng_vectors[i, i] = 1.0

    # Ridge: coef is (768, 1536); intercept zero. coef's first 768 cols act as identity
    # on the FastText portion of the fused 1536d vector (zeros in the second half).
    coef = np.zeros((768, 1536), dtype=np.float32)
    for i in range(768):
        coef[i, i] = 1.0
    intercept = np.zeros(768, dtype=np.float32)

    return DiagnosticContext(
        fused_vocab=frozenset({"lugal", "dingir", "nar", "ta"}),
        glove_vocab=frozenset(eng_vocab),
        gemma_vocab=frozenset(eng_vocab),
        corpus_frequency={},
        lemma_surface_map={},
        fasttext_model=None,
        gemma_english_vocab=eng_vocab,
        gemma_english_vectors=eng_vectors,
        ridge_gemma_coef=coef,
        ridge_gemma_intercept=intercept,
    )


def test_tier2_project_and_nearest_neighbor_identifies_correct_english():
    from scripts.coverage_diagnostic import _tier2_nearest_english

    ctx = _tier2_ctx_with_exact_identity()
    # Synthesized FastText vec that aligns with the 'cow' basis (index 2).
    ft_vec = np.zeros(768, dtype=np.float32)
    ft_vec[2] = 1.0

    top_k = _tier2_nearest_english(ft_vec, ctx, k=3)
    assert top_k[0] == "cow"


def test_tier2_score_returns_correctness_flags():
    from scripts.coverage_diagnostic import _tier2_score_anchor

    ctx = _tier2_ctx_with_exact_identity()
    ft_vec = np.zeros(768, dtype=np.float32)
    ft_vec[3] = 1.0  # 'river'

    result = _tier2_score_anchor(ft_vec, expected_english="river", ctx=ctx)
    assert result["top1"] is True
    assert result["top5"] is True
    assert result["top10"] is True

    result2 = _tier2_score_anchor(ft_vec, expected_english="mountain", ctx=ctx)
    # "mountain" is index 4; not the top-1 hit.
    assert result2["top1"] is False


def test_simulate_morpheme_composition_tier1_counts_anchors_with_all_morphemes_in_vocab():
    from scripts.coverage_diagnostic import simulate_morpheme_composition

    ctx = _tier2_ctx_with_exact_identity()
    # Mock FastText behavior using a small dict lookup: fusion is impossible without
    # a real model. We patch the simulator's ft-vector lookup via a simple dict.
    morpheme_vectors = {
        "nar": np.zeros(768, dtype=np.float32),
        "ta": np.zeros(768, dtype=np.float32),
        "lugal": np.ones(768, dtype=np.float32) * 0.5,
    }
    morpheme_vectors["nar"][0] = 1.0  # king
    morpheme_vectors["ta"][1] = 1.0   # god

    misses = [
        {"sumerian": "nar-ta", "english": "king god", "confidence": 0.9},  # both morphemes in vocab
        {"sumerian": "nar-missing", "english": "x", "confidence": 0.9},    # "missing" not in vocab
        {"sumerian": "flat", "english": "x", "confidence": 0.9},           # no hyphen, not a candidate
    ]

    result = simulate_morpheme_composition(
        misses, ctx, morpheme_vector_lookup=morpheme_vectors.get
    )
    assert result["anchors_newly_resolvable_tier1"] == 1
    assert result["trustworthiness"].startswith("inferred")


def test_simulate_morpheme_composition_tier2_scores_against_expected_english():
    from scripts.coverage_diagnostic import simulate_morpheme_composition

    ctx = _tier2_ctx_with_exact_identity()
    # "nar-ta" morpheme-mean vector lands on index 0 (king) and index 1 (god) both at 0.5.
    morpheme_vectors = {
        "nar": np.zeros(768, dtype=np.float32),
        "ta": np.zeros(768, dtype=np.float32),
    }
    morpheme_vectors["nar"][0] = 1.0  # king
    morpheme_vectors["ta"][0] = 1.0   # reinforces king
    misses = [
        {"sumerian": "nar-ta", "english": "king", "confidence": 0.9},  # expected to be top-1
    ]
    result = simulate_morpheme_composition(
        misses, ctx, morpheme_vector_lookup=morpheme_vectors.get
    )
    tier2 = result["tier2_semantic"]
    assert tier2["tested"] == 1
    assert tier2["top1_correct"] == 1


def test_simulate_subword_inference_tier1_counts_above_threshold_overlap():
    from scripts.coverage_diagnostic import simulate_subword_inference, _ngrams

    ctx = _tier2_ctx_with_exact_identity()
    trained = _ngrams("narta", 3, 6)
    misses = [
        {"sumerian": "narta", "english": "king", "confidence": 0.9},       # 100% overlap
        {"sumerian": "unrelated", "english": "god", "confidence": 0.9},    # 0% overlap
    ]
    # Subword simulator uses a stub FastText that returns index-0 basis vector.
    def fake_subword_vector(word: str) -> np.ndarray:
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1.0  # always 'king'
        return v

    result = simulate_subword_inference(
        misses, ctx, trained_ngrams=trained,
        subword_vector_lookup=fake_subword_vector,
        fasttext_min_n=3, fasttext_max_n=6,
    )
    assert result["anchors_newly_resolvable_tier1"] == 1
    tier2 = result["tier2_semantic"]
    assert tier2["tested"] == 1
    assert tier2["top1_correct"] == 1


def test_simulate_subword_inference_tier2_skips_anchors_without_english_in_gemma():
    from scripts.coverage_diagnostic import simulate_subword_inference, _ngrams

    ctx = _tier2_ctx_with_exact_identity()
    trained = _ngrams("narta", 3, 6)
    # Anchor's English side is "unlisted" which is NOT in ctx.gemma_vocab.
    misses = [
        {"sumerian": "narta", "english": "unlisted", "confidence": 0.9},
    ]
    def fake_subword_vector(word):
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1.0
        return v

    result = simulate_subword_inference(
        misses, ctx, trained_ngrams=trained,
        subword_vector_lookup=fake_subword_vector,
        fasttext_min_n=3, fasttext_max_n=6,
    )
    # Tier-1 recoverable (ngram overlap passes).
    assert result["anchors_newly_resolvable_tier1"] == 1
    # Tier-2 skipped because expected english isn't in Gemma vocab.
    tier2 = result["tier2_semantic"]
    assert tier2["tested"] == 0
    assert tier2["skipped"] == 1


# --- Rendering + main tests ------------------------------------------------


def _full_metadata():
    return {
        "diagnostic_date": "2026-04-19",
        "source_artifacts": {
            "anchors_path": "data/processed/english_anchors.json",
            "anchors_sha256": "0" * 64,
            "fasttext_model_path": "models/fasttext_sumerian.model",
            "fasttext_model_sha256": "0" * 64,
            "fused_vocab_path": "models/fused_embeddings_1536d.npz",
            "glove_path": "data/processed/glove.6B.300d.txt",
            "gemma_path": "models/english_gemma_whitened_768d.npz",
            "ridge_gemma_path": "models/ridge_weights_gemma_whitened.npz",
            "oracc_lemmas_path": "data/raw/oracc_lemmas.json",
            "cleaned_corpus_path": "data/processed/cleaned_corpus.txt",
            "cleaned_corpus_sha256": "0" * 64,
            "seed": 42,
            "subword_overlap_min": 0.5,
        },
        "baseline": {
            "total_merged": 13886,
            "survives": 1951,
            "sumerian_vocab_miss": 4,
        },
    }


def _classifier_result_for_render():
    return {
        "total_misses": 4,
        "primary_causes": {
            "normalization_recoverable":        {"count": 1, "pct": 25.0, "rows": [{"sumerian": "za3sze3la2", "english": "force", "confidence": 0.9, "trace": {"normalized_form": "za3sze3la2"}}]},
            "in_corpus_below_min_count":        {"count": 1, "pct": 25.0, "rows": [{"sumerian": "rare3", "english": "x", "confidence": 0.9, "trace": {"matched_form": "rare3", "corpus_count": 3}}]},
            "oracc_lemma_surface_recoverable":  {"count": 0, "pct": 0.0, "rows": []},
            "morpheme_composition_recoverable": {"count": 1, "pct": 25.0, "rows": [{"sumerian": "nar-ta", "english": "musician", "confidence": 0.9, "trace": {"morphemes_in_vocab": ["nar", "ta"]}}]},
            "subword_inference_recoverable":    {"count": 0, "pct": 0.0, "rows": []},
            "genuinely_missing":                {"count": 1, "pct": 25.0, "rows": [{"sumerian": "weird", "english": "absent", "confidence": 0.9, "trace": {}}]},
        },
    }


def _simulator_result_for_render():
    return {
        "interventions": {
            "ascii_normalize": {"anchors_newly_resolvable": 1, "trustworthiness": "exact", "notes": "..."},
            "lower_min_count": {"per_threshold": {"1": {"anchors_newly_resolvable": 3}, "2": {"anchors_newly_resolvable": 2}, "3": {"anchors_newly_resolvable": 1}, "4": {"anchors_newly_resolvable": 0}}, "trustworthiness": "exact", "notes": "..."},
            "oracc_lemma_expansion": {"anchors_newly_resolvable": 0, "surface_forms_added_to_vocab": 0, "trustworthiness": "exact", "notes": "..."},
            "morpheme_composition": {"anchors_newly_resolvable_tier1": 1, "tier2_semantic": {"tested": 1, "top1_correct": 1, "top5_correct": 1, "top10_correct": 1, "skipped": 0}, "trustworthiness": "inferred (compositional)", "notes": "..."},
            "subword_inference": {"anchors_newly_resolvable_tier1": 1, "tier2_semantic": {"tested": 1, "top1_correct": 0, "top5_correct": 1, "top10_correct": 1, "skipped": 0}, "trustworthiness": "inferred (character n-gram)", "notes": "..."},
        },
    }


def test_render_json_schema_version_and_structure():
    from scripts.coverage_diagnostic import render_json

    report = render_json(
        _classifier_result_for_render(),
        _simulator_result_for_render(),
        _full_metadata(),
        examples_per_bucket=1,
    )
    assert report["diagnostic_schema_version"] == 1
    assert report["diagnostic_date"] == "2026-04-19"
    assert "baseline" in report
    assert "classifier" in report
    assert "simulator" in report
    assert set(report["classifier"]["primary_causes"].keys()) == {
        "normalization_recoverable", "in_corpus_below_min_count",
        "oracc_lemma_surface_recoverable", "morpheme_composition_recoverable",
        "subword_inference_recoverable", "genuinely_missing",
    }
    for bucket in report["classifier"]["primary_causes"].values():
        assert "count" in bucket
        assert "examples" in bucket
        assert "rows" not in bucket  # heavy list stripped


def test_render_json_is_deterministic_with_seed():
    from scripts.coverage_diagnostic import render_json

    r1 = render_json(
        _classifier_result_for_render(),
        _simulator_result_for_render(),
        _full_metadata(),
        examples_per_bucket=10,
    )
    r2 = render_json(
        _classifier_result_for_render(),
        _simulator_result_for_render(),
        _full_metadata(),
        examples_per_bucket=10,
    )
    import json as _json
    assert _json.dumps(r1, sort_keys=True) == _json.dumps(r2, sort_keys=True)


def test_render_markdown_has_required_sections():
    from scripts.coverage_diagnostic import render_markdown

    md = render_markdown(
        _classifier_result_for_render(),
        _simulator_result_for_render(),
        _full_metadata(),
        examples_per_bucket=1,
    )
    assert "# Coverage Diagnostic" in md
    assert "## Baseline" in md
    assert "## Classifier" in md
    assert "## Simulator" in md
    assert "## Ranked intervention" in md
    assert "## Methodology notes" in md


def test_render_markdown_escapes_pipe_characters_in_cells():
    from scripts.coverage_diagnostic import render_markdown

    classifier_with_pipe = _classifier_result_for_render()
    classifier_with_pipe["primary_causes"]["genuinely_missing"]["rows"] = [
        {"sumerian": "|AN.USAN|", "english": "night", "confidence": 0.9, "trace": {}}
    ]
    classifier_with_pipe["primary_causes"]["genuinely_missing"]["count"] = 1

    md = render_markdown(
        classifier_with_pipe,
        _simulator_result_for_render(),
        _full_metadata(),
        examples_per_bucket=1,
    )
    assert r"\|AN.USAN\|" in md


def test_run_diagnostic_exits_zero_with_synthetic_inputs(tmp_path, monkeypatch):
    """End-to-end: run_diagnostic produces both report files with synthetic artifacts."""
    import scripts.coverage_diagnostic as mod

    # Build synthetic artifacts in tmp_path.
    anchors = [
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"},      # survives
        {"sumerian": "completelyunknown", "english": "king", "confidence": 0.9, "source": "ePSD2"},  # miss
    ]
    anchors_path = tmp_path / "anchors.json"
    anchors_path.write_text(json.dumps(anchors), encoding="utf-8")

    fused_path = tmp_path / "fused.npz"
    np.savez_compressed(
        str(fused_path),
        vectors=np.zeros((1, 1536), dtype=np.float32),
        vocab=np.array(["lugal"]),
    )

    glove_path = tmp_path / "glove.txt"
    glove_path.write_text("king " + " ".join(["0.0"] * 300) + "\n", encoding="utf-8")

    gemma_path = tmp_path / "gemma.npz"
    np.savez_compressed(
        str(gemma_path),
        vectors=np.eye(1, 768, dtype=np.float32),
        vocab=np.array(["king"]),
    )

    ridge_path = tmp_path / "ridge.npz"
    coef = np.zeros((768, 1536), dtype=np.float32)
    for i in range(768):
        coef[i, i] = 1.0
    np.savez_compressed(str(ridge_path), coef=coef, intercept=np.zeros(768, dtype=np.float32))

    lemmas_path = tmp_path / "lemmas.json"
    lemmas_path.write_text(json.dumps([]), encoding="utf-8")

    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("lugal lugal\n", encoding="utf-8")

    # Train a tiny real FastText so subword inference path works.
    from gensim.models import FastText
    tiny_corpus = [["lugal", "dingir", "nar", "ta", "mu"]] * 5
    tiny_model = FastText(vector_size=768, min_count=1, sg=1, workers=1, epochs=1)
    tiny_model.build_vocab(corpus_iterable=tiny_corpus)
    tiny_model.train(corpus_iterable=tiny_corpus, total_examples=5, epochs=1)
    model_path = tmp_path / "fasttext.model"
    tiny_model.save(str(model_path))

    out_dir = tmp_path / "results"

    exit_code = mod.run_diagnostic(
        anchors_path=anchors_path,
        fused_path=fused_path,
        glove_path=glove_path,
        gemma_path=gemma_path,
        ridge_gemma_path=ridge_path,
        oracc_lemmas_path=lemmas_path,
        cleaned_corpus_path=corpus_path,
        fasttext_model_path=model_path,
        out_dir=out_dir,
        diagnostic_date="2026-04-19",
    )
    assert exit_code == 0
    assert (out_dir / "coverage_diagnostic_2026-04-19.md").exists()
    assert (out_dir / "coverage_diagnostic_2026-04-19.json").exists()

    report = json.loads((out_dir / "coverage_diagnostic_2026-04-19.json").read_text())
    assert report["diagnostic_schema_version"] == 1
    assert report["classifier"]["total_misses"] == 1


def test_run_diagnostic_raises_on_missing_required_input(tmp_path):
    import scripts.coverage_diagnostic as mod

    with pytest.raises(FileNotFoundError):
        mod.run_diagnostic(
            anchors_path=tmp_path / "missing.json",
            fused_path=tmp_path / "missing.npz",
            glove_path=tmp_path / "missing.txt",
            gemma_path=tmp_path / "missing.npz",
            ridge_gemma_path=tmp_path / "missing.npz",
            oracc_lemmas_path=tmp_path / "missing.json",
            cleaned_corpus_path=tmp_path / "missing.txt",
            fasttext_model_path=tmp_path / "missing.model",
            out_dir=tmp_path,
            diagnostic_date="2026-04-19",
        )
