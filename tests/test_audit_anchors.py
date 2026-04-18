import json
import tempfile

import numpy as np
import pytest


# --- Classification tests ---------------------------------------------------


def _default_ctx(
    fused_vocab=None,
    glove_vocab=None,
    gemma_vocab=None,
    collision_keys=None,
    low_conf_threshold=0.3,
):
    from scripts.audit_anchors import AuditContext

    return AuditContext(
        fused_vocab=set(fused_vocab or {"lugal", "dingir"}),
        glove_vocab=set(glove_vocab or {"king", "god"}),
        gemma_vocab=set(gemma_vocab or {"king", "god"}),
        collision_keys=set(collision_keys or set()),
        low_conf_threshold=low_conf_threshold,
    )


def test_survives_requires_both_target_vocabs():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"king"}, gemma_vocab={"king", "god"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "survives"

    bucket2 = classify_anchor(
        {"sumerian": "dingir", "english": "god", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket2 == "english_glove_miss"


def test_bucket_priority_ordering_sumerian_miss_beats_multiword():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(fused_vocab={"dingir"})
    bucket = classify_anchor(
        {"sumerian": "e2.gal", "english": "royal palace", "confidence": 0.9, "source": "ePSD2"},
        ctx,
    )
    assert bucket == "sumerian_vocab_miss"


def test_junk_sumerian_detection():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx()
    for bad in ["", " ", "\t", "a", "\u200b"]:
        bucket = classify_anchor(
            {"sumerian": bad, "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
        )
        assert bucket == "junk_sumerian", f"expected junk for {bad!r}, got {bucket}"


def test_low_confidence_bucket():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(low_conf_threshold=0.3)
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "king", "confidence": 0.25, "source": "ETCSL"}, ctx
    )
    assert bucket == "low_confidence"

    # Confidence exactly at threshold is NOT low_confidence (strict <).
    at_threshold = classify_anchor(
        {"sumerian": "lugal", "english": "king", "confidence": 0.3, "source": "ETCSL"}, ctx
    )
    assert at_threshold != "low_confidence"


def test_duplicate_collision_bucket():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(collision_keys={"lugal"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "monarch", "confidence": 0.5, "source": "ETCSL"}, ctx
    )
    assert bucket == "duplicate_collision"


def test_multiword_english_detection():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx()
    for phrase in ["royal palace", "to cut off", "sun-god", "cut_off"]:
        bucket = classify_anchor(
            {"sumerian": "lugal", "english": phrase, "confidence": 0.9, "source": "ePSD2"},
            ctx,
        )
        assert bucket == "multiword_english", f"expected multiword for {phrase!r}, got {bucket}"


def test_english_both_miss_when_neither_vocab_has_word():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"god"}, gemma_vocab={"god"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "english_both_miss"


def test_english_gemma_miss_when_only_glove_has_word():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"king"}, gemma_vocab={"god"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "english_gemma_miss"


def test_english_glove_miss_when_only_gemma_has_word():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"god"}, gemma_vocab={"king"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "english_glove_miss"


def test_english_lookup_is_case_insensitive():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"king"}, gemma_vocab={"king"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "KING", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "survives"


def test_classify_all_sums_to_total():
    from scripts.audit_anchors import classify_all

    ctx = _default_ctx(fused_vocab={"lugal"}, glove_vocab={"king"}, gemma_vocab={"king"})
    anchors = [
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "missing", "english": "god", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "", "english": "king", "confidence": 0.9, "source": "ePSD2"},
    ]
    result = classify_all(anchors, ctx)
    total = sum(b["count"] for b in result["buckets"].values())
    assert total == len(anchors) == 3
    assert result["buckets"]["survives"]["count"] == 1
    assert result["buckets"]["sumerian_vocab_miss"]["count"] == 1
    assert result["buckets"]["junk_sumerian"]["count"] == 1


def test_classify_all_empty_input():
    from scripts.audit_anchors import classify_all

    ctx = _default_ctx()
    result = classify_all([], ctx)
    assert result["totals"]["merged"] == 0
    assert result["totals"]["survives"] == 0
    assert result["totals"]["dropped"] == 0
    for bucket in result["buckets"].values():
        assert bucket["count"] == 0


def test_venn_accounting_over_single_token_anchors():
    from scripts.audit_anchors import classify_all

    ctx = _default_ctx(
        fused_vocab={"s1", "s2", "s3", "s4"},
        glove_vocab={"a", "b"},
        gemma_vocab={"a", "c"},
    )
    anchors = [
        {"sumerian": "s1", "english": "a", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "s2", "english": "b", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "s3", "english": "c", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "s4", "english": "d", "confidence": 0.9, "source": "ePSD2"},
    ]
    result = classify_all(anchors, ctx)
    venn = result["english_venn"]
    assert venn["in_glove_in_gemma"] == 1
    assert venn["in_glove_not_gemma"] == 1
    assert venn["not_glove_in_gemma"] == 1
    assert venn["not_glove_not_gemma"] == 1


# --- Render tests ----------------------------------------------------------


def _default_metadata():
    return {
        "audit_date": "2026-04-18",
        "source_artifacts": {
            "anchors_path": "data/processed/english_anchors.json",
            "anchors_sha256": "0" * 64,
            "fused_vocab_path": "models/fused_embeddings_1536d.npz",
            "fused_vocab_size": 2,
            "glove_path": "data/processed/glove.6B.300d.txt",
            "glove_vocab_size": 2,
            "gemma_path": "models/english_gemma_whitened_768d.npz",
            "gemma_vocab_size": 2,
            "seed": 42,
        },
    }


def test_render_json_schema_version_and_structure():
    from scripts.audit_anchors import classify_all, render_json

    ctx = _default_ctx()
    anchors = [{"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}]
    result = classify_all(anchors, ctx)
    report = render_json(result, _default_metadata(), examples_per_bucket=1)

    assert report["audit_schema_version"] == 1
    assert report["audit_date"] == "2026-04-18"
    assert "source_artifacts" in report
    assert report["totals"]["merged"] == 1
    assert set(report["buckets"].keys()) == {
        "junk_sumerian", "duplicate_collision", "low_confidence",
        "sumerian_vocab_miss", "multiword_english", "english_both_miss",
        "english_glove_miss", "english_gemma_miss", "survives",
    }
    for bucket in report["buckets"].values():
        assert "count" in bucket
        assert "pct_total" in bucket
        assert "examples" in bucket
        assert isinstance(bucket["examples"], list)


def test_render_json_is_deterministic_with_seed():
    from scripts.audit_anchors import classify_all, render_json

    ctx = _default_ctx(fused_vocab={f"s{i}" for i in range(20)})
    anchors = [
        {"sumerian": f"s{i}", "english": f"e{i}", "confidence": 0.9, "source": "ePSD2"}
        for i in range(20)
    ]
    result = classify_all(anchors, ctx)
    report_a = render_json(result, _default_metadata(), examples_per_bucket=5)
    report_b = render_json(result, _default_metadata(), examples_per_bucket=5)

    import json as _json
    assert _json.dumps(report_a, sort_keys=True) == _json.dumps(report_b, sort_keys=True)


def test_render_json_examples_respect_per_bucket_cap():
    from scripts.audit_anchors import classify_all, render_json

    ctx = _default_ctx(fused_vocab={f"s{i}" for i in range(20)})
    anchors = [
        {"sumerian": f"s{i}", "english": "king", "confidence": 0.9, "source": "ePSD2"}
        for i in range(20)
    ]
    result = classify_all(anchors, ctx)
    report = render_json(result, _default_metadata(), examples_per_bucket=3)
    assert len(report["buckets"]["survives"]["examples"]) == 3


def test_render_markdown_contains_required_sections():
    from scripts.audit_anchors import classify_all, render_markdown

    ctx = _default_ctx()
    anchors = [{"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}]
    result = classify_all(anchors, ctx)
    md = render_markdown(result, _default_metadata(), examples_per_bucket=1)

    assert "# Anchor Audit" in md
    assert "## Summary" in md
    assert "## Dropout by bucket" in md
    assert "## Cross-cut: English-side Venn" in md
    assert "## Bucket examples" in md
    assert "## Recoverability narrative" in md
    for bucket_name in ("junk_sumerian", "survives", "sumerian_vocab_miss"):
        assert bucket_name in md


def test_render_markdown_recoverability_heuristics_present():
    from scripts.audit_anchors import classify_all, render_markdown

    ctx = _default_ctx()
    anchors = [{"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}]
    result = classify_all(anchors, ctx)
    md = render_markdown(result, _default_metadata(), examples_per_bucket=1)

    assert "high" in md.lower() or "medium" in md.lower() or "low" in md.lower()
