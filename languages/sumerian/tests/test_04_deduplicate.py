import pytest


def test_dedup_by_p_number():
    """Texts with same P-number should be merged, preferring more lines."""
    from languages.sumerian.scripts.dedup_04 import deduplicate

    texts = [
        {"p_number": "P100003", "lines": ["lugal"], "source": "CDLI"},
        {"p_number": "P100003", "lines": ["lugal", "e2-gal"], "source": "ORACC"},
        {"p_number": "P100010", "lines": ["dingir"], "source": "CDLI"},
    ]

    result = deduplicate(texts)

    assert len(result) == 2
    p3 = next(t for t in result if t["p_number"] == "P100003")
    assert len(p3["lines"]) == 2
    assert p3["source"] == "ORACC"


def test_dedup_keeps_etcsl_without_p_number():
    """ETCSL texts identified by line_id (no P-number) should be kept."""
    from languages.sumerian.scripts.dedup_04 import deduplicate

    texts = [
        {"p_number": None, "lines": ["ud re-a"], "source": "ETCSL", "line_id": "c.1.1.1.1"},
        {"p_number": "P100003", "lines": ["lugal"], "source": "CDLI"},
    ]

    result = deduplicate(texts)
    assert len(result) == 2


def test_dedup_stats():
    """Dedup should return stats about what was merged."""
    from languages.sumerian.scripts.dedup_04 import deduplicate_with_stats

    texts = [
        {"p_number": "P100003", "lines": ["lugal"], "source": "CDLI"},
        {"p_number": "P100003", "lines": ["lugal", "e2"], "source": "ORACC"},
        {"p_number": "P100010", "lines": ["dingir"], "source": "ORACC"},
    ]

    result, stats = deduplicate_with_stats(texts)

    assert stats["total_input"] == 3
    assert stats["total_output"] == 2
    assert stats["duplicates_removed"] == 1
