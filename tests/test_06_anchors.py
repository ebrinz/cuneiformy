import pytest


def test_extract_epsd2_anchors():
    """Extract Sumerian-English pairs from ORACC lemma data."""
    from scripts.anchors_06 import extract_epsd2_anchors

    lemmas = [
        {"form": "lugal", "cf": "lugal", "gw": "king", "pos": "N"},
        {"form": "lugal-e", "cf": "lugal", "gw": "king", "pos": "N"},
        {"form": "e2", "cf": "e", "gw": "house", "pos": "N"},
        {"form": "an", "cf": "an", "gw": "heaven", "pos": "N"},
        {"form": "mu", "cf": "mu", "gw": "name", "pos": "N"},
        {"form": "mu", "cf": "mu", "gw": "name", "pos": "N"},
    ]

    anchors = extract_epsd2_anchors(lemmas, min_occurrences=1)

    cfs = [a["sumerian"] for a in anchors]
    assert "lugal" in cfs
    assert "e" in cfs
    assert "an" in cfs


def test_extract_epsd2_anchors_min_occurrences():
    """Anchors below min_occurrences threshold should be filtered."""
    from scripts.anchors_06 import extract_epsd2_anchors

    lemmas = [
        {"form": "lugal", "cf": "lugal", "gw": "king", "pos": "N"},
        {"form": "lugal", "cf": "lugal", "gw": "king", "pos": "N"},
        {"form": "lugal", "cf": "lugal", "gw": "king", "pos": "N"},
        {"form": "rare", "cf": "rare", "gw": "rare-word", "pos": "N"},
    ]

    anchors = extract_epsd2_anchors(lemmas, min_occurrences=2)

    cfs = [a["sumerian"] for a in anchors]
    assert "lugal" in cfs
    assert "rare" not in cfs


def test_extract_cooccurrence_anchors():
    """Extract anchors from ETCSL parallel translations via co-occurrence."""
    from scripts.anchors_06 import extract_cooccurrence_anchors

    parallel_lines = [
        {"transliteration": "lugal e2-gal-la-ka", "translation": "the king of the palace"},
        {"transliteration": "lugal kalam-ma", "translation": "the king of the land"},
        {"transliteration": "dingir gal-gal-e-ne", "translation": "the great gods"},
    ]

    anchors = extract_cooccurrence_anchors(parallel_lines, min_cooccurrences=2, min_confidence=0.3)

    sumerians = [a["sumerian"] for a in anchors]
    assert "lugal" in sumerians


def test_merge_anchors():
    """Merge dictionary and co-occurrence anchors, preferring higher confidence."""
    from scripts.anchors_06 import merge_anchors

    dict_anchors = [
        {"sumerian": "lugal", "english": "king", "confidence": 0.95, "source": "ePSD2"},
    ]
    cooc_anchors = [
        {"sumerian": "lugal", "english": "king", "confidence": 0.60, "source": "ETCSL"},
        {"sumerian": "e2", "english": "house", "confidence": 0.55, "source": "ETCSL"},
    ]

    merged = merge_anchors(dict_anchors, cooc_anchors)

    assert len(merged) == 2
    lugal = next(a for a in merged if a["sumerian"] == "lugal")
    assert lugal["confidence"] == 0.95
