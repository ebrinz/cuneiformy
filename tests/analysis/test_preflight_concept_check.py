import pytest


def _preflight_stub():
    """SumerianLookup stub with a known vocab and find_both behavior."""
    import numpy as np

    class Stub:
        def __init__(self):
            self.vocab = ["abzu", "zi", "nam", "namtar", "me", "ima", "kur"]
            self._eng_vocabs = {
                "gemma": {"deep", "breath", "essence", "fate", "decree", "clay", "mountain"},
                "glove": {"deep", "breath", "essence", "fate", "decree", "clay", "mountain"},
            }

        def find(self, english_word, top_k=10, space="gemma"):
            # Return a plausible top-K for known English words, else [].
            eng = english_word.lower()
            if eng not in self._eng_vocabs.get(space, set()):
                return []
            # Non-degenerate by default: return 5 multi-char Sumerian words.
            return [(w, 0.5 - i * 0.05) for i, w in enumerate(self.vocab[:5])]

        def find_both(self, english_word, top_k=10):
            return {
                "gemma": self.find(english_word, top_k=top_k, space="gemma"),
                "glove": self.find(english_word, top_k=top_k, space="glove"),
            }
    return Stub()


def test_passes_when_concept_fully_resolved():
    from scripts.analysis.preflight_concept_check import preflight_check

    lookup = _preflight_stub()
    concepts = [{"sumerian": "abzu", "english": "deep", "theme": "primordial"}]
    etcsl = [
        {"text_id": "t.1", "title": "", "lines": [
            {"line_no": 1, "transliteration": "abzu gal", "translation": "great deep"}
        ]}
    ]
    report = preflight_check(lookup, concepts, etcsl)
    verdict = report["concepts"][0]
    assert verdict["status"] == "pass"
    assert verdict["sumerian_in_vocab"] is True
    assert verdict["english_in_gemma"] is True
    assert verdict["etcsl_passages"] >= 1


def test_flags_vocab_miss():
    from scripts.analysis.preflight_concept_check import preflight_check

    lookup = _preflight_stub()
    concepts = [{"sumerian": "nonexistent", "english": "deep", "theme": "x"}]
    report = preflight_check(lookup, concepts, [])
    verdict = report["concepts"][0]
    assert verdict["status"] == "fail"
    assert verdict["sumerian_in_vocab"] is False
    assert "sumerian_vocab_miss" in verdict["failure_reasons"]


def test_flags_degenerate_top5():
    from scripts.analysis.preflight_concept_check import preflight_check

    # Build a lookup whose find() returns single-char degenerate results.
    class DegenerateStub:
        vocab = ["abzu", "a", "b", "c", "d", "e"]
        def find(self, english_word, top_k=10, space="gemma"):
            return [("a", 0.9), ("b", 0.8), ("c", 0.7), ("abzu", 0.6), ("d", 0.5)]
        def find_both(self, english_word, top_k=10):
            return {"gemma": self.find(english_word, top_k, "gemma"),
                    "glove": self.find(english_word, top_k, "glove")}
    lookup = DegenerateStub()

    concepts = [{"sumerian": "abzu", "english": "deep", "theme": "primordial"}]
    # Provide matching etcsl so only the degenerate-top5 flag trips.
    etcsl = [{"text_id": "t.1", "title": "",
              "lines": [{"line_no": 1, "transliteration": "abzu", "translation": "deep"}]}]
    report = preflight_check(lookup, concepts, etcsl)
    verdict = report["concepts"][0]
    assert "degenerate_top5" in verdict["warnings"]


def test_flags_zero_etcsl_passages():
    from scripts.analysis.preflight_concept_check import preflight_check

    lookup = _preflight_stub()
    concepts = [{"sumerian": "abzu", "english": "deep", "theme": "primordial"}]
    report = preflight_check(lookup, concepts, [])  # empty ETCSL
    verdict = report["concepts"][0]
    assert verdict["etcsl_passages"] == 0
    assert "zero_etcsl_passages" in verdict["failure_reasons"]
    assert verdict["status"] == "fail"


def test_etcsl_count_with_hyphenated_transliteration():
    """Concept 'namtar' should find ETCSL hits in 'nam-tar' transliterations."""
    from scripts.analysis.preflight_concept_check import preflight_check

    lookup = _preflight_stub()
    # 'namtar' is in the stub vocab; 'fate' is in both eng_vocabs.
    etcsl = [
        {"text_id": "t.1", "title": "", "lines": [
            {"line_no": 1, "transliteration": "nam-tar gal", "translation": "great fate"},
        ]},
    ]
    concepts = [{"sumerian": "namtar", "english": "fate", "theme": "decree"}]
    report = preflight_check(lookup, concepts, etcsl)
    verdict = report["concepts"][0]
    assert verdict["etcsl_passages"] >= 1
    assert verdict["status"] == "pass"


def test_report_schema_stable():
    from scripts.analysis.preflight_concept_check import preflight_check

    lookup = _preflight_stub()
    report = preflight_check(lookup, [], [])
    assert "preflight_schema_version" in report
    assert report["preflight_schema_version"] == 1
    assert "concepts" in report
    assert "preflight_date" in report
