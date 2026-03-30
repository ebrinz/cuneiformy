import json
import os
import tempfile
import pytest


SAMPLE_ORACC_TEXT = {
    "type": "cdl",
    "project": "epsd2/admin/ur3",
    "cdl": [
        {
            "node": "c",
            "id": "P100003.1",
            "cdl": [
                {
                    "node": "c",
                    "type": "line-start",
                    "cdl": [
                        {
                            "node": "d",
                            "ftype": "line-start",
                            "fragment": "1.",
                        },
                        {
                            "node": "l",
                            "frag": "lugal",
                            "id": "P100003.l0001",
                            "f": {
                                "lang": "sux",
                                "form": "lugal",
                                "cf": "lugal",
                                "gw": "king",
                                "pos": "N",
                                "norm": "lugal",
                            },
                        },
                        {
                            "node": "l",
                            "frag": "e2",
                            "id": "P100003.l0002",
                            "f": {
                                "lang": "sux",
                                "form": "e2",
                                "cf": "e",
                                "gw": "house",
                                "pos": "N",
                                "norm": "e",
                            },
                        },
                    ],
                }
            ],
        }
    ],
}


def test_extract_lemmas_from_oracc_json():
    """Extract lemmatized words with their English glosses."""
    from scripts.scrape_oracc_03 import extract_lemmas

    lemmas = extract_lemmas(SAMPLE_ORACC_TEXT)

    assert len(lemmas) == 2
    assert lemmas[0]["form"] == "lugal"
    assert lemmas[0]["gw"] == "king"
    assert lemmas[0]["cf"] == "lugal"
    assert lemmas[1]["gw"] == "house"


def test_extract_lines_from_oracc_json():
    """Extract transliteration lines from ORACC JSON."""
    from scripts.scrape_oracc_03 import extract_lines

    lines = extract_lines(SAMPLE_ORACC_TEXT)

    assert len(lines) == 1
    assert "lugal" in lines[0]
    assert "e2" in lines[0]


def test_extract_lemmas_skips_non_sumerian():
    """Non-Sumerian words should be skipped."""
    from scripts.scrape_oracc_03 import extract_lemmas

    text = {
        "cdl": [
            {
                "node": "c",
                "cdl": [
                    {
                        "node": "l",
                        "f": {
                            "lang": "akk",
                            "form": "sharrum",
                            "cf": "sharru",
                            "gw": "king",
                            "pos": "N",
                        },
                    }
                ],
            }
        ]
    }

    lemmas = extract_lemmas(text)
    assert len(lemmas) == 0
