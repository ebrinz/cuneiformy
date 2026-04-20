import pytest


def test_returns_passages_with_token():
    from scripts.analysis.etcsl_passage_finder import find_passages

    etcsl = [
        {
            "text_id": "t.1.2",
            "title": "Enki and Ninmah",
            "lines": [
                {"line_no": 1, "transliteration": "lugal an ki", "translation": "king of heaven and earth"},
                {"line_no": 2, "transliteration": "nam-tar gal", "translation": "great fate"},
                {"line_no": 3, "transliteration": "enki dugal", "translation": "wise Enki"},
            ],
        },
    ]

    passages = find_passages("nam-tar", etcsl, max_passages=3, context_lines=1)
    assert len(passages) == 1
    assert passages[0]["text_id"] == "t.1.2"
    assert passages[0]["matched_line_no"] == 2


def test_respects_max_passages_limit():
    from scripts.analysis.etcsl_passage_finder import find_passages

    etcsl = [
        {
            "text_id": f"t.{i}",
            "title": f"Text {i}",
            "lines": [
                {"line_no": 1, "transliteration": "nam-tar x", "translation": "fate"},
            ],
        }
        for i in range(10)
    ]

    passages = find_passages("nam-tar", etcsl, max_passages=3)
    assert len(passages) == 3


def test_zero_passages_when_token_absent():
    from scripts.analysis.etcsl_passage_finder import find_passages

    etcsl = [
        {"text_id": "t.1", "title": "test", "lines": [
            {"line_no": 1, "transliteration": "lugal", "translation": "king"}
        ]}
    ]
    assert find_passages("me", etcsl) == []


def test_context_lines_captured():
    from scripts.analysis.etcsl_passage_finder import find_passages

    etcsl = [
        {
            "text_id": "t.1",
            "title": "t",
            "lines": [
                {"line_no": 1, "transliteration": "line one", "translation": "l1"},
                {"line_no": 2, "transliteration": "line two", "translation": "l2"},
                {"line_no": 3, "transliteration": "nam-tar word", "translation": "fate"},
                {"line_no": 4, "transliteration": "line four", "translation": "l4"},
                {"line_no": 5, "transliteration": "line five", "translation": "l5"},
            ],
        }
    ]

    passages = find_passages("nam-tar", etcsl, max_passages=1, context_lines=2)
    assert len(passages) == 1
    # context includes 2 lines before and 2 lines after the matched line.
    assert len(passages[0]["context"]) == 5
