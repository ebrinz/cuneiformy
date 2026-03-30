import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest


SAMPLE_ETCSL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<teiCorpus.2>
<TEI.2>
<teiHeader>
<fileDesc>
<titleStmt><title>Enki and Ninhursag -- ancient transliteration</title></titleStmt>
</fileDesc>
</teiHeader>
<text>
<group>
<text n="t.1.1.1">
<body>
<l id="c.1.1.1.1" n="1">ud re-a ud sud-ra2 re-a</l>
<l id="c.1.1.1.2" n="2">gi6 re-a gi6 ba9-ra2 re-a</l>
<l id="c.1.1.1.3" n="3">mu re-a mu sud-ra2 re-a</l>
</body>
</text>
<text n="t.1.1.1.e">
<body>
<p corresp="c.1.1.1.1">In those days, in those far-off days,</p>
<p corresp="c.1.1.1.2">In those nights, in those far-off nights,</p>
<p corresp="c.1.1.1.3">In those years, in those far-off years,</p>
</body>
</text>
</group>
</text>
</TEI.2>
</teiCorpus.2>
"""


def test_parse_etcsl_xml():
    """Parse ETCSL XML and extract transliteration-translation pairs."""
    from scripts.scrape_etcsl_01 import parse_etcsl_xml

    texts = parse_etcsl_xml(SAMPLE_ETCSL_XML)

    assert len(texts) == 3
    assert texts[0]["transliteration"] == "ud re-a ud sud-ra2 re-a"
    assert texts[0]["translation"] == "In those days, in those far-off days,"
    assert texts[0]["line_id"] == "c.1.1.1.1"


def test_parse_etcsl_xml_unmatched_lines():
    """Lines without matching translations should still be included."""
    from scripts.scrape_etcsl_01 import parse_etcsl_xml

    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <teiCorpus.2>
    <TEI.2>
    <text><group>
    <text n="t.1.1.1">
    <body>
    <l id="c.1.1.1.1" n="1">lugal-e mu-un-na-ni-ib-gi4-gi4</l>
    </body>
    </text>
    <text n="t.1.1.1.e"><body></body></text>
    </group></text>
    </TEI.2>
    </teiCorpus.2>
    """
    texts = parse_etcsl_xml(xml)

    assert len(texts) == 1
    assert texts[0]["transliteration"] == "lugal-e mu-un-na-ni-ib-gi4-gi4"
    assert texts[0]["translation"] is None


def test_save_etcsl_texts():
    """Save parsed texts to JSON."""
    from scripts.scrape_etcsl_01 import save_texts

    texts = [
        {"transliteration": "lugal", "translation": "king", "line_id": "c.1.1.1.1"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "etcsl_texts.json")
        save_texts(texts, out_path)

        with open(out_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["transliteration"] == "lugal"
