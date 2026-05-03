import json
import os
import tempfile
import pytest


# Real ETCSL format: transliterations have <w form="..."> tags inside <l> tags
SAMPLE_TRANSLITERATION_XML = """<TEI.2 id="c.1.1.1">
<teiHeader lang="eng"><fileDesc><titleStmt>
<title>Enki and Ninhursag -- composite transliteration</title>
</titleStmt></fileDesc></teiHeader>
<text><body>
<l id="c111.1" n="1" corresp="t111.p1">
<w form="ud" lemma="ud" pos="N" label="day">ud</w>
<w form="re-a" lemma="re" pos="V" label="distant">re-a</w>
<w form="ud" lemma="ud" pos="N" label="day">ud</w>
<w form="sud-ra2" lemma="sud" pos="V" label="distant">sud-ra2</w>
<w form="re-a" lemma="re" pos="V" label="distant">re-a</w>
</l>
<l id="c111.2" n="2" corresp="t111.p1">
<w form="gi6" lemma="gi6" pos="N" label="night">gi6</w>
<w form="re-a" lemma="re" pos="V" label="distant">re-a</w>
</l>
<l id="c111.3" n="3" corresp="t111.p2">
<w form="mu" lemma="mu" pos="N" label="year">mu</w>
<w form="re-a" lemma="re" pos="V" label="distant">re-a</w>
</l>
</body></text>
</TEI.2>
"""

SAMPLE_TRANSLATION_XML = """<TEI.2 id="t.1.1.1">
<teiHeader lang="eng"><fileDesc><titleStmt>
<title>Enki and Ninhursag -- English translation</title>
</titleStmt></fileDesc></teiHeader>
<text><body>
<p id="t111.p1" n="1-2" corresp="c111.1">In those days, in those far-off days,</p>
<p id="t111.p2" n="3" corresp="c111.3">In those years, in those far-off years,</p>
</body></text>
</TEI.2>
"""


def test_parse_etcsl_transliteration():
    """Parse ETCSL transliteration XML and extract word forms from <w> tags."""
    from languages.sumerian.scripts.scrape_etcsl_01 import parse_etcsl_xml

    texts = parse_etcsl_xml(SAMPLE_TRANSLITERATION_XML)

    assert len(texts) == 3
    assert texts[0]["transliteration"] == "ud re-a ud sud-ra2 re-a"
    assert texts[0]["line_id"] == "c111.1"
    assert texts[1]["transliteration"] == "gi6 re-a"


def test_parse_translation_and_match():
    """Parse translation XML and match to transliteration lines."""
    from languages.sumerian.scripts.scrape_etcsl_01 import parse_etcsl_xml, parse_translation_xml, match_translations

    lines = parse_etcsl_xml(SAMPLE_TRANSLITERATION_XML)
    translations = parse_translation_xml(SAMPLE_TRANSLATION_XML)
    match_translations(lines, translations)

    # Line 1 corresp=t111.p1 -> "In those days..."
    assert lines[0]["translation"] == "In those days, in those far-off days,"
    # Line 3 corresp=t111.p2 -> "In those years..."
    assert lines[2]["translation"] == "In those years, in those far-off years,"


def test_parse_etcsl_xml_no_w_tags_fallback():
    """Lines without <w> tags should fall back to plain text."""
    from languages.sumerian.scripts.scrape_etcsl_01 import parse_etcsl_xml

    xml = """<TEI.2 id="c.1.1.1"><text><body>
    <l id="c111.1" n="1">lugal-e mu-un-na-ni-ib-gi4-gi4</l>
    </body></text></TEI.2>"""
    texts = parse_etcsl_xml(xml)

    assert len(texts) == 1
    assert texts[0]["transliteration"] == "lugal-e mu-un-na-ni-ib-gi4-gi4"
    assert texts[0]["translation"] is None


def test_save_etcsl_texts():
    """Save parsed texts to JSON."""
    from languages.sumerian.scripts.scrape_etcsl_01 import save_texts

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
