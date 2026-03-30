import json
import os
import tempfile
import pytest


SAMPLE_ATF = """\
&P100003 = AAS 015
#atf: lang sux
@tablet
@obverse
1. 1(disz) geme2 u4 1(disz)-sze3
2. ki dingir-ra-ta
3. da-da-ga
4. szu ba-ti
@reverse
1. mu ki-masz{ki} ba-hul

&P100004 = AAS 016
#atf: lang akk
@tablet
@obverse
1. a-na be-li2-ia
2. qi2-bi2-ma

&P100010 = AAS 022
#atf: lang sux
@tablet
@obverse
1. 2(disz) udu niga
2. ki ab-ba-sa6-ga-ta
@reverse
1. kiszib3 lu2-{d}nanna
"""


def test_parse_atf_sumerian_only():
    """Parse ATF and return only Sumerian texts."""
    from scripts.scrape_cdli_02 import parse_atf

    texts = parse_atf(SAMPLE_ATF)

    assert len(texts) == 2
    assert texts[0]["p_number"] == "P100003"
    assert texts[1]["p_number"] == "P100010"


def test_parse_atf_lines():
    """ATF lines should be extracted with transliteration content."""
    from scripts.scrape_cdli_02 import parse_atf

    texts = parse_atf(SAMPLE_ATF)

    lines = texts[0]["lines"]
    assert len(lines) == 5  # 4 obverse + 1 reverse
    assert lines[0] == "1(disz) geme2 u4 1(disz)-sze3"
    assert lines[4] == "mu ki-masz{ki} ba-hul"


def test_parse_atf_designation():
    """ATF designation should be captured."""
    from scripts.scrape_cdli_02 import parse_atf

    texts = parse_atf(SAMPLE_ATF)
    assert texts[0]["designation"] == "AAS 015"


def test_save_cdli_texts():
    """Save parsed CDLI texts to JSON."""
    from scripts.scrape_cdli_02 import save_texts

    texts = [{"p_number": "P100003", "lines": ["lugal"], "designation": "AAS 015"}]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "cdli_texts.json")
        save_texts(texts, out_path)

        with open(out_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["p_number"] == "P100003"
