# Sumerian-English Embedding Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Sumerian-to-English semantic alignment pipeline that maps Sumerian transliterated text into GloVe 300d English space using the proven heiroglyphy V15 architecture.

**Architecture:** Scrape three Sumerian corpora (ETCSL, CDLI, ORACC), extract anchor pairs from ePSD2 dictionary + ETCSL parallel translations, train FastText 768d embeddings on the merged corpus, fuse with 768d zero-padding, and align to GloVe 300d via Ridge Regression.

**Tech Stack:** Python 3.8+, gensim (FastText), scikit-learn (Ridge), numpy, scipy, pandas, requests, beautifulsoup4, lxml

---

## File Structure

```
cuneiformy/
├── scripts/
│   ├── 01_scrape_etcsl.py          # Download & parse ETCSL XML corpus
│   ├── 02_scrape_cdli.py           # Parse CDLI ATF bulk dump (Sumerian only)
│   ├── 03_scrape_oracc.py          # Download ORACC project JSON zips
│   ├── 04_deduplicate_corpus.py    # Merge & dedup across 3 sources by P-number
│   ├── 05_clean_and_tokenize.py    # ATF cleanup -> cleaned_corpus.txt
│   ├── 06_extract_anchors.py       # ePSD2 dict + ETCSL co-occurrence -> anchors
│   ├── 07_train_fasttext.py        # 768d skip-gram on cleaned corpus
│   ├── 08_fuse_embeddings.py       # text 768d + zeros 768d = 1536d
│   ├── 09_align_and_evaluate.py    # Ridge regression + Top-1/5/10 eval
│   └── 10_export_production.py     # Package vectors, vocab, lookup API
├── data/
│   ├── raw/                        # Scraped source data
│   ├── processed/                  # Cleaned corpus, anchors
│   └── dictionaries/              # ePSD2 entries
├── models/                         # Trained FastText models
├── results/                        # Evaluation results JSON
├── final_output/                   # Production vectors + lookup API
│   ├── sumerian_aligned_vectors.npz
│   ├── sumerian_aligned_vocab.pkl
│   ├── sumerian_lookup.py
│   └── metadata.json
├── tests/
│   ├── test_01_scrape_etcsl.py
│   ├── test_02_scrape_cdli.py
│   ├── test_03_scrape_oracc.py
│   ├── test_04_deduplicate.py
│   ├── test_05_clean.py
│   ├── test_06_anchors.py
│   ├── test_07_fasttext.py
│   ├── test_08_fuse.py
│   ├── test_09_align.py
│   └── test_10_export.py
├── requirements.txt
└── pytest.ini
```

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: all directories

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p scripts data/raw data/processed data/dictionaries models results final_output tests
```

- [ ] **Step 2: Write requirements.txt**

```
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
gensim>=4.3.0
pandas>=2.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
tqdm>=4.66.0
pytest>=7.4.0
```

- [ ] **Step 3: Write pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
```

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini
git commit -m "feat: scaffold project with dependencies and test config"
```

---

### Task 2: ETCSL Scraper

**Files:**
- Create: `scripts/01_scrape_etcsl.py`
- Create: `tests/test_01_scrape_etcsl.py`
- Output: `data/raw/etcsl_texts.json`

ETCSL is available as a downloadable XML corpus from Oxford Text Archive. The XML uses TEI P4 format where `<l>` tags have `id` attributes and translation `<p>` tags have `corresp` attributes linking them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_01_scrape_etcsl.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_01_scrape_etcsl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.scrape_etcsl_01'`

- [ ] **Step 3: Write the ETCSL scraper**

```python
# scripts/01_scrape_etcsl.py
"""
ETCSL Scraper: Download and parse the Electronic Text Corpus of Sumerian Literature.

Source: Oxford Text Archive (OTA) - https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2518
Format: TEI P4 XML with transliterations linked to translations via id/corresp attributes.
"""
import json
import os
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from tqdm import tqdm

# OTA download URL for ETCSL corpus
ETCSL_OTA_URL = "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2518/etcsl.zip"

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"


def download_etcsl(output_dir: Path) -> Path:
    """Download ETCSL ZIP from Oxford Text Archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "etcsl.zip"

    if zip_path.exists():
        print(f"ETCSL ZIP already downloaded: {zip_path}")
        return zip_path

    print(f"Downloading ETCSL from OTA...")
    response = requests.get(ETCSL_OTA_URL, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return zip_path


def extract_xml_files(zip_path: Path) -> list[str]:
    """Extract XML content from ETCSL ZIP archive."""
    xml_contents = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".xml"):
                xml_contents.append(zf.read(name).decode("utf-8", errors="replace"))
    return xml_contents


def parse_etcsl_xml(xml_content: str) -> list[dict]:
    """
    Parse ETCSL TEI P4 XML and extract transliteration-translation pairs.

    Transliterations are in <l id="c.X.X.X.N"> tags.
    Translations are in <p corresp="c.X.X.X.N"> tags.
    They link via the id/corresp attribute.
    """
    # Strip namespace declarations that may interfere with parsing
    xml_content = xml_content.replace(' xmlns="', ' xmlns:ignore="')

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return []

    # Build translation lookup: corresp -> translation text
    translations = {}
    for p_tag in root.iter("p"):
        corresp = p_tag.get("corresp")
        if corresp:
            text = "".join(p_tag.itertext()).strip()
            if text:
                translations[corresp] = text

    # Extract transliteration lines
    texts = []
    for l_tag in root.iter("l"):
        line_id = l_tag.get("id")
        if not line_id:
            continue

        transliteration = "".join(l_tag.itertext()).strip()
        if not transliteration:
            continue

        translation = translations.get(line_id)

        texts.append({
            "transliteration": transliteration,
            "translation": translation,
            "line_id": line_id,
            "source": "ETCSL",
        })

    return texts


def save_texts(texts: list[dict], output_path: str) -> None:
    """Save parsed texts to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(texts)} lines to {output_path}")


def main():
    zip_path = download_etcsl(DATA_RAW)
    xml_files = extract_xml_files(zip_path)
    print(f"Found {len(xml_files)} XML files in archive")

    all_texts = []
    for xml_content in tqdm(xml_files, desc="Parsing XML"):
        texts = parse_etcsl_xml(xml_content)
        all_texts.extend(texts)

    output_path = DATA_RAW / "etcsl_texts.json"
    save_texts(all_texts, str(output_path))

    # Stats
    with_translation = sum(1 for t in all_texts if t["translation"])
    print(f"Total lines: {len(all_texts)}")
    print(f"With translations: {with_translation}")
    print(f"Without translations: {len(all_texts) - with_translation}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Make scripts importable for tests**

The tests import from `scripts.scrape_etcsl_01`. To handle the numbered filename convention, add an `__init__.py` shim:

```python
# scripts/__init__.py
```

```python
# scripts/scrape_etcsl_01.py
# Re-export from the actual script for test imports
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "scrape_etcsl",
    os.path.join(os.path.dirname(__file__), "01_scrape_etcsl.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

parse_etcsl_xml = _mod.parse_etcsl_xml
save_texts = _mod.save_texts
download_etcsl = _mod.download_etcsl
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_01_scrape_etcsl.py -v`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/01_scrape_etcsl.py scripts/__init__.py scripts/scrape_etcsl_01.py tests/test_01_scrape_etcsl.py
git commit -m "feat: add ETCSL scraper with XML parsing and translation linking"
```

---

### Task 3: CDLI Scraper (ATF Parser)

**Files:**
- Create: `scripts/02_scrape_cdli.py`
- Create: `tests/test_02_scrape_cdli.py`
- Output: `data/raw/cdli_texts.json`

CDLI provides a bulk ATF dump via GitHub (`cdli-gh/data`). We parse ATF format and filter to Sumerian (`#atf: lang sux`) texts only.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_02_scrape_cdli.py
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

    # Should have 2 Sumerian texts, not the Akkadian one
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_02_scrape_cdli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the CDLI ATF parser**

```python
# scripts/02_scrape_cdli.py
"""
CDLI Scraper: Parse CDLI bulk ATF dump for Sumerian transliterations.

Source: https://github.com/cdli-gh/data (cdliatf_unblocked.atf)
Format: ATF (ASCII Transliteration Format)
Filter: Only texts tagged #atf: lang sux (Sumerian)
"""
import json
import os
import re
import subprocess
from pathlib import Path

from tqdm import tqdm

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
CDLI_REPO = DATA_RAW / "cdli-data"
ATF_FILE = CDLI_REPO / "cdliatf_unblocked.atf"


def download_cdli(output_dir: Path) -> Path:
    """Clone CDLI data repo (with LFS) if not already present."""
    repo_dir = output_dir / "cdli-data"
    if repo_dir.exists() and ATF_FILE.exists():
        print(f"CDLI data already present: {ATF_FILE}")
        return ATF_FILE

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Cloning CDLI data repository (this may take a while)...")
    subprocess.run(
        ["git", "lfs", "install"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/cdli-gh/data", str(repo_dir)],
        check=True,
    )
    return ATF_FILE


def parse_atf(atf_content: str) -> list[dict]:
    """
    Parse ATF format and extract Sumerian-only texts.

    ATF structure:
        &P###### = Designation
        #atf: lang sux
        @tablet
        @obverse
        1. transliteration line
        2. another line
        @reverse
        1. reverse line

    Returns list of dicts with p_number, designation, and lines.
    """
    texts = []
    current_text = None
    is_sumerian = False

    for line in atf_content.split("\n"):
        line = line.strip()

        # New text block
        if line.startswith("&"):
            # Save previous text if it was Sumerian
            if current_text and is_sumerian and current_text["lines"]:
                texts.append(current_text)

            # Parse P-number and designation
            match = re.match(r"&(P\d+)\s*=\s*(.*)", line)
            if match:
                current_text = {
                    "p_number": match.group(1),
                    "designation": match.group(2).strip(),
                    "lines": [],
                    "source": "CDLI",
                }
                is_sumerian = False
            else:
                current_text = None
                is_sumerian = False
            continue

        if current_text is None:
            continue

        # Language tag
        if line.startswith("#atf: lang"):
            lang = line.replace("#atf: lang", "").strip()
            is_sumerian = lang.startswith("sux")
            continue

        # Skip structural markers and comments
        if line.startswith("@") or line.startswith("#") or line.startswith("$"):
            continue

        # Skip empty lines
        if not line:
            continue

        # Transliteration lines: "N. text" or "N'. text"
        if is_sumerian and re.match(r"\d+[\.']\s", line):
            # Strip line number prefix
            text = re.sub(r"^\d+[\.']\s+", "", line)
            if text:
                current_text["lines"].append(text)

    # Don't forget the last text
    if current_text and is_sumerian and current_text["lines"]:
        texts.append(current_text)

    return texts


def save_texts(texts: list[dict], output_path: str) -> None:
    """Save parsed texts to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(texts)} Sumerian texts to {output_path}")


def main():
    atf_path = download_cdli(DATA_RAW)

    print(f"Reading ATF file: {atf_path}")
    with open(atf_path, encoding="utf-8", errors="replace") as f:
        atf_content = f.read()

    texts = parse_atf(atf_content)

    output_path = DATA_RAW / "cdli_texts.json"
    save_texts(texts, str(output_path))

    total_lines = sum(len(t["lines"]) for t in texts)
    print(f"Sumerian texts: {len(texts)}")
    print(f"Total lines: {total_lines}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create import shim**

```python
# scripts/scrape_cdli_02.py
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "scrape_cdli",
    os.path.join(os.path.dirname(__file__), "02_scrape_cdli.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

parse_atf = _mod.parse_atf
save_texts = _mod.save_texts
download_cdli = _mod.download_cdli
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_02_scrape_cdli.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/02_scrape_cdli.py scripts/scrape_cdli_02.py tests/test_02_scrape_cdli.py
git commit -m "feat: add CDLI ATF parser filtering Sumerian texts by language tag"
```

---

### Task 4: ORACC Scraper

**Files:**
- Create: `scripts/03_scrape_oracc.py`
- Create: `tests/test_03_scrape_oracc.py`
- Output: `data/raw/oracc_texts.json`

ORACC provides JSON zip archives per project. Each text has lemmatized words with `cf` (citation form) and `gw` (guide word / English gloss). We target Sumerian-heavy projects.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_03_scrape_oracc.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_03_scrape_oracc.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the ORACC scraper**

```python
# scripts/03_scrape_oracc.py
"""
ORACC Scraper: Download and parse ORACC project JSON archives.

Source: http://oracc.museum.upenn.edu/{PROJECT}/json.zip
Format: Hierarchical JSON with lemmatized Sumerian words.

Each word node with an 'f' key contains:
  - form: transliteration
  - cf: citation form (dictionary headword)
  - gw: guide word (English gloss)
  - pos: part of speech
  - lang: language code
"""
import json
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

# Sumerian-heavy ORACC projects
ORACC_PROJECTS = [
    "epsd2/admin/ur3",
    "epsd2/admin/ed3b",
    "epsd2/admin/ed3a",
    "epsd2/literary",
    "etcsri",
    "dcclt",
    "cams",
]

ORACC_BASE_URL = "http://oracc.museum.upenn.edu"


def download_project_json(project: str, output_dir: Path) -> Path | None:
    """Download a project's json.zip archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"oracc_{project.replace('/', '_')}.zip"

    if zip_path.exists():
        return zip_path

    url = f"{ORACC_BASE_URL}/{project}/json.zip"
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)
        return zip_path
    except requests.RequestException as e:
        print(f"  Failed to download {project}: {e}")
        return None


def _walk_cdl(node: Any, lemmas: list[dict], line_words: list[list[str]], current_line: list[str]) -> None:
    """Recursively walk ORACC CDL JSON tree to extract lemmas and lines."""
    if isinstance(node, dict):
        # Lemma node: has 'f' key with language data
        if "f" in node:
            f = node["f"]
            lang = f.get("lang", "")
            if lang.startswith("sux"):
                lemma = {
                    "form": f.get("form", ""),
                    "cf": f.get("cf", ""),
                    "gw": f.get("gw", ""),
                    "pos": f.get("pos", ""),
                    "norm": f.get("norm", ""),
                }
                if lemma["form"]:
                    lemmas.append(lemma)
                    current_line.append(lemma["form"])

        # Line boundary detection
        if node.get("ftype") == "line-start" or node.get("type") == "line-start":
            if current_line:
                line_words.append(list(current_line))
                current_line.clear()

        # Recurse into children
        if "cdl" in node:
            for child in node["cdl"]:
                _walk_cdl(child, lemmas, line_words, current_line)

    elif isinstance(node, list):
        for child in node:
            _walk_cdl(child, lemmas, line_words, current_line)


def extract_lemmas(text_json: dict) -> list[dict]:
    """Extract all Sumerian lemmatized words from an ORACC text JSON."""
    lemmas = []
    line_words = []
    current_line = []
    _walk_cdl(text_json.get("cdl", []), lemmas, line_words, current_line)
    return lemmas


def extract_lines(text_json: dict) -> list[str]:
    """Extract transliteration lines from an ORACC text JSON."""
    lemmas = []
    line_words = []
    current_line = []
    _walk_cdl(text_json.get("cdl", []), lemmas, line_words, current_line)
    # Flush last line
    if current_line:
        line_words.append(current_line)
    return [" ".join(words) for words in line_words if words]


def parse_project_zip(zip_path: Path) -> tuple[list[dict], list[dict]]:
    """Parse all text JSONs from a project ZIP. Returns (all_lemmas, all_texts)."""
    all_lemmas = []
    all_texts = []

    with zipfile.ZipFile(zip_path) as zf:
        json_files = [n for n in zf.namelist() if "corpusjson" in n and n.endswith(".json")]

        for name in json_files:
            try:
                data = json.loads(zf.read(name))
            except (json.JSONDecodeError, KeyError):
                continue

            # Extract P-number from filename
            p_number = Path(name).stem  # e.g., "P100003"

            lemmas = extract_lemmas(data)
            lines = extract_lines(data)

            if lemmas:
                all_lemmas.extend(lemmas)

            if lines:
                all_texts.append({
                    "p_number": p_number,
                    "lines": lines,
                    "source": "ORACC",
                })

    return all_lemmas, all_texts


def save_texts(texts: list[dict], output_path: str) -> None:
    """Save parsed texts to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(texts)} texts to {output_path}")


def main():
    oracc_dir = DATA_RAW / "oracc"
    oracc_dir.mkdir(parents=True, exist_ok=True)

    all_texts = []
    all_lemmas = []

    for project in tqdm(ORACC_PROJECTS, desc="Downloading ORACC projects"):
        print(f"\nProcessing {project}...")
        zip_path = download_project_json(project, oracc_dir)
        if zip_path is None:
            continue

        lemmas, texts = parse_project_zip(zip_path)
        all_lemmas.extend(lemmas)
        all_texts.extend(texts)
        print(f"  {len(texts)} texts, {len(lemmas)} lemmas")

    # Save texts for corpus
    save_texts(all_texts, str(DATA_RAW / "oracc_texts.json"))

    # Save lemmas separately (useful for anchor extraction)
    save_texts(all_lemmas, str(DATA_RAW / "oracc_lemmas.json"))

    total_lines = sum(len(t["lines"]) for t in all_texts)
    unique_glosses = len({l["gw"] for l in all_lemmas if l["gw"]})
    print(f"\nTotal texts: {len(all_texts)}")
    print(f"Total lines: {total_lines}")
    print(f"Total lemmas: {len(all_lemmas)}")
    print(f"Unique English glosses: {unique_glosses}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create import shim**

```python
# scripts/scrape_oracc_03.py
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "scrape_oracc",
    os.path.join(os.path.dirname(__file__), "03_scrape_oracc.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_lemmas = _mod.extract_lemmas
extract_lines = _mod.extract_lines
save_texts = _mod.save_texts
parse_project_zip = _mod.parse_project_zip
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_03_scrape_oracc.py -v`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/03_scrape_oracc.py scripts/scrape_oracc_03.py tests/test_03_scrape_oracc.py
git commit -m "feat: add ORACC JSON scraper with lemma extraction for Sumerian"
```

---

### Task 5: Corpus Deduplication

**Files:**
- Create: `scripts/04_deduplicate_corpus.py`
- Create: `tests/test_04_deduplicate.py`
- Output: `data/processed/merged_corpus.json`

Merge ETCSL, CDLI, and ORACC texts. Deduplicate by CDLI P-number where available. ETCSL texts without P-numbers are kept as-is.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_04_deduplicate.py
import pytest


def test_dedup_by_p_number():
    """Texts with same P-number should be merged, preferring more lines."""
    from scripts.dedup_04 import deduplicate

    texts = [
        {"p_number": "P100003", "lines": ["lugal"], "source": "CDLI"},
        {"p_number": "P100003", "lines": ["lugal", "e2-gal"], "source": "ORACC"},
        {"p_number": "P100010", "lines": ["dingir"], "source": "CDLI"},
    ]

    result = deduplicate(texts)

    assert len(result) == 2
    # Should keep the one with more lines
    p3 = next(t for t in result if t["p_number"] == "P100003")
    assert len(p3["lines"]) == 2
    assert p3["source"] == "ORACC"


def test_dedup_keeps_etcsl_without_p_number():
    """ETCSL texts identified by line_id (no P-number) should be kept."""
    from scripts.dedup_04 import deduplicate

    texts = [
        {"p_number": None, "lines": ["ud re-a"], "source": "ETCSL", "line_id": "c.1.1.1.1"},
        {"p_number": "P100003", "lines": ["lugal"], "source": "CDLI"},
    ]

    result = deduplicate(texts)
    assert len(result) == 2


def test_dedup_stats():
    """Dedup should return stats about what was merged."""
    from scripts.dedup_04 import deduplicate_with_stats

    texts = [
        {"p_number": "P100003", "lines": ["lugal"], "source": "CDLI"},
        {"p_number": "P100003", "lines": ["lugal", "e2"], "source": "ORACC"},
        {"p_number": "P100010", "lines": ["dingir"], "source": "ORACC"},
    ]

    result, stats = deduplicate_with_stats(texts)

    assert stats["total_input"] == 3
    assert stats["total_output"] == 2
    assert stats["duplicates_removed"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_04_deduplicate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write deduplication script**

```python
# scripts/04_deduplicate_corpus.py
"""
Corpus Deduplication: Merge ETCSL, CDLI, and ORACC texts.

Deduplicates by CDLI P-number. When duplicates exist, keeps the version
with the most transliteration lines. Texts without P-numbers (some ETCSL)
are kept as-is.
"""
import json
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def deduplicate(texts: list[dict]) -> list[dict]:
    """Deduplicate texts by P-number, keeping the version with more lines."""
    result, _ = deduplicate_with_stats(texts)
    return result


def deduplicate_with_stats(texts: list[dict]) -> tuple[list[dict], dict]:
    """Deduplicate and return stats."""
    total_input = len(texts)

    # Separate texts with and without P-numbers
    no_p = [t for t in texts if not t.get("p_number")]
    with_p = [t for t in texts if t.get("p_number")]

    # Group by P-number, keep version with most lines
    best = {}
    for t in with_p:
        p = t["p_number"]
        if p not in best or len(t.get("lines", [])) > len(best[p].get("lines", [])):
            best[p] = t

    result = list(best.values()) + no_p

    stats = {
        "total_input": total_input,
        "total_output": len(result),
        "duplicates_removed": total_input - len(result),
        "texts_without_p_number": len(no_p),
        "unique_p_numbers": len(best),
    }

    return result, stats


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    all_texts = []

    # Load ETCSL -- convert line-based format to text-based
    etcsl_path = DATA_RAW / "etcsl_texts.json"
    if etcsl_path.exists():
        with open(etcsl_path) as f:
            etcsl_lines = json.load(f)
        # Group ETCSL lines by composition (first 3 parts of line_id)
        compositions = {}
        for line in etcsl_lines:
            parts = line["line_id"].rsplit(".", 1)[0]  # e.g., "c.1.1.1"
            if parts not in compositions:
                compositions[parts] = {
                    "p_number": None,
                    "lines": [],
                    "translations": [],
                    "source": "ETCSL",
                    "composition_id": parts,
                }
            compositions[parts]["lines"].append(line["transliteration"])
            if line.get("translation"):
                compositions[parts]["translations"].append(line["translation"])
        all_texts.extend(compositions.values())
        print(f"ETCSL: {len(compositions)} compositions from {len(etcsl_lines)} lines")

    # Load CDLI
    cdli_path = DATA_RAW / "cdli_texts.json"
    if cdli_path.exists():
        with open(cdli_path) as f:
            cdli_texts = json.load(f)
        all_texts.extend(cdli_texts)
        print(f"CDLI: {len(cdli_texts)} texts")

    # Load ORACC
    oracc_path = DATA_RAW / "oracc_texts.json"
    if oracc_path.exists():
        with open(oracc_path) as f:
            oracc_texts = json.load(f)
        all_texts.extend(oracc_texts)
        print(f"ORACC: {len(oracc_texts)} texts")

    # Deduplicate
    result, stats = deduplicate_with_stats(all_texts)

    # Save
    output_path = DATA_PROCESSED / "merged_corpus.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nDeduplication stats: {json.dumps(stats, indent=2)}")
    total_lines = sum(len(t.get("lines", [])) for t in result)
    print(f"Total texts: {len(result)}")
    print(f"Total lines: {total_lines}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create import shim**

```python
# scripts/dedup_04.py
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "dedup",
    os.path.join(os.path.dirname(__file__), "04_deduplicate_corpus.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

deduplicate = _mod.deduplicate
deduplicate_with_stats = _mod.deduplicate_with_stats
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_04_deduplicate.py -v`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/04_deduplicate_corpus.py scripts/dedup_04.py tests/test_04_deduplicate.py
git commit -m "feat: add corpus deduplication merging ETCSL, CDLI, ORACC by P-number"
```

---

### Task 6: Corpus Cleaning & Tokenization

**Files:**
- Create: `scripts/05_clean_and_tokenize.py`
- Create: `tests/test_05_clean.py`
- Output: `data/processed/cleaned_corpus.txt`

Clean ATF transliterations: strip editorial markers, normalize separators, produce one line per text for FastText training.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_05_clean.py
import pytest


def test_clean_atf_line():
    """Clean ATF editorial markers and normalize."""
    from scripts.clean_05 import clean_atf_line

    # Damaged signs, determinatives, corrections
    assert clean_atf_line("1(disz) geme2 u4 1(disz)-sze3") == "1(disz) geme2 u4 1(disz) sze3"
    assert clean_atf_line("ki-masz{ki}") == "ki masz ki"
    assert clean_atf_line("[lugal]-e") == "lugal e"
    assert clean_atf_line("mu!(BU)") == "mu"
    assert clean_atf_line("dingir#") == "dingir"
    assert clean_atf_line("a?-ba") == "a ba"


def test_clean_atf_line_whitespace():
    """Multiple spaces should collapse to one."""
    from scripts.clean_05 import clean_atf_line

    assert clean_atf_line("lugal    e2") == "lugal e2"


def test_clean_atf_line_empty():
    """Lines that become empty after cleaning should return empty string."""
    from scripts.clean_05 import clean_atf_line

    assert clean_atf_line("[...]") == ""
    assert clean_atf_line("$ broken") == ""


def test_build_corpus():
    """Build cleaned corpus from merged texts."""
    from scripts.clean_05 import build_corpus

    texts = [
        {"lines": ["lugal-e mu-un-na-ni-ib-gi4-gi4", "e2-gal-la ba-an-ku4"]},
        {"lines": ["dingir gal-gal-e-ne"]},
    ]

    lines = build_corpus(texts)

    assert len(lines) == 2  # One line per text (lines joined)
    assert "lugal" in lines[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_05_clean.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the cleaning script**

```python
# scripts/05_clean_and_tokenize.py
"""
Corpus Cleaning: Convert ATF transliterations to clean, tokenized text for FastText.

ATF cleanup:
- Strip editorial markers: [...], (...), #, !, ?
- Replace hyphens with spaces (morpheme boundaries)
- Remove determinatives notation: {ki}, {d}, etc. -> keep the content
- Remove line numbers and numerical notations
- Normalize whitespace

Output: cleaned_corpus.txt (one line per text, space-separated tokens)
"""
import json
import re
from pathlib import Path

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def clean_atf_line(line: str) -> str:
    """Clean a single ATF transliteration line."""
    # Skip dollar-line (structural comments)
    if line.startswith("$"):
        return ""

    # Remove correction notations: word!(SIGN) -> word
    line = re.sub(r"!\([^)]*\)", "", line)

    # Remove square brackets (restoration markers) but keep content
    line = line.replace("[", "").replace("]", "")

    # Remove parentheses (interpolation) but keep content inside
    # But preserve number notations like 1(disz)
    # Strategy: keep (content) when preceded by a digit, remove otherwise
    line = re.sub(r"(?<!\d)\(([^)]*)\)", r"\1", line)

    # Remove damage marker #
    line = line.replace("#", "")

    # Remove uncertainty marker ?
    line = line.replace("?", "")

    # Handle determinatives {ki}, {d}, etc. - keep the content
    line = re.sub(r"\{([^}]*)\}", r" \1 ", line)

    # Replace hyphens with spaces (morpheme boundaries)
    line = line.replace("-", " ")

    # Replace dots used as separators
    line = line.replace(".", " ")

    # Normalize whitespace
    line = re.sub(r"\s+", " ", line).strip()

    # Remove lines that are only whitespace or ellipsis
    if not line or line == "...":
        return ""

    return line


def build_corpus(texts: list[dict]) -> list[str]:
    """
    Build cleaned corpus lines from merged texts.

    Each text's lines are cleaned and joined into a single line,
    matching the one-sentence-per-line format FastText expects.
    """
    corpus_lines = []
    for text in texts:
        cleaned_words = []
        for line in text.get("lines", []):
            cleaned = clean_atf_line(line)
            if cleaned:
                cleaned_words.append(cleaned)
        if cleaned_words:
            corpus_lines.append(" ".join(cleaned_words))
    return corpus_lines


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Load merged corpus
    merged_path = DATA_PROCESSED / "merged_corpus.json"
    with open(merged_path) as f:
        texts = json.load(f)

    corpus_lines = build_corpus(texts)

    # Write cleaned corpus
    output_path = DATA_PROCESSED / "cleaned_corpus.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line + "\n")

    # Stats
    total_tokens = sum(len(line.split()) for line in corpus_lines)
    vocab = set()
    for line in corpus_lines:
        vocab.update(line.split())

    print(f"Corpus lines: {len(corpus_lines)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Unique tokens: {len(vocab)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create import shim**

```python
# scripts/clean_05.py
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "clean",
    os.path.join(os.path.dirname(__file__), "05_clean_and_tokenize.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

clean_atf_line = _mod.clean_atf_line
build_corpus = _mod.build_corpus
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_05_clean.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/05_clean_and_tokenize.py scripts/clean_05.py tests/test_05_clean.py
git commit -m "feat: add ATF cleaning and tokenization for Sumerian transliterations"
```

---

### Task 7: Anchor Extraction

**Files:**
- Create: `scripts/06_extract_anchors.py`
- Create: `tests/test_06_anchors.py`
- Output: `data/processed/english_anchors.json`, `data/dictionaries/epsd2_entries.json`

Two anchor sources: (1) ePSD2 dictionary entries via ORACC lemma data, (2) ETCSL co-occurrence analysis on parallel translations.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_06_anchors.py
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

    # Should deduplicate by (cf, gw) pair
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
    # "lugal" appears in 2 lines where "king" also appears -> should be anchor
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
    assert lugal["confidence"] == 0.95  # kept higher confidence
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_06_anchors.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the anchor extraction script**

```python
# scripts/06_extract_anchors.py
"""
Anchor Extraction: Build Sumerian-English word pairs from two sources.

Source 1 (ePSD2): ORACC lemma data provides citation forms with English guide words.
  - High confidence, dictionary-quality pairs
  - ~7K+ unique entries expected

Source 2 (ETCSL co-occurrence): Parallel Sumerian/English text analysis.
  - Lower confidence but broader coverage
  - Uses conditional probability P(english|sumerian)

Output: english_anchors.json with fields: sumerian, english, confidence, source, frequency
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"


def extract_epsd2_anchors(lemmas: list[dict], min_occurrences: int = 5) -> list[dict]:
    """
    Extract Sumerian-English pairs from ORACC lemmatization data.

    Each lemma has:
      - cf: citation form (Sumerian dictionary headword)
      - gw: guide word (English gloss)

    Deduplicates by (cf, gw) and filters by occurrence count.
    """
    # Count (cf, gw) pairs
    pair_counts = Counter()
    for lemma in lemmas:
        cf = lemma.get("cf", "").strip().lower()
        gw = lemma.get("gw", "").strip().lower()
        if cf and gw and gw != "x":
            pair_counts[(cf, gw)] += 1

    anchors = []
    for (cf, gw), count in pair_counts.items():
        if count >= min_occurrences:
            # Confidence based on frequency (more occurrences = higher confidence)
            # Cap at 0.95 for dictionary entries
            confidence = min(0.95, 0.5 + (count / 100))
            anchors.append({
                "sumerian": cf,
                "english": gw,
                "confidence": round(confidence, 4),
                "frequency": count,
                "source": "ePSD2",
            })

    return anchors


def extract_cooccurrence_anchors(
    parallel_lines: list[dict],
    min_cooccurrences: int = 3,
    min_confidence: float = 0.3,
) -> list[dict]:
    """
    Extract anchors from parallel Sumerian-English lines via co-occurrence.

    For each (sumerian_word, english_word) pair, calculates:
      P(english|sumerian) = co-occurrence_count / sumerian_word_total_count
    """
    # Common English stop words to skip
    stop_words = {"the", "of", "and", "in", "to", "a", "is", "was", "for", "on",
                  "with", "his", "her", "its", "their", "he", "she", "it", "they",
                  "that", "this", "by", "from", "at", "an", "be", "has", "had",
                  "not", "but", "who", "which", "as", "or", "if", "my", "your"}

    # Count co-occurrences
    cooc = defaultdict(Counter)  # sumerian_word -> {english_word: count}
    sum_counts = Counter()       # sumerian_word -> total count

    for line in parallel_lines:
        trans = line.get("transliteration", "")
        transl = line.get("translation", "")
        if not trans or not transl:
            continue

        # Tokenize both sides
        sum_words = set(re.findall(r"[a-zA-Z\u0161\u0160\u1E2B\u1E2A\u1E6D\u1E6C\u1E63\u1E62\u011D\u011C]+\d*", trans.lower()))
        eng_words = set(re.findall(r"[a-z]+", transl.lower())) - stop_words

        for sw in sum_words:
            sum_counts[sw] += 1
            for ew in eng_words:
                cooc[sw][ew] += 1

    # Calculate conditional probabilities
    anchors = []
    for sw, eng_counts in cooc.items():
        total = sum_counts[sw]
        if total < min_cooccurrences:
            continue

        # Take the highest-probability English word for this Sumerian word
        best_ew, best_count = eng_counts.most_common(1)[0]
        confidence = best_count / total

        if confidence >= min_confidence:
            anchors.append({
                "sumerian": sw,
                "english": best_ew,
                "confidence": round(confidence, 4),
                "frequency": total,
                "source": "ETCSL",
            })

    return anchors


def merge_anchors(dict_anchors: list[dict], cooc_anchors: list[dict]) -> list[dict]:
    """
    Merge dictionary and co-occurrence anchors.

    For duplicate Sumerian words, keep the entry with higher confidence.
    """
    best = {}

    for anchor in dict_anchors + cooc_anchors:
        key = anchor["sumerian"]
        if key not in best or anchor["confidence"] > best[key]["confidence"]:
            best[key] = anchor

    return sorted(best.values(), key=lambda a: a["confidence"], reverse=True)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_DICTS.mkdir(parents=True, exist_ok=True)

    # Source 1: ePSD2 from ORACC lemmas
    lemma_path = DATA_RAW / "oracc_lemmas.json"
    if lemma_path.exists():
        with open(lemma_path) as f:
            lemmas = json.load(f)
        dict_anchors = extract_epsd2_anchors(lemmas, min_occurrences=5)
        print(f"ePSD2 anchors: {len(dict_anchors)}")

        # Save dictionary entries
        with open(DATA_DICTS / "epsd2_entries.json", "w", encoding="utf-8") as f:
            json.dump(dict_anchors, f, ensure_ascii=False, indent=2)
    else:
        print("No ORACC lemma data found, skipping ePSD2 anchors")
        dict_anchors = []

    # Source 2: ETCSL co-occurrence
    etcsl_path = DATA_RAW / "etcsl_texts.json"
    if etcsl_path.exists():
        with open(etcsl_path) as f:
            etcsl_lines = json.load(f)
        # Filter to lines with translations
        parallel = [l for l in etcsl_lines if l.get("translation")]
        cooc_anchors = extract_cooccurrence_anchors(parallel, min_cooccurrences=3, min_confidence=0.3)
        print(f"ETCSL co-occurrence anchors: {len(cooc_anchors)}")
    else:
        print("No ETCSL data found, skipping co-occurrence anchors")
        cooc_anchors = []

    # Merge
    merged = merge_anchors(dict_anchors, cooc_anchors)

    output_path = DATA_PROCESSED / "english_anchors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nMerged anchors: {len(merged)}")
    print(f"From ePSD2: {sum(1 for a in merged if a['source'] == 'ePSD2')}")
    print(f"From ETCSL: {sum(1 for a in merged if a['source'] == 'ETCSL')}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create import shim**

```python
# scripts/anchors_06.py
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "anchors",
    os.path.join(os.path.dirname(__file__), "06_extract_anchors.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_epsd2_anchors = _mod.extract_epsd2_anchors
extract_cooccurrence_anchors = _mod.extract_cooccurrence_anchors
merge_anchors = _mod.merge_anchors
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_06_anchors.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/06_extract_anchors.py scripts/anchors_06.py tests/test_06_anchors.py
git commit -m "feat: add anchor extraction from ePSD2 dictionary and ETCSL co-occurrence"
```

---

### Task 8: FastText Training

**Files:**
- Create: `scripts/07_train_fasttext.py`
- Create: `tests/test_07_fasttext.py`
- Output: `models/fasttext_sumerian.model`, `models/fasttext_sumerian.vec`

Train 768d FastText skip-gram on the cleaned Sumerian corpus, mirroring heiroglyphy V15 hyperparameters.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_07_fasttext.py
import os
import tempfile
import pytest


def test_corpus_iterator():
    """CorpusIterator should yield lines from a text file."""
    from scripts.fasttext_07 import CorpusIterator

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("lugal e2 gal\n")
        f.write("dingir an ki\n")
        f.write("mu sar ra\n")
        f.flush()

        lines = list(CorpusIterator(f.name))

    os.unlink(f.name)

    assert len(lines) == 3
    assert lines[0] == ["lugal", "e2", "gal"]
    assert lines[1] == ["dingir", "an", "ki"]


def test_train_fasttext_model():
    """Train a small FastText model and verify dimensions."""
    from scripts.fasttext_07 import train_fasttext, CorpusIterator

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        # Need enough data for min_count to work
        for _ in range(100):
            f.write("lugal e2 gal dingir an ki mu sar ra nam\n")
        f.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_fasttext(
                corpus_path=f.name,
                output_dir=tmpdir,
                vector_size=32,  # small for test
                window=5,
                min_count=1,
                epochs=2,
            )

            assert model.vector_size == 32
            assert "lugal" in model.wv

    os.unlink(f.name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_07_fasttext.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write FastText training script**

```python
# scripts/07_train_fasttext.py
"""
FastText Training: Train 768d skip-gram embeddings on cleaned Sumerian corpus.

Hyperparameters (from heiroglyphy V15):
  vector_size: 768
  window: 10
  min_count: 5
  epochs: 10
  sg: 1 (skip-gram)
"""
from pathlib import Path

from gensim.models import FastText

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"


class CorpusIterator:
    """Iterate over lines in a text file, yielding tokenized lists."""

    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def train_fasttext(
    corpus_path: str,
    output_dir: str,
    vector_size: int = 768,
    window: int = 10,
    min_count: int = 5,
    epochs: int = 10,
) -> FastText:
    """Train FastText skip-gram model on corpus."""
    print(f"Training FastText: dim={vector_size}, window={window}, min_count={min_count}, epochs={epochs}")

    corpus = CorpusIterator(corpus_path)

    model = FastText(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        epochs=epochs,
        workers=4,
    )

    # Save model and word vectors
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "fasttext_sumerian.model"
    vec_path = output_dir / "fasttext_sumerian.vec"

    model.save(str(model_path))
    model.wv.save_word2vec_format(str(vec_path))

    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Model saved to: {model_path}")
    print(f"Vectors saved to: {vec_path}")

    return model


def main():
    corpus_path = DATA_PROCESSED / "cleaned_corpus.txt"
    train_fasttext(
        corpus_path=str(corpus_path),
        output_dir=str(MODELS_DIR),
        vector_size=768,
        window=10,
        min_count=5,
        epochs=10,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create import shim**

```python
# scripts/fasttext_07.py
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "fasttext_train",
    os.path.join(os.path.dirname(__file__), "07_train_fasttext.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CorpusIterator = _mod.CorpusIterator
train_fasttext = _mod.train_fasttext
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_07_fasttext.py -v`
Expected: 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/07_train_fasttext.py scripts/fasttext_07.py tests/test_07_fasttext.py
git commit -m "feat: add FastText 768d skip-gram training for Sumerian"
```

---

### Task 9: Embedding Fusion

**Files:**
- Create: `scripts/08_fuse_embeddings.py`
- Create: `tests/test_08_fuse.py`
- Output: `models/fused_embeddings_1536d.npz`

Concatenate FastText 768d text vectors with 768d zero-padding to produce 1536d fused vectors. This replicates heiroglyphy V15's dimensionality regularization trick.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_08_fuse.py
import numpy as np
import pytest


def test_fuse_with_zero_padding():
    """Fusing text vectors with zero padding should produce 1536d vectors."""
    from scripts.fuse_08 import fuse_embeddings

    # Simulate 3 words with 768d text vectors
    vocab = ["lugal", "e2", "dingir"]
    text_vectors = np.random.randn(3, 768).astype(np.float32)

    fused, fused_vocab = fuse_embeddings(vocab, text_vectors)

    assert fused.shape == (3, 1536)
    # First 768 dims should be the text vectors
    np.testing.assert_array_equal(fused[:, :768], text_vectors)
    # Last 768 dims should be zeros
    np.testing.assert_array_equal(fused[:, 768:], np.zeros((3, 768)))
    assert fused_vocab == vocab


def test_fuse_preserves_dtype():
    """Fused vectors should be float32."""
    from scripts.fuse_08 import fuse_embeddings

    vocab = ["lugal"]
    text_vectors = np.random.randn(1, 768).astype(np.float32)

    fused, _ = fuse_embeddings(vocab, text_vectors)
    assert fused.dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_08_fuse.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the fusion script**

```python
# scripts/08_fuse_embeddings.py
"""
Embedding Fusion: Concatenate FastText 768d with 768d zero-padding.

Produces 1536d fused vectors. The zero-padding acts as implicit regularization
during Ridge regression alignment, as discovered in heiroglyphy V15.
"""
import numpy as np
from gensim.models import KeyedVectors
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def fuse_embeddings(
    vocab: list[str],
    text_vectors: np.ndarray,
    pad_dim: int = 768,
) -> tuple[np.ndarray, list[str]]:
    """
    Fuse text embeddings with zero-padding.

    Args:
        vocab: List of words
        text_vectors: (N, text_dim) array of text embeddings
        pad_dim: Dimension of zero-padding (default 768)

    Returns:
        fused: (N, text_dim + pad_dim) fused vectors
        vocab: Same word list (passthrough)
    """
    n, text_dim = text_vectors.shape
    padding = np.zeros((n, pad_dim), dtype=np.float32)
    fused = np.concatenate([text_vectors, padding], axis=1)
    return fused, vocab


def main():
    # Load FastText vectors
    vec_path = MODELS_DIR / "fasttext_sumerian.vec"
    print(f"Loading FastText vectors from {vec_path}")
    kv = KeyedVectors.load_word2vec_format(str(vec_path))

    vocab = list(kv.index_to_key)
    text_vectors = np.array([kv[w] for w in vocab], dtype=np.float32)
    print(f"Loaded {len(vocab)} words, {text_vectors.shape[1]}d")

    # Fuse
    fused, _ = fuse_embeddings(vocab, text_vectors)
    print(f"Fused shape: {fused.shape}")

    # Save
    output_path = MODELS_DIR / "fused_embeddings_1536d.npz"
    np.savez_compressed(
        str(output_path),
        vectors=fused,
        vocab=np.array(vocab),
    )
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create import shim**

```python
# scripts/fuse_08.py
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "fuse",
    os.path.join(os.path.dirname(__file__), "08_fuse_embeddings.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

fuse_embeddings = _mod.fuse_embeddings
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_08_fuse.py -v`
Expected: 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/08_fuse_embeddings.py scripts/fuse_08.py tests/test_08_fuse.py
git commit -m "feat: add embedding fusion with 768d zero-padding for 1536d vectors"
```

---

### Task 10: Ridge Alignment & Evaluation

**Files:**
- Create: `scripts/09_align_and_evaluate.py`
- Create: `tests/test_09_align.py`
- Output: `results/alignment_results.json`

Train Ridge Regression (alpha=0.001) to map 1536d Sumerian space to 300d GloVe English space. Evaluate Top-1/5/10 accuracy on 20% held-out anchors.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_09_align.py
import numpy as np
import pytest


def test_build_training_data():
    """Build X (Sumerian) and Y (English) matrices from anchors."""
    from scripts.align_09 import build_training_data

    anchors = [
        {"sumerian": "lugal", "english": "king"},
        {"sumerian": "e2", "english": "house"},
        {"sumerian": "unknown", "english": "missing"},  # not in vocab
    ]

    sum_vocab = {"lugal": 0, "e2": 1, "dingir": 2}
    sum_vectors = np.random.randn(3, 1536).astype(np.float32)

    eng_vocab = {"king": 0, "house": 1, "water": 2}
    eng_vectors = np.random.randn(3, 300).astype(np.float32)

    X, Y, valid_anchors = build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors
    )

    assert X.shape == (2, 1536)  # 2 valid anchors
    assert Y.shape == (2, 300)
    assert len(valid_anchors) == 2


def test_evaluate_alignment():
    """Evaluate Top-K accuracy of alignment."""
    from scripts.align_09 import evaluate_alignment

    # Create a simple scenario where nearest neighbor is correct
    np.random.seed(42)
    n_test = 10
    dim = 300

    # Make Y_pred close to Y_test so top-1 should be high
    Y_test = np.random.randn(n_test, dim).astype(np.float32)
    Y_pred = Y_test + np.random.randn(n_test, dim).astype(np.float32) * 0.01

    # GloVe vocab includes the test words plus noise
    glove_vocab = [f"word_{i}" for i in range(n_test + 50)]
    glove_vectors = np.vstack([
        Y_test,
        np.random.randn(50, dim).astype(np.float32),
    ])

    test_english = [f"word_{i}" for i in range(n_test)]

    results = evaluate_alignment(Y_pred, test_english, glove_vocab, glove_vectors)

    assert "top1" in results
    assert "top5" in results
    assert "top10" in results
    assert results["top1"] > 0.5  # Should be very high with low noise


def test_train_ridge():
    """Train Ridge regression and verify it produces correct output shape."""
    from scripts.align_09 import train_ridge

    X = np.random.randn(100, 1536).astype(np.float32)
    Y = np.random.randn(100, 300).astype(np.float32)

    model = train_ridge(X, Y, alpha=0.001)

    # Predict should produce (N, 300)
    Y_pred = model.predict(X[:5])
    assert Y_pred.shape == (5, 300)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_09_align.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the alignment and evaluation script**

```python
# scripts/09_align_and_evaluate.py
"""
Ridge Alignment & Evaluation: Map Sumerian embeddings to GloVe English space.

Pipeline:
  1. Load fused 1536d Sumerian vectors
  2. Load GloVe 300d English vectors
  3. Load anchor pairs
  4. Build training data (only anchors present in both vocabs)
  5. 80/20 train/test split (random_state=42)
  6. Train Ridge regression (alpha=0.001)
  7. Evaluate Top-1/5/10 accuracy on test set
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def build_training_data(
    anchors: list[dict],
    sum_vocab: dict[str, int],
    sum_vectors: np.ndarray,
    eng_vocab: dict[str, int],
    eng_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Build aligned X (Sumerian) and Y (English) matrices from anchor pairs.

    Only includes anchors where both the Sumerian and English words
    are present in their respective vocabularies.
    """
    X_list = []
    Y_list = []
    valid = []

    for anchor in anchors:
        s_word = anchor["sumerian"]
        e_word = anchor["english"]

        if s_word in sum_vocab and e_word in eng_vocab:
            X_list.append(sum_vectors[sum_vocab[s_word]])
            Y_list.append(eng_vectors[eng_vocab[e_word]])
            valid.append(anchor)

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list), np.array(Y_list), valid


def train_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 0.001) -> Ridge:
    """Train Ridge regression to map X -> Y."""
    model = Ridge(alpha=alpha)
    model.fit(X, Y)
    return model


def evaluate_alignment(
    Y_pred: np.ndarray,
    test_english: list[str],
    glove_vocab: list[str],
    glove_vectors: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """
    Evaluate alignment accuracy using Top-K nearest neighbor retrieval.

    For each test sample, finds K nearest neighbors in GloVe space
    and checks if the correct English word is among them.
    """
    # Normalize predicted vectors
    norms = np.linalg.norm(Y_pred, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Y_pred_norm = Y_pred / norms

    # Normalize GloVe vectors
    g_norms = np.linalg.norm(glove_vectors, axis=1, keepdims=True)
    g_norms[g_norms == 0] = 1
    glove_norm = glove_vectors / g_norms

    # Compute cosine distances
    distances = cdist(Y_pred_norm, glove_norm, metric="cosine")

    # Build vocab index for fast lookup
    vocab_set = {w: i for i, w in enumerate(glove_vocab)}

    results = {}
    for k in ks:
        correct = 0
        for i, eng_word in enumerate(test_english):
            if eng_word not in vocab_set:
                continue
            # Get K nearest neighbors
            nn_indices = np.argsort(distances[i])[:k]
            nn_words = [glove_vocab[j] for j in nn_indices]
            if eng_word in nn_words:
                correct += 1
        total = len(test_english)
        results[f"top{k}"] = (correct / total * 100) if total > 0 else 0.0

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load fused Sumerian vectors
    fused_path = MODELS_DIR / "fused_embeddings_1536d.npz"
    print(f"Loading fused vectors from {fused_path}")
    fused_data = np.load(str(fused_path), allow_pickle=True)
    sum_vectors = fused_data["vectors"]
    sum_vocab_list = list(fused_data["vocab"])
    sum_vocab = {w: i for i, w in enumerate(sum_vocab_list)}
    print(f"Sumerian vocab: {len(sum_vocab)} words, {sum_vectors.shape[1]}d")

    # Load GloVe
    glove_path = DATA_PROCESSED / "glove.6B.300d.txt"
    print(f"Loading GloVe from {glove_path}")
    glove_vocab = []
    glove_vectors_list = []
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            glove_vocab.append(word)
            glove_vectors_list.append(vec)
    glove_vectors = np.array(glove_vectors_list)
    eng_vocab = {w: i for i, w in enumerate(glove_vocab)}
    print(f"GloVe vocab: {len(glove_vocab)} words, {glove_vectors.shape[1]}d")

    # Load anchors
    anchor_path = DATA_PROCESSED / "english_anchors.json"
    with open(anchor_path) as f:
        anchors = json.load(f)
    print(f"Loaded {len(anchors)} anchors")

    # Build training data
    X, Y, valid_anchors = build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, glove_vectors
    )
    print(f"Valid anchors: {len(valid_anchors)} / {len(anchors)} ({len(valid_anchors)/len(anchors)*100:.1f}%)")

    # Train/test split
    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X, Y, valid_anchors, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train Ridge
    print("Training Ridge regression (alpha=0.001)...")
    model = train_ridge(X_train, Y_train, alpha=0.001)

    # Predict on test set
    Y_pred = model.predict(X_test)

    # Evaluate
    test_english = [a["english"] for a in anchors_test]
    results = evaluate_alignment(Y_pred, test_english, glove_vocab, glove_vectors)

    print(f"\n=== RESULTS ===")
    print(f"Top-1 Accuracy:  {results['top1']:.2f}%")
    print(f"Top-5 Accuracy:  {results['top5']:.2f}%")
    print(f"Top-10 Accuracy: {results['top10']:.2f}%")

    # Save results
    full_results = {
        "accuracy": results,
        "config": {
            "alignment": "Ridge",
            "alpha": 0.001,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "valid_anchors": len(valid_anchors),
            "total_anchors": len(anchors),
            "sumerian_vocab": len(sum_vocab),
            "fused_dim": sum_vectors.shape[1],
            "glove_dim": glove_vectors.shape[1],
        },
    }

    results_path = RESULTS_DIR / "alignment_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save Ridge model weights for production use
    np.savez_compressed(
        str(MODELS_DIR / "ridge_weights.npz"),
        coef=model.coef_,
        intercept=model.intercept_,
    )
    print("Ridge weights saved")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create import shim**

```python
# scripts/align_09.py
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "align",
    os.path.join(os.path.dirname(__file__), "09_align_and_evaluate.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_training_data = _mod.build_training_data
train_ridge = _mod.train_ridge
evaluate_alignment = _mod.evaluate_alignment
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_09_align.py -v`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/09_align_and_evaluate.py scripts/align_09.py tests/test_09_align.py
git commit -m "feat: add Ridge regression alignment with Top-K evaluation"
```

---

### Task 11: Production Export & Lookup API

**Files:**
- Create: `scripts/10_export_production.py`
- Create: `final_output/sumerian_lookup.py`
- Create: `tests/test_10_export.py`
- Output: `final_output/sumerian_aligned_vectors.npz`, `final_output/sumerian_aligned_vocab.pkl`, `final_output/metadata.json`

Project all Sumerian vocab into GloVe space and package with a lookup API mirroring heiroglyphy's `egyptian_lookup.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_10_export.py
import numpy as np
import pickle
import tempfile
import os
import json
import pytest


def test_project_all_vectors():
    """Project all Sumerian vectors into GloVe space using Ridge weights."""
    from scripts.export_10 import project_all_vectors

    sum_vectors = np.random.randn(100, 1536).astype(np.float32)
    coef = np.random.randn(300, 1536).astype(np.float32)
    intercept = np.random.randn(300).astype(np.float32)

    projected = project_all_vectors(sum_vectors, coef, intercept)

    assert projected.shape == (100, 300)
    assert projected.dtype == np.float16  # stored as float16 for size


def test_sumerian_lookup_find():
    """SumerianLookup.find should return nearest Sumerian words for English query."""
    from final_output.sumerian_lookup import SumerianLookup

    # Create tiny test data
    np.random.seed(42)
    n_sum = 5
    n_eng = 10
    dim = 300

    sum_vectors = np.random.randn(n_sum, dim).astype(np.float16)
    sum_vocab = ["lugal", "e2", "dingir", "an", "ki"]
    eng_vectors = np.random.randn(n_eng, dim).astype(np.float32)
    eng_vocab = [f"word_{i}" for i in range(n_eng)]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save test data
        np.savez_compressed(
            os.path.join(tmpdir, "sumerian_aligned_vectors.npz"),
            vectors=sum_vectors,
        )
        with open(os.path.join(tmpdir, "sumerian_aligned_vocab.pkl"), "wb") as f:
            pickle.dump(sum_vocab, f)

        lookup = SumerianLookup(
            vectors_path=os.path.join(tmpdir, "sumerian_aligned_vectors.npz"),
            vocab_path=os.path.join(tmpdir, "sumerian_aligned_vocab.pkl"),
            glove_vectors=eng_vectors,
            glove_vocab=eng_vocab,
        )

        results = lookup.find("word_0", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Each result is (sumerian_word, similarity_score)
        assert results[0][0] in sum_vocab
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_10_export.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the export script**

```python
# scripts/10_export_production.py
"""
Production Export: Project all Sumerian vectors into GloVe space and package.

Outputs:
  - sumerian_aligned_vectors.npz (float16 for size)
  - sumerian_aligned_vocab.pkl
  - metadata.json
"""
import json
import pickle
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FINAL_OUTPUT = Path(__file__).parent.parent / "final_output"


def project_all_vectors(
    sum_vectors: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    """Project all Sumerian vectors into GloVe space using learned Ridge weights."""
    # coef shape: (300, 1536), intercept shape: (300,)
    projected = sum_vectors @ coef.T + intercept
    return projected.astype(np.float16)


def main():
    FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load fused Sumerian vectors
    fused_data = np.load(str(MODELS_DIR / "fused_embeddings_1536d.npz"), allow_pickle=True)
    sum_vectors = fused_data["vectors"]
    sum_vocab = list(fused_data["vocab"])
    print(f"Sumerian vectors: {sum_vectors.shape}")

    # Load Ridge weights
    ridge_data = np.load(str(MODELS_DIR / "ridge_weights.npz"))
    coef = ridge_data["coef"]
    intercept = ridge_data["intercept"]
    print(f"Ridge coef: {coef.shape}")

    # Project all vectors
    aligned = project_all_vectors(sum_vectors, coef, intercept)
    print(f"Aligned vectors: {aligned.shape}, dtype: {aligned.dtype}")

    # Save
    np.savez_compressed(
        str(FINAL_OUTPUT / "sumerian_aligned_vectors.npz"),
        vectors=aligned,
    )
    with open(FINAL_OUTPUT / "sumerian_aligned_vocab.pkl", "wb") as f:
        pickle.dump(sum_vocab, f)

    # Load results for metadata
    results_path = RESULTS_DIR / "alignment_results.json"
    with open(results_path) as f:
        results = json.load(f)

    metadata = {
        "methodology": "Cuneiformy SOTA (1536d fused -> 300d GloVe)",
        "text_embeddings": "768d FastText (min_count=5, window=10)",
        "visual_embeddings": "768d zero-padding (regularization)",
        "alignment": "Ridge Regression (alpha=0.001)",
        "accuracy": results["accuracy"],
        "vocab_size": len(sum_vocab),
        "vector_dim": 300,
        "config": results["config"],
    }

    with open(FINAL_OUTPUT / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProduction files saved to {FINAL_OUTPUT}/")
    print(f"  sumerian_aligned_vectors.npz ({aligned.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  sumerian_aligned_vocab.pkl")
    print(f"  metadata.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the lookup API**

```python
# final_output/__init__.py
```

```python
# final_output/sumerian_lookup.py
"""
Sumerian Semantic Lookup: Find Sumerian words by English meaning.

Usage:
    lookup = SumerianLookup(
        vectors_path="sumerian_aligned_vectors.npz",
        vocab_path="sumerian_aligned_vocab.pkl",
        glove_vectors=glove_vectors,
        glove_vocab=glove_vocab,
    )

    # Find Sumerian words for an English concept
    lookup.find("king")  # -> [("lugal", 0.72), ("enszi2", 0.45), ...]

    # Vector analogy: king is to queen as god is to ?
    lookup.find_analogy("king", "queen", "god")

    # Weighted blend of concepts
    lookup.find_blend({"water": 0.7, "god": 0.3})
"""
import pickle
import numpy as np
from scipy.spatial.distance import cdist


class SumerianLookup:
    def __init__(
        self,
        vectors_path: str,
        vocab_path: str,
        glove_vectors: np.ndarray,
        glove_vocab: list[str],
    ):
        # Load Sumerian vectors
        data = np.load(vectors_path, allow_pickle=True)
        self.vectors = data["vectors"].astype(np.float32)
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        # Normalize Sumerian vectors
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors_norm = self.vectors / norms

        # GloVe
        self.glove_vectors = glove_vectors
        self.glove_vocab = glove_vocab
        self.glove_word_to_idx = {w: i for i, w in enumerate(glove_vocab)}

        # Normalize GloVe
        g_norms = np.linalg.norm(self.glove_vectors, axis=1, keepdims=True)
        g_norms[g_norms == 0] = 1
        self.glove_norm = self.glove_vectors / g_norms

    def _get_english_vector(self, word: str) -> np.ndarray | None:
        """Get normalized GloVe vector for an English word."""
        idx = self.glove_word_to_idx.get(word.lower())
        if idx is None:
            return None
        return self.glove_norm[idx]

    def find(self, english_word: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find Sumerian words most similar to an English concept."""
        vec = self._get_english_vector(english_word)
        if vec is None:
            return []

        # Cosine similarity against all Sumerian vectors
        sims = self.vectors_norm @ vec
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [(self.vocab[i], float(sims[i])) for i in top_indices]

    def find_analogy(
        self, a: str, b: str, c: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Vector analogy: a is to b as c is to ? Returns Sumerian words."""
        va = self._get_english_vector(a)
        vb = self._get_english_vector(b)
        vc = self._get_english_vector(c)
        if any(v is None for v in [va, vb, vc]):
            return []

        # target = c - a + b
        target = vc - va + vb
        target = target / (np.linalg.norm(target) + 1e-10)

        sims = self.vectors_norm @ target
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [(self.vocab[i], float(sims[i])) for i in top_indices]

    def find_blend(
        self, weights: dict[str, float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find Sumerian words matching a weighted blend of English concepts."""
        target = np.zeros(self.vectors.shape[1], dtype=np.float32)
        for word, weight in weights.items():
            vec = self._get_english_vector(word)
            if vec is not None:
                target += weight * vec

        norm = np.linalg.norm(target)
        if norm == 0:
            return []
        target = target / norm

        sims = self.vectors_norm @ target
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [(self.vocab[i], float(sims[i])) for i in top_indices]
```

- [ ] **Step 5: Create import shim for export script**

```python
# scripts/export_10.py
from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "export",
    os.path.join(os.path.dirname(__file__), "10_export_production.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

project_all_vectors = _mod.project_all_vectors
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_10_export.py -v`
Expected: 2 tests PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/10_export_production.py scripts/export_10.py final_output/__init__.py final_output/sumerian_lookup.py tests/test_10_export.py
git commit -m "feat: add production export and Sumerian semantic lookup API"
```

---

### Task 12: GloVe Download Helper

**Files:**
- Create: `scripts/download_glove.py`

The GloVe vectors (glove.6B.300d.txt) are needed for alignment and the lookup API. This is a standalone helper since the file is large (~1GB) and shared with heiroglyphy.

- [ ] **Step 1: Write the download helper**

```python
# scripts/download_glove.py
"""
Download GloVe 6B 300d pre-trained English vectors.

Source: https://nlp.stanford.edu/data/glove.6B.zip
Extracts glove.6B.300d.txt to data/processed/
"""
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
HEIROGLYPHY_GLOVE = Path(__file__).parent.parent.parent / "heiroglyphy" / "heiro_v5_getdata" / "data" / "processed" / "glove.6B.300d.txt"


def download_glove():
    """Download GloVe or symlink from heiroglyphy if available."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "glove.6B.300d.txt"

    if output_path.exists():
        print(f"GloVe already present: {output_path}")
        return output_path

    # Check if heiroglyphy has it
    if HEIROGLYPHY_GLOVE.exists():
        print(f"Symlinking GloVe from heiroglyphy: {HEIROGLYPHY_GLOVE}")
        os.symlink(HEIROGLYPHY_GLOVE, output_path)
        return output_path

    # Download
    print("Downloading GloVe 6B (862 MB)...")
    zip_path = DATA_PROCESSED / "glove.6B.zip"

    response = requests.get(GLOVE_URL, stream=True, timeout=300)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Extract only the 300d file
    print("Extracting glove.6B.300d.txt...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("glove.6B.300d.txt", str(DATA_PROCESSED))

    # Clean up zip
    zip_path.unlink()
    print(f"GloVe saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    download_glove()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/download_glove.py
git commit -m "feat: add GloVe download helper with heiroglyphy symlink support"
```

---

### Task 13: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

A smoke test that runs the full pipeline on synthetic data to verify all scripts chain together correctly.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration.py
"""
End-to-end integration test using synthetic data.
Verifies the full pipeline: corpus -> embeddings -> fusion -> alignment -> lookup.
"""
import json
import numpy as np
import os
import pickle
import tempfile
import pytest


def test_full_pipeline_synthetic():
    """Run the full pipeline on tiny synthetic data."""
    from scripts.clean_05 import clean_atf_line, build_corpus
    from scripts.anchors_06 import extract_epsd2_anchors, merge_anchors
    from scripts.fasttext_07 import train_fasttext
    from scripts.fuse_08 import fuse_embeddings
    from scripts.align_09 import build_training_data, train_ridge, evaluate_alignment
    from scripts.export_10 import project_all_vectors

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create synthetic corpus
        corpus_lines = []
        words = ["lugal", "e2", "dingir", "an", "ki", "mu", "nam", "en", "gal", "nin",
                 "sar", "kur", "igi", "sag", "du", "gin", "ba", "bi", "na", "zu"]
        np.random.seed(42)
        for _ in range(500):
            n = np.random.randint(3, 10)
            line = " ".join(np.random.choice(words, n))
            corpus_lines.append(line)

        corpus_path = os.path.join(tmpdir, "corpus.txt")
        with open(corpus_path, "w") as f:
            for line in corpus_lines:
                f.write(line + "\n")

        # 2. Train FastText (small)
        model = train_fasttext(
            corpus_path=corpus_path,
            output_dir=tmpdir,
            vector_size=32,
            window=5,
            min_count=1,
            epochs=5,
        )
        assert len(model.wv) >= 15

        # 3. Fuse with zero padding
        vocab = list(model.wv.index_to_key)
        text_vecs = np.array([model.wv[w] for w in vocab], dtype=np.float32)
        fused, fused_vocab = fuse_embeddings(vocab, text_vecs, pad_dim=32)
        assert fused.shape[1] == 64  # 32 + 32

        # 4. Create synthetic GloVe (tiny)
        glove_words = ["king", "house", "god", "heaven", "earth", "name", "fate",
                       "lord", "great", "queen", "write", "mountain", "eye", "head",
                       "go", "walk", "give", "this", "that", "know"]
        glove_vecs = np.random.randn(len(glove_words), 16).astype(np.float32)
        eng_vocab = {w: i for i, w in enumerate(glove_words)}

        # 5. Create anchors
        anchors = [
            {"sumerian": "lugal", "english": "king", "confidence": 0.95, "source": "ePSD2"},
            {"sumerian": "e2", "english": "house", "confidence": 0.90, "source": "ePSD2"},
            {"sumerian": "dingir", "english": "god", "confidence": 0.85, "source": "ePSD2"},
            {"sumerian": "an", "english": "heaven", "confidence": 0.80, "source": "ePSD2"},
            {"sumerian": "ki", "english": "earth", "confidence": 0.80, "source": "ePSD2"},
            {"sumerian": "mu", "english": "name", "confidence": 0.75, "source": "ePSD2"},
            {"sumerian": "nam", "english": "fate", "confidence": 0.70, "source": "ePSD2"},
            {"sumerian": "en", "english": "lord", "confidence": 0.70, "source": "ePSD2"},
            {"sumerian": "gal", "english": "great", "confidence": 0.65, "source": "ePSD2"},
            {"sumerian": "nin", "english": "queen", "confidence": 0.60, "source": "ePSD2"},
        ]

        sum_vocab_dict = {w: i for i, w in enumerate(fused_vocab)}

        # 6. Build training data
        X, Y, valid = build_training_data(anchors, sum_vocab_dict, fused, eng_vocab, glove_vecs)
        assert len(valid) == 10

        # 7. Train Ridge
        ridge = train_ridge(X, Y, alpha=0.001)
        Y_pred = ridge.predict(X)
        assert Y_pred.shape == (10, 16)

        # 8. Project all vectors
        projected = project_all_vectors(fused, ridge.coef_, ridge.intercept_)
        assert projected.shape == (len(fused_vocab), 16)
        assert projected.dtype == np.float16

        # Pipeline completes without error
        print("Integration test passed!")
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test with synthetic data"
```

---

### Task 14: .gitignore and Final Cleanup

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Write .gitignore**

```
# Data (too large for git)
data/raw/
data/processed/glove.6B.*
data/processed/cleaned_corpus.txt
data/processed/merged_corpus.json
data/processed/english_anchors.json
data/dictionaries/

# Models (too large for git)
models/

# Results (regenerable)
results/

# Production vectors (too large for git)
final_output/*.npz
final_output/*.pkl

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/
dist/
build/
*.egg

# Virtual environment
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore for data, models, and build artifacts"
```
