"""
ETCSL Scraper: Download and parse the Electronic Text Corpus of Sumerian Literature.

Source: Oxford Text Archive (OTA) - https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2518
Format: TEI P4 XML with transliterations linked to translations via id/corresp attributes.
"""
import json
import os
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from tqdm import tqdm

ETCSL_OTA_URL = "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2518/etcsl.zip"

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"


def download_etcsl(output_dir: Path) -> Path:
    """Download ETCSL ZIP from Oxford Text Archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "etcsl.zip"

    if zip_path.exists():
        print(f"ETCSL ZIP already downloaded: {zip_path}")
        return zip_path

    print("Downloading ETCSL from OTA...")
    response = requests.get(ETCSL_OTA_URL, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return zip_path


def load_etcsl_files(zip_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load transliteration and translation XML files from ETCSL ZIP, keyed by composition ID."""
    transliterations = {}
    translations = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml"):
                continue
            content = zf.read(name).decode("utf-8", errors="replace")
            if "/transliterations/" in name:
                # e.g., etcsl/transliterations/c.2.5.5.4.xml -> c.2.5.5.4
                comp_id = Path(name).stem
                transliterations[comp_id] = content
            elif "/translations/" in name:
                # e.g., etcsl/translations/t.4.08.32.xml -> t.4.08.32
                comp_id = Path(name).stem
                translations[comp_id] = content
    return transliterations, translations


def _comp_id_to_translation_key(comp_id: str) -> str:
    """Convert transliteration comp_id to translation comp_id. c.X.X.X -> t.X.X.X"""
    return "t." + comp_id[2:] if comp_id.startswith("c.") else comp_id


def parse_etcsl_xml(xml_content: str) -> list[dict]:
    """
    Parse a single ETCSL transliteration XML file.

    Real ETCSL format:
    - Transliteration files have <l> tags containing <w form="word"> tags
    - The transliteration text is extracted from w/@form attributes
    - Line IDs like "c2554.A.1" with corresp like "t2554.p1"
    """
    xml_content = xml_content.replace(' xmlns="', ' xmlns:ignore="')
    # Replace undefined HTML entities (e.g., &c; &aacute; &commat;) that aren't valid XML
    xml_content = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)([a-zA-Z0-9]+);", r"_\1_", xml_content)

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return []

    texts = []
    for l_tag in root.iter("l"):
        line_id = l_tag.get("id")
        if not line_id:
            continue

        # Extract word forms from <w> tags
        w_tags = l_tag.findall(".//w")
        if w_tags:
            forms = []
            for w in w_tags:
                form = w.get("form", "")
                if form and form != "X" and form != "&X;":
                    forms.append(form)
            transliteration = " ".join(forms)
        else:
            # Fallback: plain text content
            transliteration = "".join(l_tag.itertext()).strip()

        if not transliteration:
            continue

        corresp = l_tag.get("corresp")

        texts.append({
            "transliteration": transliteration,
            "translation": None,  # filled in by match_translations
            "line_id": line_id,
            "corresp": corresp,
            "source": "ETCSL",
        })

    return texts


def parse_translation_xml(xml_content: str) -> dict[str, str]:
    """
    Parse a translation XML file. Returns {paragraph_id: translation_text}.

    Translation <p> tags have id like "t40832.p1" and corresp like "c40832.1"
    pointing to the first line of the transliteration range.
    """
    xml_content = xml_content.replace(' xmlns="', ' xmlns:ignore="')
    xml_content = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)([a-zA-Z0-9]+);", r"_\1_", xml_content)

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return {}

    translations = {}
    for p_tag in root.iter("p"):
        p_id = p_tag.get("id")
        corresp = p_tag.get("corresp")
        if p_id and corresp:
            text = "".join(p_tag.itertext()).strip()
            if text:
                # Key by the corresp which points to transliteration line
                translations[corresp] = text
                # Also key by the paragraph id
                translations[p_id] = text
    return translations


def match_translations(lines: list[dict], translations: dict[str, str]) -> None:
    """Match translation paragraphs to transliteration lines."""
    for line in lines:
        corresp = line.get("corresp")
        line_id = line.get("line_id", "")

        # Try direct corresp match (line's corresp -> translation paragraph id)
        if corresp and corresp in translations:
            line["translation"] = translations[corresp]
            continue

        # Try matching line_id as a translation corresp target
        # Translation corresp points to line IDs like "c40832.1"
        if line_id in translations:
            line["translation"] = translations[line_id]


def save_texts(texts: list[dict], output_path: str) -> None:
    """Save parsed texts to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(texts)} lines to {output_path}")


def main():
    zip_path = download_etcsl(DATA_RAW)
    transliteration_files, translation_files = load_etcsl_files(zip_path)
    print(f"Found {len(transliteration_files)} transliteration files, {len(translation_files)} translation files")

    all_texts = []
    for comp_id, xml_content in tqdm(transliteration_files.items(), desc="Parsing transliterations"):
        lines = parse_etcsl_xml(xml_content)

        # Find matching translation file
        trans_key = _comp_id_to_translation_key(comp_id)
        if trans_key in translation_files:
            translations = parse_translation_xml(translation_files[trans_key])
            match_translations(lines, translations)

        all_texts.extend(lines)

    # Clean up: remove corresp field from output
    for t in all_texts:
        t.pop("corresp", None)

    output_path = DATA_RAW / "etcsl_texts.json"
    save_texts(all_texts, str(output_path))

    with_translation = sum(1 for t in all_texts if t["translation"])
    print(f"Total lines: {len(all_texts)}")
    print(f"With translations: {with_translation}")
    print(f"Without translations: {len(all_texts) - with_translation}")


if __name__ == "__main__":
    main()
