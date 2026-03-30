"""
ETCSL Scraper: Download and parse the Electronic Text Corpus of Sumerian Literature.

Source: Oxford Text Archive (OTA) - https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2518
Format: TEI P4 XML with transliterations linked to translations via id/corresp attributes.
"""
import json
import os
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
    xml_content = xml_content.replace(' xmlns="', ' xmlns:ignore="')

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return []

    translations = {}
    for p_tag in root.iter("p"):
        corresp = p_tag.get("corresp")
        if corresp:
            text = "".join(p_tag.itertext()).strip()
            if text:
                translations[corresp] = text

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

    with_translation = sum(1 for t in all_texts if t["translation"])
    print(f"Total lines: {len(all_texts)}")
    print(f"With translations: {with_translation}")
    print(f"Without translations: {len(all_texts) - with_translation}")


if __name__ == "__main__":
    main()
