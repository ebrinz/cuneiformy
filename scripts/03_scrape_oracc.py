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

            p_number = Path(name).stem

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

    save_texts(all_texts, str(DATA_RAW / "oracc_texts.json"))
    save_texts(all_lemmas, str(DATA_RAW / "oracc_lemmas.json"))

    total_lines = sum(len(t["lines"]) for t in all_texts)
    unique_glosses = len({l["gw"] for l in all_lemmas if l["gw"]})
    print(f"\nTotal texts: {len(all_texts)}")
    print(f"Total lines: {total_lines}")
    print(f"Total lemmas: {len(all_lemmas)}")
    print(f"Unique English glosses: {unique_glosses}")


if __name__ == "__main__":
    main()
