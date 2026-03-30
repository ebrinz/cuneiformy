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
