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

    print("Extracting glove.6B.300d.txt...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("glove.6B.300d.txt", str(DATA_PROCESSED))

    zip_path.unlink()
    print(f"GloVe saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    download_glove()
