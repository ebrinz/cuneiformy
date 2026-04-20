"""
ETCSL passage retrieval for per-concept source-text grounding.

Given a Sumerian token (normalized), find 1-N passages in
data/raw/etcsl_texts.json where the token appears, returning the matched
line plus surrounding context lines.

Used in §7 of each deep dive to connect geometric claims to textual reality.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations


def find_passages(
    sumerian_token: str,
    etcsl_texts: list[dict],
    max_passages: int = 3,
    context_lines: int = 2,
) -> list[dict]:
    """Find ETCSL passages containing the token.

    Returns a list of {text_id, title, matched_line_no, transliteration,
    translation, context} dicts. `context` is [context_lines before] +
    matched line + [context_lines after].
    """
    results = []
    for text in etcsl_texts:
        lines = text.get("lines", [])
        for i, line in enumerate(lines):
            trans = line.get("transliteration", "") or ""
            # Whitespace-split to match whole tokens (avoids false positives
            # inside compound words the way substring-search would).
            if sumerian_token in trans.split():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                results.append({
                    "text_id": text.get("text_id"),
                    "title": text.get("title", ""),
                    "matched_line_no": line.get("line_no", i),
                    "transliteration": trans,
                    "translation": line.get("translation", ""),
                    "context": lines[start:end],
                })
                if len(results) >= max_passages:
                    return results
                break  # one match per text
    return results
