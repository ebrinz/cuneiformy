"""
ETCSL passage retrieval for per-concept source-text grounding.

Given a Sumerian token (normalized), find 1-N passages in
data/raw/etcsl_texts.json where the token appears, returning the matched
line plus surrounding context lines.

Used in §7 of each deep dive to connect geometric claims to textual reality.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations

from scripts.sumerian_normalize import normalize_sumerian_token


def find_passages(
    sumerian_token: str,
    etcsl_texts: list[dict],
    max_passages: int = 3,
    context_lines: int = 2,
) -> list[dict]:
    """Find ETCSL passages containing the token (normalized match).

    Both the query and each transliteration word are run through
    normalize_sumerian_token before matching, so callers can pass either the
    ASCII-normalized form (e.g. 'namtar') or the conventional Sumerological
    form (e.g. 'nam-tar') and get the same results.

    Returns a list of {text_id, title, matched_line_no, transliteration,
    translation, context} dicts. `context` is [context_lines before] +
    matched line + [context_lines after].

    Supports two ETCSL JSON schemas:
      - Nested: [{text_id, title, lines: [{line_no, transliteration, ...}]}, ...]
      - Flat:   [{transliteration, translation, line_id, source}, ...]
        (e.g. data/raw/etcsl_texts.json actual format)
    """
    query_normalized = normalize_sumerian_token(sumerian_token)
    results = []

    # Detect schema from first entry.
    if etcsl_texts and "lines" in etcsl_texts[0]:
        # --- Nested schema ---
        for text in etcsl_texts:
            lines = text.get("lines", [])
            for i, line in enumerate(lines):
                trans = line.get("transliteration", "") or ""
                normalized_tokens = {normalize_sumerian_token(t) for t in trans.split()}
                if query_normalized in normalized_tokens:
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
    else:
        # --- Flat schema ---
        # Each dict is a single line: {transliteration, translation, line_id, source}
        # context_lines are index-adjacent entries in the flat list.
        for i, entry in enumerate(etcsl_texts):
            trans = entry.get("transliteration", "") or ""
            normalized_tokens = {normalize_sumerian_token(t) for t in trans.split()}
            if query_normalized in normalized_tokens:
                start = max(0, i - context_lines)
                end = min(len(etcsl_texts), i + context_lines + 1)
                results.append({
                    "text_id": entry.get("line_id", ""),
                    "title": entry.get("source", ""),
                    "matched_line_no": entry.get("line_id", i),
                    "transliteration": trans,
                    "translation": entry.get("translation", ""),
                    "context": etcsl_texts[start:end],
                })
                if len(results) >= max_passages:
                    return results

    return results
