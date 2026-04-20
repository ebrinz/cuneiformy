"""
Canonical Sumerian cosmogony concept slate for the document.

Each concept pairs a Sumerian token with its English seed and thematic tag.
Pre-flight check in preflight_concept_check.py validates each against the
current SumerianLookup + ETCSL corpus, flagging substitutions if needed.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""

# Restoration record (preflight 2026-04-18):
#   "namtar" is restored to the primary slate. The previous substitution with
#   "kur" was caused by a matching bug: raw ETCSL transliterations use "nam-tar"
#   (hyphenated), while the pipeline produces "namtar" (joined). The match loop
#   was comparing directly against split tokens without normalization, so zero
#   ETCSL passages were found. The fix applies normalize_sumerian_token to both
#   query and transliteration tokens in etcsl_passage_finder.py and
#   preflight_concept_check.py, giving "namtar" 35+ confirmed ETCSL passages.
#   "kur" moves to ALTERNATE_CONCEPTS where it remains available.
PRIMARY_CONCEPTS = [
    {"sumerian": "abzu",    "english": "deep",     "theme": "primordial"},
    {"sumerian": "zi",      "english": "breath",   "theme": "animation"},
    {"sumerian": "nam",     "english": "essence",  "theme": "naming"},
    {"sumerian": "namtar",  "english": "fate",     "theme": "decree"},
    {"sumerian": "me",      "english": "decree",   "theme": "civilization"},
]

# Substitutes drawn on if a primary concept fails pre-flight.
ALTERNATE_CONCEPTS = [
    {"sumerian": "ima",     "english": "clay",     "theme": "matter"},
    {"sumerian": "kur",     "english": "mountain", "theme": "netherworld"},
    {"sumerian": "an",      "english": "heaven",   "theme": "primordial"},
    {"sumerian": "ki",      "english": "earth",    "theme": "primordial"},
]

# Anunnaki-adjacent vocabulary for the narrative-spine UMAP figure (§3).
ANUNNAKI_VOCABULARY = [
    "an", "ki", "enki", "enlil", "nammu", "ninmah", "inanna", "utu", "nanna",
    "nam", "me", "namtar", "zi", "abzu", "ima", "kur", "dingir", "lugal",
]

# Tokens used to define the "cosmogonic axis" in the §9 synthesis figure.
COSMOGONIC_POLES = {
    "primordial_pole": ["abzu", "nammu"],     # pre-creation
    "decree_pole": ["me", "namtar", "dingir"],  # civilizational order
}
