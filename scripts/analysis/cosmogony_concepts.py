"""
Canonical Sumerian cosmogony concept slate for the document.

Each concept pairs a Sumerian token with its English seed and thematic tag.
Pre-flight check in preflight_concept_check.py validates each against the
current SumerianLookup + ETCSL corpus, flagging substitutions if needed.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""

# Substitution record (preflight 2026-04-20):
#   "namtar" failed zero_etcsl_passages: the Workstream 2b normalization chain
#   drops hyphens, producing "namtar" in the fused vocab; but the raw ETCSL
#   transliterations retain the conventional "nam-tar" form (35 whole-token
#   occurrences as "nam-tar", 0 as "namtar"). ETCSL grounding is therefore
#   impossible for this token without a separate hyphen-aware lookup.
#   Substituted with "kur" (netherworld/mountain, theme: netherworld) — closest
#   thematic match to namtar's netherworld role, 502 ETCSL passages confirmed.
#   Flag for §12 appendix: "namtar" normalization gap noted; see preflight JSON.
PRIMARY_CONCEPTS = [
    {"sumerian": "abzu",    "english": "deep",     "theme": "primordial"},
    {"sumerian": "zi",      "english": "breath",   "theme": "animation"},
    {"sumerian": "nam",     "english": "essence",  "theme": "naming"},
    {"sumerian": "kur",     "english": "mountain", "theme": "netherworld"},
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
