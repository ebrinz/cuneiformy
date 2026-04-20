"""
Civilization-agnostic anomaly lenses for aligned-embedding analysis.

Each lens takes numpy arrays + lookup maps + threshold parameters, and
returns a ranked list of anomaly rows. No I/O, no hardcoded paths, no
civilization-specific constants.

Used by scripts/analysis/anomaly_framework.py to build per-civilization atlases.

See: docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md
"""
from __future__ import annotations

# Lens implementations land in later tasks.
