"""
Maps raw entity surface forms extracted by the ML models to canonical IDs.
Uses alias lookup + fuzzy substring matching.
"""

from __future__ import annotations
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.entity_aliases import ALIASES, ENTITY_TYPES, ENTITY_COLORS


def normalize(surface: str) -> str | None:
    """
    Return canonical ID for a surface form, or None if unrecognized.
    1. Exact alias lookup (case-insensitive)
    2. Substring containment check
    """
    key = surface.lower().strip()

    # Exact match
    if key in ALIASES:
        return ALIASES[key]

    # Substring match: does any alias appear in the surface form?
    for alias, canonical in ALIASES.items():
        if alias in key or key in alias:
            return canonical

    return None


def normalize_triplet(triplet: dict) -> dict | None:
    """
    Normalize head and tail of a REBEL triplet to canonical IDs.
    Returns None if neither entity is recognized.
    """
    head_id = normalize(triplet["head"])
    tail_id = normalize(triplet["tail"])

    # Keep triplet only if at least one entity is recognized
    if head_id is None and tail_id is None:
        return None

    return {
        "head": head_id or _slugify(triplet["head"]),
        "head_label": triplet["head"],
        "relation": triplet["relation"].replace(" ", "_").lower(),
        "tail": tail_id or _slugify(triplet["tail"]),
        "tail_label": triplet["tail"],
        "source": triplet.get("source", ""),
    }


def get_entity_type(entity_id: str) -> str:
    return ENTITY_TYPES.get(entity_id, "Other")


def get_entity_color(entity_id: str) -> str:
    etype = get_entity_type(entity_id)
    return ENTITY_COLORS.get(etype, "#CCCCCC")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
