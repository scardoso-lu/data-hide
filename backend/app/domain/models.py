"""Shared domain value types for the PII anonymization pipeline.

This module is the single source of truth for column-policy constants,
policy/profile dataclasses, and the EntityRegistry used across the domain
layer.  Importing from here keeps ``classification.py`` and
``anonymization.py`` free of cross-module definition duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


# ─── Tier category constants ──────────────────────────────────────────────────

IDENTIFIER = "IDENTIFIER"
SENSITIVE = "SENSITIVE"
FREE_TEXT = "FREE_TEXT"
QUASI_IDENTIFIER = "QUASI_IDENTIFIER"


# ─── Action constants ─────────────────────────────────────────────────────────

ACTION_HASH = "hash"          # → KeyVault deterministic pseudonymizer
ACTION_TOKENIZE = "tokenize"  # → EntityRegistry (PERSON_0, PERSON_1, …)
ACTION_REDACT = "redact"      # → fixed sentinel ("[REDACTED]")
ACTION_BIN = "bin"            # → defer to existing bin/anonymize_gps_columns
ACTION_SCAN = "scan"          # → defer to row-by-row Presidio (free text)


# ─── Column-policy dataclasses ────────────────────────────────────────────────

@dataclass(frozen=True)
class ColumnPolicy:
    """Resolved policy for one column.

    Attributes
    ----------
    column : the original column name.
    entity_type : Presidio entity type (PERSON, EMAIL_ADDRESS, IDENTIFIER, …)
        or "FREE_TEXT" when the column should be scanned cell-by-cell.
    action : one of ACTION_HASH / ACTION_TOKENIZE / ACTION_REDACT /
        ACTION_BIN / ACTION_SCAN.
    source : which classifier tier produced this policy — "purview",
        "presidio_structured", "embedding_similarity" or "fallback".
        Carried through for auditability.
    score : confidence in [0.0, 1.0]; 1.0 for Purview (authoritative),
        Presidio-structured's aggregate confidence, or the spaCy cosine
        similarity, depending on tier.
    """

    column: str
    entity_type: str
    action: str
    source: str
    score: float


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    categories: tuple[str, ...]


# ─── Entity registry ──────────────────────────────────────────────────────────

class EntityRegistry:
    def __init__(self) -> None:
        self._map: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    def token_for(self, entity_type: str, original: str) -> str:
        key = (entity_type, original.strip().lower())
        if key not in self._map:
            n = self._counters.get(entity_type, 0)
            self._map[key] = f"{entity_type}_{n}"
            self._counters[entity_type] = n + 1
        return self._map[key]

    def unique_counts(self) -> dict[str, int]:
        return dict(self._counters)


# ─── Analyzer protocol ────────────────────────────────────────────────────────

@runtime_checkable
class AnalyzerProtocol(Protocol):
    def analyze(self, text: str, entities: list, language: str, score_threshold: float) -> list: ...
    def get_supported_entities(self, language: str) -> list[str]: ...
