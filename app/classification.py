"""Column classification helpers for GDPR-oriented anonymization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
from typing import Any, Iterable

import pandas as pd


IDENTIFIER = "IDENTIFIER"
SENSITIVE = "SENSITIVE"
FREE_TEXT = "FREE_TEXT"
QUASI_IDENTIFIER = "QUASI_IDENTIFIER"


# ─────────────────────────────────────────────────────────────────────────────
# Column-policy classification (Phase 1 scaffolding)
#
# A `ColumnPolicy` is the resolved decision for one column: which entity type
# it carries and what masking action to apply (hash / tokenise / redact /
# defer to binning / leave to the existing row-by-row scan).
#
# `classify_pii_columns()` (Phase 2) walks three tiers — Purview → Presidio-
# structured value sampling → spaCy embedding similarity against the
# CONCEPT_SEEDS dictionary below — and assigns one of these policies.
# ─────────────────────────────────────────────────────────────────────────────


# Actions a column policy can apply to every non-null cell in the column.
ACTION_HASH = "hash"          # → KeyVault deterministic pseudonymizer
ACTION_TOKENIZE = "tokenize"  # → EntityRegistry (PERSON_0, PERSON_1, …)
ACTION_REDACT = "redact"      # → fixed sentinel ("[REDACTED]")
ACTION_BIN = "bin"            # → defer to existing bin/anonymize_gps_columns
ACTION_SCAN = "scan"          # → defer to row-by-row Presidio (free text)


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


# Per-entity seed phrases in each supported language.  spaCy `_lg` models hold
# 300-dim GloVe vectors in their own language; classification iterates over
# all four models and takes the highest similarity, so a column called
# "prenom" matches PERSON via the French model and "Vorname" matches via the
# German model — without any column-name list.
CONCEPT_SEEDS: dict[str, dict[str, str]] = {
    "PERSON": {
        "en": "person name",
        "fr": "nom de personne",
        "de": "personenname",
        "lb": "personenumm",
    },
    "EMAIL_ADDRESS": {
        "en": "email address",
        "fr": "adresse e-mail",
        "de": "e-mail-adresse",
        "lb": "e-mail-adress",
    },
    "PHONE_NUMBER": {
        "en": "phone number",
        "fr": "numéro de téléphone",
        "de": "telefonnummer",
        "lb": "telefonsnummer",
    },
    "STREET_ADDRESS": {
        "en": "postal address",
        "fr": "adresse postale",
        "de": "postanschrift",
        "lb": "postadress",
    },
    "DATE_OF_BIRTH": {
        "en": "date of birth",
        "fr": "date de naissance",
        "de": "geburtsdatum",
        "lb": "gebuertsdatum",
    },
    "IDENTIFIER": {
        # Technical / database-centric seeds.  The generic "unique
        # identifier" / "identifiant unique" wording was too close to
        # "description" / "category" in spaCy's vector space (cosine ≈
        # 0.60) and produced false positives on free-text columns.
        "en": "primary key column",
        "fr": "clé primaire",
        "de": "primärschlüssel",
        "lb": "primärschlëssel",
    },
    "IBAN_CODE": {
        "en": "bank account number",
        "fr": "numéro de compte bancaire",
        "de": "bankkontonummer",
        "lb": "bankkontosnummer",
    },
    "CREDIT_CARD": {
        "en": "credit card number",
        "fr": "numéro de carte de crédit",
        "de": "kreditkartennummer",
        "lb": "kreditkaartnummer",
    },
    "SALARY": {
        "en": "salary amount",
        "fr": "montant du salaire",
        "de": "gehaltsbetrag",
        "lb": "salärbetrag",
    },
    "HEALTH_CONDITION": {
        # "medical condition" / "condition médicale" scored too close to
        # "description" / "condition" in spaCy's vector space.  "diagnosis"
        # is the more medically-specific anchor and pulls health-related
        # columns up while keeping generic descriptive columns below the
        # threshold.
        "en": "medical diagnosis",
        "fr": "diagnostic médical",
        "de": "medizinische diagnose",
        "lb": "medezinesch diagnos",
    },
    "FREE_TEXT": {
        "en": "free text comment",
        "fr": "commentaire en texte libre",
        "de": "freitext-kommentar",
        "lb": "fräi text kommentar",
    },
}

# How each entity type translates to a masking action.  Identifier-like
# entities are hashed deterministically (so equal source values map to equal
# pseudonyms across runs, preserving join keys), all other direct identifiers
# are tokenised per-run, DOB is deferred to the existing binning layer, and
# FREE_TEXT is deferred to row-by-row Presidio.
DEFAULT_ENTITY_ACTIONS: dict[str, str] = {
    # Identifier-like entities are hashed so equal source values produce
    # equal pseudonyms across runs (preserves join keys).
    "IDENTIFIER": ACTION_HASH,
    "CUSTOMER_EMPLOYEE_ID": ACTION_HASH,
    "US_SSN": ACTION_HASH,
    "US_ITIN": ACTION_HASH,
    "US_DRIVER_LICENSE": ACTION_HASH,
    "US_PASSPORT": ACTION_HASH,
    "LU_CCSS": ACTION_HASH,
    "LU_PASSPORT": ACTION_HASH,
    "NATIONAL_TAX_ID": ACTION_HASH,
    "EU_VAT": ACTION_HASH,
    "IBAN_CODE": ACTION_HASH,
    "CREDIT_CARD": ACTION_HASH,
    "MEDICAL_LICENSE": ACTION_HASH,
    "MEDICAL_RECORD": ACTION_HASH,
    "HEALTH_INSURANCE": ACTION_HASH,
    "SWIFT_BIC": ACTION_HASH,
    "INSURANCE_POLICY": ACTION_HASH,
    "VEHICLE_PLATE": ACTION_HASH,
    "CONTRACT_NUMBER": ACTION_HASH,
    "BOOKING_REF": ACTION_HASH,
    "COURT_CASE": ACTION_HASH,
    "INVOICE_NUMBER": ACTION_HASH,
    # Direct identifiers tokenised per-run via the EntityRegistry.
    "PERSON": ACTION_TOKENIZE,
    "EMAIL_ADDRESS": ACTION_TOKENIZE,
    "PHONE_NUMBER": ACTION_TOKENIZE,
    "STREET_ADDRESS": ACTION_TOKENIZE,
    "LOCATION": ACTION_TOKENIZE,
    "URL": ACTION_TOKENIZE,
    "IP_ADDRESS": ACTION_TOKENIZE,
    "CRYPTO": ACTION_TOKENIZE,
    "SALARY": ACTION_TOKENIZE,
    # GDPR Art. 9 / Art. 10 sensitive categories.
    "HEALTH_CONDITION": ACTION_TOKENIZE,
    "NRP": ACTION_TOKENIZE,
    "RELIGION": ACTION_TOKENIZE,
    "ETHNICITY": ACTION_TOKENIZE,
    "SEXUAL_ORIENTATION": ACTION_TOKENIZE,
    "TRADE_UNION": ACTION_TOKENIZE,
    "CRIMINAL_RECORD": ACTION_TOKENIZE,
    # Date columns are deferred to the existing temporal-binning layer
    # so day-level granularity is preserved for analytics.
    "DATE_OF_BIRTH": ACTION_BIN,
    "DATE_TIME": ACTION_BIN,
    # Free text: continue with the existing row-by-row Presidio scan.
    "FREE_TEXT": ACTION_SCAN,
}


# Default cosine-similarity threshold above which the embedding-similarity
# tier (B2) commits to a classification.  Overridable per-deployment via
# `COLUMN_SIMILARITY_THRESHOLD` — kept conservative so a column named e.g.
# `product_name` does not get tokenised as PERSON by accident.
DEFAULT_COLUMN_SIMILARITY_THRESHOLD: float = 0.55


def _column_similarity_threshold() -> float:
    raw = os.environ.get("COLUMN_SIMILARITY_THRESHOLD")
    if raw is None:
        return DEFAULT_COLUMN_SIMILARITY_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_COLUMN_SIMILARITY_THRESHOLD


def _presidio_structured_enabled() -> bool:
    raw = os.environ.get("ENABLE_PRESIDIO_STRUCTURED", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# Maps Purview's canonical classification type names → the entity types this
# pipeline already knows about.  The Purview taxonomy is maintained by
# Microsoft (closed set, stable across releases) so this mapping does not
# grow when new column-naming conventions emerge in customer datasets — it
# only changes when Microsoft introduces a new Purview classification type.
PURVIEW_TYPE_TO_ENTITY: dict[str, str] = {
    "MICROSOFT.PERSONAL.NAME": "PERSON",
    "MICROSOFT.PERSONAL.PERSON_NAME": "PERSON",
    "MICROSOFT.GENERAL.PERSON_NAME": "PERSON",
    "MICROSOFT.PERSONAL.EMAIL": "EMAIL_ADDRESS",
    "MICROSOFT.GENERAL.EMAIL": "EMAIL_ADDRESS",
    "MICROSOFT.PERSONAL.PHONE_NUMBER": "PHONE_NUMBER",
    "MICROSOFT.GENERAL.PHONE_NUMBER": "PHONE_NUMBER",
    "MICROSOFT.PERSONAL.PHYSICAL_ADDRESS": "STREET_ADDRESS",
    "MICROSOFT.PERSONAL.DATE_OF_BIRTH": "DATE_OF_BIRTH",
    "MICROSOFT.PERSONAL.IPADDRESS": "IP_ADDRESS",
    "MICROSOFT.FINANCIAL.CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "MICROSOFT.FINANCIAL.EU.IBAN_CODE": "IBAN_CODE",
    "MICROSOFT.FINANCIAL.IBAN_CODE": "IBAN_CODE",
    "MICROSOFT.FINANCIAL.US.SOCIAL_SECURITY_NUMBER": "US_SSN",
    "MICROSOFT.GOVERNMENT.US.SOCIAL_SECURITY_NUMBER": "US_SSN",
    "MICROSOFT.PERSONAL.NATIONAL_ID": "IDENTIFIER",
    "MICROSOFT.GENERAL.PERSON_ID": "IDENTIFIER",
}

_GPS_NAME_TOKENS = frozenset({
    "lat", "latitude", "lon", "lng", "longitude",
    "gps", "coord", "coords", "coordinate", "coordinates",
    "geom", "geometry", "wkt", "point",
})
_WKT_POINT_RE = re.compile(r"POINT\s*\(", re.IGNORECASE)

_TIMESTAMP_NAME_TOKENS = frozenset({
    "time", "timestamp", "datetime", "date", "at",
    "created", "updated", "recorded", "captured", "ts", "dt", "occurred",
})


def _is_text_column(dtype) -> bool:
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", name.lower()) if t]


def _sample(series: pd.Series, limit: int = 100) -> list[object]:
    return [v for v in series.dropna().head(limit).tolist()]


def _unique_ratio(values: Iterable[object]) -> float:
    vals = [str(v) for v in values if not pd.isna(v)]
    if not vals:
        return 0.0
    return len(set(vals)) / len(vals)


def _looks_like_identifier_values(values: list[object]) -> bool:
    if not values:
        return False
    text_values = [str(v).strip() for v in values if str(v).strip()]
    if not text_values:
        return False
    ratio = _unique_ratio(text_values)
    structured = sum(bool(re.fullmatch(r"[A-Za-z0-9_.:@/-]{3,128}", v)) for v in text_values)
    has_identifier_shape = sum(bool(re.search(r"\d", v) and re.search(r"[A-Za-z]", v)) for v in text_values)
    return ratio >= 0.75 and structured / len(text_values) >= 0.8 and has_identifier_shape / len(text_values) >= 0.5


def _looks_like_free_text(values: list[object]) -> bool:
    strings = [v for v in values if isinstance(v, str) and v.strip()]
    if not strings:
        return False
    avg_len = sum(len(v) for v in strings) / len(strings)
    avg_words = sum(len(v.split()) for v in strings) / len(strings)
    jsonish = sum(v.lstrip().startswith(("{", "[")) for v in strings) / len(strings)
    return avg_len >= 32 or avg_words >= 5 or jsonish >= 0.5


def _looks_like_quasi_identifier(series: pd.Series, values: list[object]) -> bool:
    if not values:
        return False
    ratio = _unique_ratio(values)
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return True
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        if numeric.empty:
            return False
        return ratio <= 0.4 and numeric.between(0, 130).mean() >= 0.8
    return 0 < ratio <= 0.35


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    categories: tuple[str, ...]


class ColumnClassifier:
    """Classifies columns from names and observed value shape.

    The classifier deliberately avoids project-specific keyword catalogs. It
    uses generic identifier naming conventions, cardinality, type, and sample
    text shape so it can adapt to new enterprise schemas.
    """

    def classify(self, df: pd.DataFrame) -> list[ColumnProfile]:
        profiles: list[ColumnProfile] = []
        for col in df.columns:
            series = df[col]
            values = _sample(series)
            tokens = _tokens(str(col))
            cats: list[str] = []

            if self._is_identifier_column(tokens, values):
                cats.append(IDENTIFIER)
            if _looks_like_quasi_identifier(series, values) and IDENTIFIER not in cats:
                cats.append(QUASI_IDENTIFIER)
            if _is_text_column(series.dtype) and _looks_like_free_text(values):
                cats.append(FREE_TEXT)
            if self._is_sensitive_shape(series, values):
                cats.append(SENSITIVE)

            profiles.append(ColumnProfile(col, tuple(dict.fromkeys(cats))))
        return profiles

    @staticmethod
    def _is_identifier_column(tokens: list[str], values: list[object]) -> bool:
        token_set = set(tokens)
        id_token = bool(token_set & {"id", "ids", "uuid", "guid", "identifier", "key"})
        id_suffix = any(t.endswith("id") and len(t) > 2 for t in tokens)
        return id_token or id_suffix or _looks_like_identifier_values(values)

    @staticmethod
    def _is_sensitive_shape(series: pd.Series, values: list[object]) -> bool:
        if not values:
            return False
        if pd.api.types.is_bool_dtype(series.dtype):
            return False
        non_null = len(values)
        unique = len({str(v) for v in values})
        entropy = 0.0
        counts: dict[str, int] = {}
        for v in values:
            counts[str(v)] = counts.get(str(v), 0) + 1
        for count in counts.values():
            p = count / non_null
            entropy -= p * math.log2(p)
        return unique <= max(8, non_null * 0.2) and entropy > 0 and non_null >= 5


def classify_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    return ColumnClassifier().classify(df)


def columns_by_category(df: pd.DataFrame, category: str) -> list[str]:
    return [p.name for p in classify_columns(df) if category in p.categories]


def flag_free_text_columns(df: pd.DataFrame) -> list[str]:
    return columns_by_category(df, FREE_TEXT)


def detect_quasi_identifiers(df: pd.DataFrame, explicit_cols: list[str] | None = None) -> list[str]:
    if explicit_cols:
        return [c for c in explicit_cols if c in df.columns]
    return columns_by_category(df, QUASI_IDENTIFIER)


def detect_identifier_columns(df: pd.DataFrame, explicit_cols: list[str] | None = None) -> list[str]:
    if explicit_cols:
        return [c for c in explicit_cols if c in df.columns]
    return columns_by_category(df, IDENTIFIER)


def detect_gps_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that contain GPS coordinates (numeric lat/lon or WKT POINT strings).

    A numeric column qualifies when its name contains a GPS keyword and all
    non-null values fall within [-180, 180].  A string column qualifies when
    ≥80 % of its non-null sample values match the WKT POINT(…) pattern.
    """
    result: list[str] = []
    for col in df.columns:
        series = df[col]
        tokens = set(_tokens(str(col)))
        if _is_numeric_gps_column(series, tokens) or _is_wkt_gps_column(series):
            result.append(col)
    return result


def _is_numeric_gps_column(series: pd.Series, tokens: set[str]) -> bool:
    if not (tokens & _GPS_NAME_TOKENS):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.isna().any():
        return False
    return bool(numeric.between(-180, 180).all())


def detect_timestamp_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that contain timestamps or dates.

    A datetime64 column always qualifies.  A string/object column qualifies
    when its name contains a timestamp keyword and ≥80 % of the non-null
    sample parses as a datetime.
    """
    result: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            result.append(col)
            continue
        if not _is_text_column(series.dtype):
            continue
        tokens = set(_tokens(str(col)))
        if not (tokens & _TIMESTAMP_NAME_TOKENS):
            continue
        sample = series.dropna().head(20)
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                result.append(col)
        except Exception:
            pass
    return result


def _is_wkt_gps_column(series: pd.Series) -> bool:
    if not _is_text_column(series.dtype):
        return False
    sample = [v for v in series.dropna().head(20).tolist() if isinstance(v, str)]
    if not sample:
        return False
    matches = sum(1 for v in sample if _WKT_POINT_RE.match(v.strip()))
    return matches / len(sample) >= 0.8


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — three-tier PII column classifier.
#
# `classify_pii_columns(df)` walks each column through:
#   A.  Purview classifications (when supplied)            — authoritative
#   B1. presidio-structured value sampling                 — value signals
#   B2. spaCy embedding similarity vs CONCEPT_SEEDS         — name signals
#   C.  Free-text fallback                                  — backstop
#
# Returns `{column_name: ColumnPolicy}`.  Columns whose dtype is non-text and
# that no tier could classify are omitted.
# ─────────────────────────────────────────────────────────────────────────────


def _make_policy(column: str, entity_type: str, *, source: str, score: float) -> ColumnPolicy:
    action = DEFAULT_ENTITY_ACTIONS.get(entity_type, ACTION_SCAN)
    return ColumnPolicy(
        column=column,
        entity_type=entity_type,
        action=action,
        source=source,
        score=score,
    )


_COLUMN_NAME_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalise_column_name(name: str) -> str:
    """Convert `cust_id` / `firstName` / `First-Name` → `first name` / `cust id`."""
    # camelCase / PascalCase → snake_case before splitting on separators.
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    cleaned = _COLUMN_NAME_NON_ALNUM_RE.sub(" ", snake.lower()).strip()
    return cleaned


# Module-level cache so the four spaCy `_lg` models — each ~600MB — are
# loaded exactly once per process.  Calling `classify_pii_columns()`
# repeatedly (e.g. one call per Delta table in a multi-table run) reuses
# the same `nlp` objects instead of re-instantiating them.
_SIMILARITY_MODEL_CACHE: dict[frozenset, dict[str, Any]] = {}


def _load_similarity_models(supported_languages: Iterable[str] | None = None) -> dict[str, Any]:
    """Load (and cache) each language's spaCy `_lg` model for embedding
    similarity.

    Imports happen inside the function so that environments without spaCy
    (e.g. unit-test runs with the FakeAnalyzer) don't fail at import time.
    Returns a `{lang: nlp}` dict; languages whose model fails to load are
    silently skipped — Tier B2 then simply uses the languages that loaded.
    Subsequent calls with the same language set return the cached dict.
    """
    if supported_languages is None:
        try:
            from .anonymization import SPACY_MODELS, SUPPORTED_LANGUAGES
            model_overrides = SPACY_MODELS
            langs = list(SUPPORTED_LANGUAGES)
        except Exception:
            return {}
    else:
        from .anonymization import SPACY_MODELS as model_overrides
        langs = list(supported_languages)

    cache_key = frozenset((lang, model_overrides.get(lang, "")) for lang in langs)
    cached = _SIMILARITY_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import spacy
    except Exception:
        return {}

    models: dict[str, Any] = {}
    for lang in langs:
        model_name = model_overrides.get(lang)
        if not model_name:
            continue
        try:
            models[lang] = spacy.load(model_name)
        except Exception:
            continue

    _SIMILARITY_MODEL_CACHE[cache_key] = models
    return models


def _best_entity_for_column_name(
    column_name: str,
    similarity_models: dict[str, Any],
) -> tuple[str | None, float]:
    """Return the best `(entity_type, similarity)` for the column name across
    every loaded spaCy language model.

    Each language model only compares the column phrase against ITS OWN
    language's seed (e.g. the French model only sees French seeds).  Cross-
    language pairing is intentionally rejected because spaCy GloVe vectors
    treat cognates (`description` vs `identifier`) as semantically close
    even when they aren't — that misclassified `description` as IDENTIFIER
    under the French model in early testing.  By forcing same-language
    comparisons we rely on each model's own monolingual vector space.

    Multilingual coverage still works because each column name is fed to
    EVERY loaded model; whichever language matches gives the strongest
    score — French `prenom` ↔ French `nom de personne`, German `Vorname`
    ↔ German `personenname`, etc.
    """
    phrase = _normalise_column_name(column_name)
    if not phrase:
        return None, 0.0

    best_entity: str | None = None
    best_score: float = 0.0
    for lang, nlp in similarity_models.items():
        try:
            col_doc = nlp(phrase)
        except Exception:
            continue
        if not col_doc.has_vector or col_doc.vector_norm == 0:
            continue
        for entity, seeds in CONCEPT_SEEDS.items():
            seed_phrase = seeds.get(lang)
            if not seed_phrase:
                continue
            try:
                seed_doc = nlp(seed_phrase)
            except Exception:
                continue
            if not seed_doc.has_vector or seed_doc.vector_norm == 0:
                continue
            try:
                score = float(col_doc.similarity(seed_doc))
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_entity = entity
    return best_entity, best_score


def _tier_a_purview(
    df: pd.DataFrame,
    purview_classifications: dict[str, str] | None,
    policies: dict[str, ColumnPolicy],
) -> None:
    """Apply Purview-supplied column classifications.  No-op when the caller
    didn't pass a mapping (Purview not configured / unreachable)."""
    if not purview_classifications:
        return
    for column, purview_type in purview_classifications.items():
        if column not in df.columns or column in policies:
            continue
        entity = PURVIEW_TYPE_TO_ENTITY.get(str(purview_type).upper())
        if not entity:
            continue
        policies[column] = _make_policy(column, entity, source="purview", score=1.0)


def _tier_b1_presidio_structured(
    df: pd.DataFrame,
    analyzer: Any | None,
    policies: dict[str, ColumnPolicy],
    enabled: bool,
) -> None:
    """Sample column values, run Presidio per cell, aggregate to a
    {column: entity_type} map for any column where the dominant entity
    crosses presidio-structured's default vote threshold."""
    if not enabled or analyzer is None:
        return
    candidate_cols = [
        c for c in df.columns
        if c not in policies and _is_text_column(df[c].dtype)
    ]
    if not candidate_cols:
        return
    try:
        from presidio_structured import PandasAnalysisBuilder
        builder = PandasAnalysisBuilder(analyzer=analyzer)
        analysis = builder.generate_analysis(df[candidate_cols])
    except Exception:
        return  # log path — caller wraps this in a logger.warning if desired
    for column, entity in (analysis.entity_mapping or {}).items():
        if column in policies or not entity:
            continue
        policies[column] = _make_policy(
            column, str(entity), source="presidio_structured", score=0.9,
        )


def _tier_b2_embedding(
    df: pd.DataFrame,
    similarity_models: dict[str, Any],
    threshold: float,
    policies: dict[str, ColumnPolicy],
) -> None:
    """spaCy embedding similarity between each unclassified column name and
    every CONCEPT_SEEDS entry, across every loaded language."""
    if not similarity_models:
        return
    for column in df.columns:
        if column in policies or not _is_text_column(df[column].dtype):
            continue
        entity, score = _best_entity_for_column_name(column, similarity_models)
        if entity is None or score < threshold:
            continue
        policies[column] = _make_policy(
            column, entity, source="embedding_similarity", score=score,
        )


def _tier_c_fallback(df: pd.DataFrame, policies: dict[str, ColumnPolicy]) -> None:
    """Unclassified text columns drop to a FREE_TEXT policy → action=SCAN,
    which keeps the existing row-by-row Presidio scan as the backstop."""
    for column in df.columns:
        if column in policies or not _is_text_column(df[column].dtype):
            continue
        policies[column] = _make_policy(
            column, FREE_TEXT, source="fallback", score=0.0,
        )


def classify_pii_columns(
    df: pd.DataFrame,
    *,
    purview_classifications: dict[str, str] | None = None,
    analyzer: Any | None = None,
    similarity_models: dict[str, Any] | None = None,
    similarity_threshold: float | None = None,
    structured_enabled: bool | None = None,
) -> dict[str, ColumnPolicy]:
    """Walk three tiers (Purview → Presidio-structured → spaCy similarity)
    and return one `ColumnPolicy` per text column.

    Parameters
    ----------
    df : the source DataFrame.
    purview_classifications : optional ``{column_name: purview_type_name}``
        mapping fetched out-of-band from Microsoft Purview.  Tier A entries
        bypass downstream tiers (authoritative).
    analyzer : a Presidio `AnalyzerEngine` (or compatible shim) used by
        Tier B1 to sample column values.  When None, Tier B1 is skipped.
    similarity_models : pre-loaded ``{lang_code: spacy.nlp}`` dict used by
        Tier B2.  When None, the function loads the SPACY_MODELS set
        configured in ``app.anonymization``; pass an explicit dict in tests
        to avoid the heavy model load.
    similarity_threshold : cosine cut-off above which a B2 match commits.
        Defaults to env ``COLUMN_SIMILARITY_THRESHOLD`` or 0.55.
    structured_enabled : opt-out for Tier B1.  Defaults to env
        ``ENABLE_PRESIDIO_STRUCTURED`` (on by default).

    Returns
    -------
    dict[str, ColumnPolicy] keyed by column name.  Non-text columns and
    columns no tier could place are absent from the result; the caller's
    pipeline continues to treat them with existing layers (GPS, bin, etc.).
    """
    policies: dict[str, ColumnPolicy] = {}

    _tier_a_purview(df, purview_classifications, policies)

    if structured_enabled is None:
        structured_enabled = _presidio_structured_enabled()
    _tier_b1_presidio_structured(df, analyzer, policies, structured_enabled)

    if similarity_models is None:
        similarity_models = _load_similarity_models()
    threshold = similarity_threshold if similarity_threshold is not None else _column_similarity_threshold()
    _tier_b2_embedding(df, similarity_models, threshold, policies)

    _tier_c_fallback(df, policies)

    return policies


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — apply policies.
# ─────────────────────────────────────────────────────────────────────────────


REDACTED_SENTINEL = "[REDACTED]"


def apply_column_policies(
    df: pd.DataFrame,
    policies: dict[str, ColumnPolicy],
    *,
    registry: Any | None = None,
    pseudonymizer: Any | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply each `ColumnPolicy`'s action to every non-null cell in its column.

    ``ACTION_HASH``      → ``pseudonymizer(value)``     (deterministic, joinable)
    ``ACTION_TOKENIZE``  → ``registry.token_for(entity_type, value)``
    ``ACTION_REDACT``    → ``REDACTED_SENTINEL``
    ``ACTION_BIN``       → no-op (deferred to existing binning layers)
    ``ACTION_SCAN``      → no-op (deferred to row-by-row Presidio scan)

    Returns a fresh DataFrame plus a stats dict::

        {
          "columns_processed": [list of column names actually mutated],
          "actions_applied": {col: action},
          "entity_types": {col: entity_type},
          "values_masked": {col: int},
          "skipped_columns": {col: reason},
        }

    The stats dict is intended for the audit log so an operator can verify
    which classifier tier acted on which column.

    The DataFrame is copied before mutation so callers can keep the original.
    """
    df = df.copy()
    stats = {
        "columns_processed": [],
        "actions_applied": {},
        "entity_types": {},
        "values_masked": {},
        "skipped_columns": {},
    }

    for column, policy in policies.items():
        if column not in df.columns:
            stats["skipped_columns"][column] = "missing_column"
            continue

        if policy.action == ACTION_BIN:
            stats["skipped_columns"][column] = "deferred_to_binning"
            continue
        if policy.action == ACTION_SCAN:
            stats["skipped_columns"][column] = "deferred_to_row_scan"
            continue

        if policy.action == ACTION_HASH and pseudonymizer is None:
            stats["skipped_columns"][column] = "missing_pseudonymizer"
            continue
        if policy.action == ACTION_TOKENIZE and registry is None:
            stats["skipped_columns"][column] = "missing_registry"
            continue

        mutated = _apply_one_policy(df, column, policy, registry, pseudonymizer)
        if mutated == 0:
            stats["skipped_columns"][column] = "no_non_null_values"
            continue

        stats["columns_processed"].append(column)
        stats["actions_applied"][column] = policy.action
        stats["entity_types"][column] = policy.entity_type
        stats["values_masked"][column] = mutated

    return df, stats


def _apply_one_policy(
    df: pd.DataFrame,
    column: str,
    policy: ColumnPolicy,
    registry: Any | None,
    pseudonymizer: Any | None,
) -> int:
    """In-place mask of one column according to its policy.  Returns the
    number of cells mutated (non-null inputs)."""
    series = df[column]
    mask = series.notna()
    if not mask.any():
        return 0

    if policy.action == ACTION_HASH:
        df.loc[mask, column] = series.loc[mask].map(pseudonymizer)
    elif policy.action == ACTION_TOKENIZE:
        df.loc[mask, column] = series.loc[mask].map(
            lambda v, e=policy.entity_type: registry.token_for(e, str(v))
        )
    elif policy.action == ACTION_REDACT:
        df.loc[mask, column] = REDACTED_SENTINEL

    return int(mask.sum())


def free_text_columns_from_policies(policies: dict[str, ColumnPolicy]) -> list[str]:
    """List of columns whose policy is ACTION_SCAN — the row-by-row Presidio
    layer should optionally apply more aggressive thresholds here."""
    return [
        column for column, policy in policies.items()
        if policy.action == ACTION_SCAN
    ]


def already_masked_columns_from_policies(policies: dict[str, ColumnPolicy]) -> set[str]:
    """Columns that the column-policy layer has already masked — the row-by-
    row layer must skip them or it would tokenise the existing PERSON_0
    tokens as PERSON_1, etc."""
    return {
        column for column, policy in policies.items()
        if policy.action in (ACTION_HASH, ACTION_TOKENIZE, ACTION_REDACT)
    }

