"""Column classification helpers for GDPR-oriented anonymization.

Polars-native: every function in this module accepts and returns
``polars.DataFrame`` / ``polars.Series``.  pandas appears at exactly one
boundary — the 500-row sample handed to presidio-structured (Tier B1),
which only ships a pandas builder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
from typing import Any, Iterable

import polars as pl


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


# Per-entity seed phrases in each supported language.  spaCy models hold
# word vectors in their own language; classification iterates over all four
# models and takes the highest similarity, so a column called "prenom"
# matches PERSON via the French model and "Vorname" matches via the German
# model — without any column-name list.
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


def _is_text_column(dtype: pl.DataType) -> bool:
    """String columns plus Object columns (heterogeneous Python values)."""
    return dtype == pl.String or dtype == pl.Object


def _is_temporal_dtype(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.Datetime) or dtype == pl.Date


def _is_null_value(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", name.lower()) if t]


def _sample(series: pl.Series, limit: int = 100) -> list[object]:
    return series.drop_nulls().head(limit).to_list()


def _parse_datetime_ratio(values: list[object]) -> float:
    """Fraction of values that parse as a datetime (lenient, dateutil-based)."""
    from dateutil import parser as _date_parser

    if not values:
        return 0.0
    parsed = 0
    for v in values:
        if not isinstance(v, str) or not v.strip():
            continue
        try:
            _date_parser.parse(v)
            parsed += 1
        except (ValueError, OverflowError, TypeError):
            continue
    return parsed / len(values)


def _unique_ratio(values: Iterable[object]) -> float:
    vals = [str(v) for v in values if not _is_null_value(v)]
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


def _looks_like_quasi_identifier(series: pl.Series, values: list[object]) -> bool:
    if not values:
        return False
    ratio = _unique_ratio(values)
    if _is_temporal_dtype(series.dtype):
        return True
    if series.dtype.is_numeric():
        numeric = [float(v) for v in values if not _is_null_value(v)]
        if not numeric:
            return False
        in_range = sum(1 for v in numeric if 0 <= v <= 130) / len(numeric)
        return ratio <= 0.4 and in_range >= 0.8
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

    def classify(self, df: pl.DataFrame) -> list[ColumnProfile]:
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
        id_token = bool(token_set & _ID_NAME_TOKENS)
        # Composite names like "employeeid" / "customerid" where the suffix "id"
        # is merged without a separator — require length > 2 so the bare token
        # "id" (already in _ID_NAME_TOKENS above) is not double-counted.
        id_suffix = any(t.endswith("id") and len(t) > 2 for t in tokens)
        return id_token or id_suffix or _looks_like_identifier_values(values)

    @staticmethod
    def _is_sensitive_shape(series: pl.Series, values: list[object]) -> bool:
        if not values:
            return False
        if series.dtype == pl.Boolean:
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


def classify_columns(df: pl.DataFrame) -> list[ColumnProfile]:
    return ColumnClassifier().classify(df)


def columns_by_category(df: pl.DataFrame, category: str) -> list[str]:
    return [p.name for p in classify_columns(df) if category in p.categories]


def flag_free_text_columns(df: pl.DataFrame) -> list[str]:
    return columns_by_category(df, FREE_TEXT)


def detect_quasi_identifiers(df: pl.DataFrame, explicit_cols: list[str] | None = None) -> list[str]:
    if explicit_cols:
        return [c for c in explicit_cols if c in df.columns]
    return columns_by_category(df, QUASI_IDENTIFIER)


def detect_identifier_columns(df: pl.DataFrame, explicit_cols: list[str] | None = None) -> list[str]:
    if explicit_cols:
        return [c for c in explicit_cols if c in df.columns]
    return columns_by_category(df, IDENTIFIER)


def has_tracking_columns(df: pl.DataFrame, gps_cols: list[str] | None = None) -> bool:
    """Return True when the DataFrame looks like a tracking table.

    K-anonymity is only meaningful for tables that contain location or network
    tracking data — GPS coordinates, IP addresses, or movement records.
    Applying it to regular business tables (HR, finance, absence records)
    suppresses legitimate data without a privacy benefit.

    A table is considered a tracking table when:
    * GPS columns were detected (caller passes ``gps_cols`` if already known), OR
    * Any column name contains the standalone token ``"ip"`` — covers
      ``ip_address``, ``source_ip``, ``client_ip``, ``remote_ip``, etc.
    """
    if gps_cols:
        return True
    for col in df.columns:
        if "ip" in set(_tokens(str(col))):
            return True
    return False


def detect_gps_columns(df: pl.DataFrame) -> list[str]:
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


def _is_numeric_gps_column(series: pl.Series, tokens: set[str]) -> bool:
    if not (tokens & _GPS_NAME_TOKENS):
        return False
    non_null = series.drop_nulls()
    if len(non_null) == 0:
        return False
    try:
        if series.dtype == pl.Object:
            values = [float(v) for v in non_null.to_list()]
        elif series.dtype == pl.String:
            cast = non_null.cast(pl.Float64, strict=False)
            if cast.null_count() > 0:
                return False
            values = cast.to_list()
        elif series.dtype.is_numeric():
            values = non_null.cast(pl.Float64).to_list()
        else:
            return False
    except (ValueError, TypeError):
        return False
    return all(-180 <= v <= 180 for v in values)


def detect_timestamp_columns(df: pl.DataFrame) -> list[str]:
    """Return column names that contain timestamps or dates.

    A temporal (Datetime/Date) column always qualifies.  A string/object
    column qualifies when its name contains a timestamp keyword and ≥80 % of
    the non-null sample parses as a datetime.
    """
    result: list[str] = []
    for col in df.columns:
        series = df[col]
        if _is_temporal_dtype(series.dtype):
            result.append(col)
            continue
        if not _is_text_column(series.dtype):
            continue
        tokens = set(_tokens(str(col)))
        if not (tokens & _TIMESTAMP_NAME_TOKENS):
            continue
        sample = _sample(series, limit=20)
        if not sample:
            continue
        if _parse_datetime_ratio(sample) >= 0.8:
            result.append(col)
    return result


def _is_wkt_gps_column(series: pl.Series) -> bool:
    if not _is_text_column(series.dtype):
        return False
    sample = [v for v in _sample(series, limit=20) if isinstance(v, str)]
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


# Tokens that unambiguously mark a column as an identifier regardless of its
# values.  Used by both `ColumnClassifier._is_identifier_column` (the simple
# classify_columns path) and `_tier_a1_name_pattern` (the PII-column classifier
# path) so both paths apply exactly the same name-based rule.
#
# The set deliberately excludes ambiguous words ("number", "code", "ref") that
# can appear in non-identifier columns.  Only tokens where the name alone is
# sufficient evidence that the column holds a join key or primary key are
# included here.
_ID_NAME_TOKENS: frozenset[str] = frozenset({
    "id", "ids", "uuid", "guid", "identifier", "identifiers", "key", "pk", "fk",
})

_COLUMN_NAME_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalise_column_name(name: str) -> str:
    """Convert `cust_id` / `firstName` / `First-Name` → `first name` / `cust id`."""
    # camelCase / PascalCase → snake_case before splitting on separators.
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    cleaned = _COLUMN_NAME_NON_ALNUM_RE.sub(" ", snake.lower()).strip()
    return cleaned


# Module-level cache so the spaCy models are loaded exactly once per process.
# Calling `classify_pii_columns()` repeatedly (e.g. one call per Delta table
# in a multi-table run) reuses the same `nlp` objects instead of
# re-instantiating them.
_SIMILARITY_MODEL_CACHE: dict[frozenset, dict[str, Any]] = {}

# Single-resident slot for the sequential Tier B2 walk: (model_name, nlp).
# Holding at most ONE extra spaCy model at a time caps peak RSS — the
# previous model is dereferenced (and gc'd) before the next one loads.
_SEQUENTIAL_MODEL_SLOT: list = [None]


def _models_from_analyzer(analyzer: Any) -> dict[str, Any]:
    """Extract the already-loaded spaCy pipelines from a Presidio analyzer.

    Presidio's ``SpacyNlpEngine`` keeps its models in ``nlp_engine.nlp``
    (``{lang_code: spacy.Language}``).  Reusing them for Tier B2 embedding
    similarity avoids loading a SECOND copy of each model via
    ``spacy.load`` — previously the single largest memory cost of the
    pipeline (every model resident twice).
    """
    try:
        nlp_map = getattr(getattr(analyzer, "nlp_engine", None), "nlp", None)
        if isinstance(nlp_map, dict) and nlp_map:
            return dict(nlp_map)
    except Exception:
        pass
    return {}


def _sequential_model_for(lang: str) -> Any | None:
    """Load one language's spaCy model into the single-resident slot.

    Returns the model, or None when loading fails (Tier B2 then simply skips
    that language).  Requesting a different model releases the previous one
    first, so at most one sequentially-loaded model is resident at any time.
    """
    from .anonymization import SPACY_MODELS

    model_name = SPACY_MODELS.get(lang)
    if not model_name:
        return None
    slot = _SEQUENTIAL_MODEL_SLOT[0]
    if slot is not None and slot[0] == model_name:
        return slot[1]
    # Release the previous model BEFORE loading the next so the two never
    # coexist in memory; trim so the freed pages leave RSS too.
    _SEQUENTIAL_MODEL_SLOT[0] = None
    import gc
    gc.collect()
    from .anonymization import _trim_native_heap
    _trim_native_heap()
    try:
        import spacy
        nlp = spacy.load(model_name)
    except Exception:
        return None
    _SEQUENTIAL_MODEL_SLOT[0] = (model_name, nlp)
    return nlp


def release_sequential_model() -> None:
    """Free the single-resident Tier B2 model (called between tables/runs)."""
    _SEQUENTIAL_MODEL_SLOT[0] = None
    import gc
    gc.collect()
    from .anonymization import _trim_native_heap
    _trim_native_heap()


def _load_similarity_models(supported_languages: Iterable[str] | None = None) -> dict[str, Any]:
    """Load (and cache) each language's spaCy model for embedding similarity.

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
    df: pl.DataFrame,
    purview_classifications: dict[str, str | list[str]] | None,
    policies: dict[str, ColumnPolicy],
    *,
    purview_must_anonymize_type: str | None = None,
) -> None:
    """Apply Purview-supplied column classifications.  No-op when the caller
    didn't pass a mapping (Purview not configured / unreachable).

    Each column value may be a single classification type string (legacy /
    test path) or a list of type strings as returned by the Purview catalog
    API.  When ``purview_must_anonymize_type`` is set, any column carrying
    that type (case-insensitive) is assigned ``ACTION_REDACT`` directly,
    overriding even a pre-computed policy that arrived from the Phase 1
    sampling pass.  Known Microsoft types respect the ``column in policies``
    guard so that Phase 1 results are not silently overwritten.
    """
    if not purview_classifications:
        return
    must_upper = purview_must_anonymize_type.upper() if purview_must_anonymize_type else None
    for column, raw_types in purview_classifications.items():
        if column not in df.columns:
            continue
        types: list[str] = [raw_types] if isinstance(raw_types, str) else list(raw_types)
        # Must-anonymize is authoritative: overrides any pre-computed policy.
        if must_upper and any(t.upper() == must_upper for t in types):
            policies[column] = ColumnPolicy(
                column=column,
                entity_type="MUST_ANONYMIZE",
                action=ACTION_REDACT,
                source="purview",
                score=1.0,
            )
            continue
        # Known Microsoft types do not override existing policies — Phase 1
        # sampling results (name-pattern, presidio-structured, spaCy) stand.
        if column in policies:
            continue
        for t in types:
            entity = PURVIEW_TYPE_TO_ENTITY.get(t.upper())
            if entity:
                policies[column] = _make_policy(column, entity, source="purview", score=1.0)
                break


def _tier_a1_name_pattern(df: pl.DataFrame, policies: dict[str, ColumnPolicy]) -> None:
    """Pin identifier-named columns to ACTION_HASH before value-sampling tiers run.

    Presidio-structured (Tier B1) and spaCy embedding (Tier B2) both look at
    *values* or *vector similarity*, so a column named ``employee_id`` whose
    values happen to contain alphanumeric codes can be misclassified as PERSON
    or another PII entity — leading to tokenisation instead of deterministic
    hashing and breaking downstream join keys.

    This tier runs AFTER Purview (Tier A, which is authoritative and keeps any
    Purview-supplied classification) but BEFORE B1/B2, so the name signal takes
    priority over value-content signals for identifier-named columns.

    Two naming patterns are recognised:

    * **Standalone token** — the column name tokenises to one of ``_ID_NAME_TOKENS``
      (e.g. ``id``, ``uuid``, ``guid``, ``key``).
    * **Compound suffix** — the column name ends with an ``_ID_NAME_TOKENS`` token
      after splitting on non-alphanumeric separators, e.g. ``employee_id``,
      ``customer_id``, ``employeeid``, ``customerUUID``.

    Only text-typed columns are evaluated; numeric/boolean identifier columns
    are left to the existing binning layer.
    """
    for column in df.columns:
        if column in policies:
            continue
        if not _is_text_column(df.schema[column]):
            continue
        tokens = _tokens(str(column))
        if not tokens:
            continue
        token_set = set(tokens)
        is_id_name = (
            bool(token_set & _ID_NAME_TOKENS)
            # Unseparated composites: "employeeid", "customerid"
            or any(t.endswith("id") and len(t) > 2 for t in tokens)
        )
        if is_id_name:
            policies[column] = _make_policy(
                column, "IDENTIFIER", source="name_pattern", score=1.0,
            )


def _tier_b1_presidio_structured(
    df: pl.DataFrame,
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
        if c not in policies and _is_text_column(df.schema[c])
    ]
    if not candidate_cols:
        return
    # A few hundred rows is sufficient to determine dominant entity types per
    # column; scanning the full table here is pure waste since the same rows
    # get a second Presidio pass during anonymization.
    _TIER_B1_SAMPLE_ROWS = 500
    try:
        from presidio_structured import PandasAnalysisBuilder
        builder = PandasAnalysisBuilder(analyzer=analyzer)
        # presidio-structured only ships a pandas builder — this is the single
        # pandas boundary in the module.  500 rows: the conversion is
        # negligible.  astype(object) matches the input shape the library is
        # exercised against upstream.
        sample = df.select(candidate_cols).head(_TIER_B1_SAMPLE_ROWS).to_pandas().astype(object)
        analysis = builder.generate_analysis(sample)
    except Exception:
        return  # log path — caller wraps this in a logger.warning if desired
    for column, entity in (analysis.entity_mapping or {}).items():
        if column in policies or not entity:
            continue
        policies[column] = _make_policy(
            column, str(entity), source="presidio_structured", score=0.9,
        )


def _score_columns_for_language(
    columns: list[str],
    lang: str,
    nlp: Any,
    best: dict[str, tuple[str | None, float]],
) -> None:
    """Score every candidate column name against ``lang``'s seeds with one
    model, updating the running per-column best.  Same-language comparison
    only — see `_best_entity_for_column_name` for the rationale."""
    for column in columns:
        entity, score = _best_entity_for_column_name(column, {lang: nlp})
        prev_entity, prev_score = best.get(column, (None, 0.0))
        if entity is not None and score > prev_score:
            best[column] = (entity, score)


def _tier_b2_embedding(
    df: pl.DataFrame,
    similarity_models: dict[str, Any] | None,
    threshold: float,
    policies: dict[str, ColumnPolicy],
    *,
    sequential_languages: list[str] | None = None,
) -> None:
    """spaCy embedding similarity between each unclassified column name and
    every CONCEPT_SEEDS entry, across every supported language.

    Memory model: languages whose models are already resident (passed in
    ``similarity_models``, e.g. extracted from the analyzer) are scored
    first.  The remaining ``sequential_languages`` are walked ONE MODEL AT A
    TIME through the single-resident slot — load, score all columns,
    release, next — so peak RSS never holds more than one extra model.
    """
    candidates = [
        c for c in df.columns
        if c not in policies and _is_text_column(df.schema[c])
    ]
    if not candidates:
        return

    best: dict[str, tuple[str | None, float]] = {}

    for lang, nlp in (similarity_models or {}).items():
        _score_columns_for_language(candidates, lang, nlp, best)

    for lang in (sequential_languages or []):
        nlp = _sequential_model_for(lang)
        if nlp is None:
            continue
        _score_columns_for_language(candidates, lang, nlp, best)

    for column, (entity, score) in best.items():
        if entity is None or score < threshold:
            continue
        policies[column] = _make_policy(
            column, entity, source="embedding_similarity", score=score,
        )


def _tier_c_fallback(df: pl.DataFrame, policies: dict[str, ColumnPolicy]) -> None:
    """Unclassified text columns get a FREE_TEXT fallback policy.

    Columns whose sampled values look like free text (long strings, many
    words, or JSON blobs) get ACTION_SCAN so the row-by-row Presidio pass
    covers them.  Short structured columns (codes, categories, flags) get
    ACTION_BIN, which skips both the column-policy masking layer and the
    row-by-row scan — Presidio would find nothing useful there anyway.
    """
    for column in df.columns:
        if column in policies or not _is_text_column(df.schema[column]):
            continue
        action = ACTION_SCAN if _looks_like_free_text(_sample(df[column])) else ACTION_BIN
        policies[column] = ColumnPolicy(
            column=column,
            entity_type=FREE_TEXT,
            action=action,
            source="fallback",
            score=0.0,
        )


def classify_pii_columns(
    df: pl.DataFrame,
    *,
    purview_classifications: dict[str, str | list[str]] | None = None,
    purview_must_anonymize_type: str | None = None,
    analyzer: Any | None = None,
    similarity_models: dict[str, Any] | None = None,
    similarity_threshold: float | None = None,
    structured_enabled: bool | None = None,
) -> dict[str, ColumnPolicy]:
    """Walk four tiers and return one `ColumnPolicy` per text column.

    Tier order
    ----------
    A.   Purview classifications (authoritative — supplied externally).
    A1.  Name-pattern pre-classification: columns whose name contains a
         recognised identifier token (``id``, ``uuid``, ``guid``, ``key``, or
         a compound suffix like ``employee_id`` / ``customer_id``) are pinned
         to ``ACTION_HASH`` *before* value-sampling tiers run.  This prevents
         Presidio-structured and spaCy from misclassifying identifier columns
         as ``PERSON`` / ``EMAIL_ADDRESS`` etc. based on their *values*.
    B1.  Presidio-structured value sampling.
    B2.  spaCy embedding similarity vs ``CONCEPT_SEEDS``.
    C.   Free-text fallback.

    Parameters
    ----------
    df : the source DataFrame (Polars).
    purview_classifications : optional ``{column_name: purview_type_name}``
        mapping fetched out-of-band from Microsoft Purview.  Tier A entries
        bypass all downstream tiers (authoritative).
    analyzer : a Presidio ``AnalyzerEngine`` (or compatible shim) used by
        Tier B1 to sample column values.  When None, Tier B1 is skipped.
    similarity_models : pre-loaded ``{lang_code: spacy.nlp}`` dict used by
        Tier B2.  When None the function loads the SPACY_MODELS set
        configured in ``app.anonymization``; pass an explicit dict in tests
        to avoid the heavy model load.
    similarity_threshold : cosine cut-off above which a B2 match commits.
        Defaults to env ``COLUMN_SIMILARITY_THRESHOLD`` or 0.55.
    structured_enabled : opt-out for Tier B1.  Defaults to env
        ``ENABLE_PRESIDIO_STRUCTURED`` (on by default).

    Returns
    -------
    dict[str, ColumnPolicy] keyed by column name.  Non-text columns and
    columns no tier could place are absent from the result.
    """
    policies: dict[str, ColumnPolicy] = {}

    _tier_a_purview(df, purview_classifications, policies,
                    purview_must_anonymize_type=purview_must_anonymize_type)
    _tier_a1_name_pattern(df, policies)  # pin id-named columns before value-sampling

    if structured_enabled is None:
        structured_enabled = _presidio_structured_enabled()
    _tier_b1_presidio_structured(df, analyzer, policies, structured_enabled)

    threshold = similarity_threshold if similarity_threshold is not None else _column_similarity_threshold()
    if similarity_models is not None:
        # Caller-supplied models (tests, custom deployments): use as-is.
        _tier_b2_embedding(df, similarity_models, threshold, policies)
    else:
        # Default path: reuse whatever models the analyzer already has
        # resident (zero extra memory), then cover the remaining languages
        # sequentially — one model at a time through the single-resident
        # slot.  This replaces the old behaviour of spacy.load-ing a SECOND
        # copy of every model into _SIMILARITY_MODEL_CACHE.
        from .anonymization import SPACY_MODELS, SUPPORTED_LANGUAGES

        resident = _models_from_analyzer(analyzer)
        seen_models = {SPACY_MODELS.get(lang) for lang in resident}
        missing: list[str] = []
        for lang in SUPPORTED_LANGUAGES:
            model_name = SPACY_MODELS.get(lang)
            if lang in resident or model_name in seen_models:
                continue  # model already covered (e.g. lb shares de's model)
            seen_models.add(model_name)
            missing.append(lang)
        _tier_b2_embedding(
            df, resident, threshold, policies, sequential_languages=missing,
        )

    _tier_c_fallback(df, policies)

    return policies


def classify_pii_columns_multi_pass(
    samples: list[pl.DataFrame],
    *,
    analyzer: Any,
    similarity_threshold: float | None = None,
    structured_enabled: bool | None = None,
) -> list[dict[str, ColumnPolicy]]:
    """Language-major classification across MANY tables at once.

    Memory contract: at most ONE spaCy model is resident at any moment.

    Pass structure::

        EN pass   — the (English-only) analyzer runs Tier A1 + B1 + B2(en)
                    for every sample, then the caller releases the engine.
        FR pass   — the French model loads once, scores every sample's
                    unclassified column names, and is released.
        DE pass   — same (Luxembourgish shares the German model).
        Commit    — per table: best B2 score across all passes is applied,
                    then the Tier C fallback runs (no model needed).

    ``samples`` are small per-table head frames (≤500 rows) — classification
    never needs the full table.  Returns one policy dict per input sample,
    in order.  Per-table tier failures are non-fatal (empty policies for
    that table), matching the single-table classifier's behaviour in the
    pipeline.
    """
    from .anonymization import SPACY_MODELS, SUPPORTED_LANGUAGES

    if structured_enabled is None:
        structured_enabled = _presidio_structured_enabled()
    threshold = (
        similarity_threshold if similarity_threshold is not None
        else _column_similarity_threshold()
    )

    policies_per_table: list[dict[str, ColumnPolicy]] = [{} for _ in samples]
    b2_best_per_table: list[dict[str, tuple[str | None, float]]] = [{} for _ in samples]

    def _candidates(df: pl.DataFrame, policies: dict[str, ColumnPolicy]) -> list[str]:
        return [c for c in df.columns if c not in policies and _is_text_column(df.schema[c])]

    # ── EN pass: A1 + B1 (analyzer) + B2 with the analyzer's own EN model ──
    resident = _models_from_analyzer(analyzer)
    for i, df in enumerate(samples):
        try:
            _tier_a1_name_pattern(df, policies_per_table[i])
            _tier_b1_presidio_structured(df, analyzer, policies_per_table[i], structured_enabled)
            for lang, nlp in resident.items():
                _score_columns_for_language(
                    _candidates(df, policies_per_table[i]), lang, nlp, b2_best_per_table[i],
                )
        except Exception:  # pragma: no cover — defensive, mirrors pipeline's non-fatal handling
            continue
    covered_models = {SPACY_MODELS.get(lang) for lang in resident}

    # ── Remaining language passes: one model resident at a time ───────────
    for lang in SUPPORTED_LANGUAGES:
        model_name = SPACY_MODELS.get(lang)
        if lang in resident or model_name in covered_models:
            continue
        covered_models.add(model_name)
        nlp = _sequential_model_for(lang)
        if nlp is None:
            continue
        for i, df in enumerate(samples):
            _score_columns_for_language(
                _candidates(df, policies_per_table[i]), lang, nlp, b2_best_per_table[i],
            )
    release_sequential_model()

    # ── Commit B2 winners, then the model-free Tier C fallback ────────────
    for i, df in enumerate(samples):
        for column, (entity, score) in b2_best_per_table[i].items():
            if entity is None or score < threshold or column in policies_per_table[i]:
                continue
            policies_per_table[i][column] = _make_policy(
                column, entity, source="embedding_similarity", score=score,
            )
        _tier_c_fallback(df, policies_per_table[i])

    return policies_per_table


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — apply policies.
# ─────────────────────────────────────────────────────────────────────────────


REDACTED_SENTINEL = "[REDACTED]"


def apply_column_policies(
    df: pl.DataFrame,
    policies: dict[str, ColumnPolicy],
    *,
    registry: Any | None = None,
    pseudonymizer: Any | None = None,
    inplace: bool = False,
) -> tuple[pl.DataFrame, dict]:
    """Apply each `ColumnPolicy`'s action to every non-null cell in its column.

    ``ACTION_HASH``      → ``pseudonymizer(value)``     (deterministic, joinable)
    ``ACTION_TOKENIZE``  → ``registry.token_for(entity_type, value)``
    ``ACTION_REDACT``    → ``REDACTED_SENTINEL``
    ``ACTION_BIN``       → no-op (deferred to existing binning layers)
    ``ACTION_SCAN``      → no-op (deferred to row-by-row Presidio scan)

    Returns a new DataFrame plus a stats dict::

        {
          "columns_processed": [list of column names actually mutated],
          "actions_applied": {col: action},
          "entity_types": {col: entity_type},
          "values_masked": {col: int},
          "skipped_columns": {col: reason},
        }

    The stats dict is intended for the audit log so an operator can verify
    which classifier tier acted on which column.

    Polars frames are persistent structures — ``with_columns`` returns a new
    frame that shares the unchanged column buffers, so no defensive copy is
    needed.  ``inplace`` is accepted for caller compatibility and ignored.
    """
    del inplace  # Polars frames never mutate the caller's frame
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

        df, mutated = _apply_one_policy(df, column, policy, registry, pseudonymizer)
        if mutated == 0:
            stats["skipped_columns"][column] = "no_non_null_values"
            continue

        stats["columns_processed"].append(column)
        stats["actions_applied"][column] = policy.action
        stats["entity_types"][column] = policy.entity_type
        stats["values_masked"][column] = mutated

    return df, stats


def _apply_one_policy(
    df: pl.DataFrame,
    column: str,
    policy: ColumnPolicy,
    registry: Any | None,
    pseudonymizer: Any | None,
) -> tuple[pl.DataFrame, int]:
    """Mask one column according to its policy.  Returns the new frame and the
    number of cells mutated (non-null inputs).  All three actions write string
    tokens, so the column dtype becomes String."""
    series = df[column]
    mutated = len(series) - series.null_count()
    if mutated == 0:
        return df, 0

    if policy.action == ACTION_HASH:
        new = series.map_elements(pseudonymizer, return_dtype=pl.String, skip_nulls=True)
    elif policy.action == ACTION_TOKENIZE:
        new = series.map_elements(
            lambda v, e=policy.entity_type: registry.token_for(e, str(v)),
            return_dtype=pl.String,
            skip_nulls=True,
        )
    elif policy.action == ACTION_REDACT:
        new = series.map_elements(
            lambda _v: REDACTED_SENTINEL, return_dtype=pl.String, skip_nulls=True,
        )
    else:  # pragma: no cover — callers filter BIN/SCAN before this point
        return df, 0

    return df.with_columns(new.alias(column)), mutated


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
