"""GDPR-oriented anonymization and privacy transformations."""

from __future__ import annotations

import copy
import os
from decimal import Decimal
from functools import lru_cache
import ipaddress
import json
import logging
import re
from typing import Any, Callable

import pandas as pd

from .classification import _is_text_column

# spaCy models for each supported language.
# Luxembourgish (lb) has no dedicated spaCy model; the German model is the
# closest approximation given the linguistic relationship between the two.
# Each language's model can be overridden at deployment time via
# `SPACY_MODEL_<LANG>` (e.g. `SPACY_MODEL_EN=en_core_web_sm` for low-resource
# environments).  The defaults preserve the historical behaviour the test
# suite was built against.
_SPACY_MODEL_DEFAULTS: dict[str, str] = {
    "en": "en_core_web_lg",
    "fr": "fr_core_news_lg",
    "de": "de_core_news_lg",
    "lb": "de_core_news_lg",
}
SPACY_MODELS: dict[str, str] = {
    lang: os.environ.get(f"SPACY_MODEL_{lang.upper()}", default)
    for lang, default in _SPACY_MODEL_DEFAULTS.items()
}
SUPPORTED_LANGUAGES: list[str] = list(SPACY_MODELS.keys())
# Entity groups — named bundles that deployments can compose into the
# active entity set.  Each group is informational *and* operational: a
# policy file or env var can opt-out a whole group (e.g. drop `_US_NATIONAL`
# in EU-only deployments).  The default `GDPR_ENTITIES` below is the union
# of all groups and preserves the historical, full-coverage behaviour.

_DIRECT_IDENTIFIERS: list[str] = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "URL",
]
_FINANCIAL: list[str] = [
    "CREDIT_CARD", "IBAN_CODE", "CRYPTO", "SWIFT_BIC", "SALARY",
    "INVOICE_NUMBER", "INSURANCE_POLICY",
]
_US_NATIONAL: list[str] = [
    "US_SSN", "US_DRIVER_LICENSE", "US_PASSPORT", "US_ITIN", "US_BANK_NUMBER",
]
_EU_NATIONAL: list[str] = [
    "LU_CCSS", "LU_PASSPORT", "EU_VAT", "NATIONAL_TAX_ID",
]
_MEDICAL: list[str] = [
    "MEDICAL_LICENSE", "MEDICAL_RECORD", "HEALTH_CONDITION", "HEALTH_INSURANCE",
]
_LOCATION_QID: list[str] = [
    "LOCATION", "STREET_ADDRESS", "POSTAL_CODE",
]
_DATE_QID: list[str] = [
    "DATE_OF_BIRTH",
]
_ART9_SPECIAL: list[str] = [
    "NRP", "RELIGION", "ETHNICITY", "SEXUAL_ORIENTATION", "TRADE_UNION",
]
_ART10: list[str] = [
    "CRIMINAL_RECORD",
]
_B2B_REFERENCES: list[str] = [
    "CONTRACT_NUMBER", "BOOKING_REF", "CUSTOMER_EMPLOYEE_ID",
    "VEHICLE_PLATE", "COURT_CASE",
]

ENTITY_GROUPS: dict[str, list[str]] = {
    "direct_identifiers": _DIRECT_IDENTIFIERS,
    "financial": _FINANCIAL,
    "us_national": _US_NATIONAL,
    "eu_national": _EU_NATIONAL,
    "medical": _MEDICAL,
    "location_qid": _LOCATION_QID,
    "date_qid": _DATE_QID,
    "art9_special": _ART9_SPECIAL,
    "art10": _ART10,
    "b2b_references": _B2B_REFERENCES,
}


def _compose_gdpr_entities(groups: dict[str, list[str]] | None = None) -> list[str]:
    """Flatten a dict of entity groups into the ordered, de-duplicated entity list
    that Presidio's `analyze(entities=...)` parameter expects.
    """
    seen: dict[str, None] = {}
    for name in (groups or ENTITY_GROUPS):
        for entity in (groups or ENTITY_GROUPS)[name]:
            seen.setdefault(entity, None)
    return list(seen)


# Region toggle.  `ANONYMIZATION_REGIONS` is a comma-separated list of region
# codes; only their associated national-ID groups are kept active.  Setting
# it to "all" (the default) keeps every region enabled — matching the
# historical behaviour the locked test suite expects.
#
# Examples:
#   ANONYMIZATION_REGIONS=eu        — keep EU national IDs, drop US national IDs
#   ANONYMIZATION_REGIONS=eu,us     — keep both
#   ANONYMIZATION_REGIONS=all       — every region (default)
_REGION_TO_GROUP = {
    "us": "us_national",
    "eu": "eu_national",
}


def _active_groups() -> dict[str, list[str]]:
    """Return a filtered copy of ENTITY_GROUPS based on `ANONYMIZATION_REGIONS`."""
    raw = os.environ.get("ANONYMIZATION_REGIONS", "all").strip().lower()
    if raw == "all" or not raw:
        return ENTITY_GROUPS
    enabled_regions = {token.strip() for token in raw.split(",") if token.strip()}
    region_groups = set(_REGION_TO_GROUP.values())
    kept_region_groups = {_REGION_TO_GROUP[r] for r in enabled_regions if r in _REGION_TO_GROUP}
    return {
        name: members
        for name, members in ENTITY_GROUPS.items()
        if name not in region_groups or name in kept_region_groups
    }


GDPR_ENTITIES: list[str] | None = _compose_gdpr_entities(_active_groups())

# Per-entity score thresholds.  Replaces the previous single PRESIDIO_SCORE_THRESHOLD.
# DEFAULT_SCORE_THRESHOLD applies to any entity not explicitly listed; the
# overrides below keep high-precision recognizers (Luhn-validated cards,
# checksum-validated IBANs, regex-strict emails/URLs) catching even their
# low-score variants, while leaving low-precision recognizers at the default
# 0.4 so they remain context-dependent.
DEFAULT_SCORE_THRESHOLD: float = 0.4
PRESIDIO_SCORE_THRESHOLDS: dict[str, float] = {
    # High-precision: deterministic regex + checksum / structural validation.
    "CREDIT_CARD": 0.0,
    "IBAN_CODE": 0.0,
    "EMAIL_ADDRESS": 0.0,
    "URL": 0.0,
    "CRYPTO": 0.0,
    # SSN with dashes/spaces scores 0.5; allow 0.3 so context-only matches
    # (e.g. "ITIN 912-34-5678") still pass via Presidio's US_SSN.
    "US_SSN": 0.3,
}

# Backwards-compatible alias retained for any external callers.
PRESIDIO_SCORE_THRESHOLD: float = DEFAULT_SCORE_THRESHOLD


def _score_threshold_for(entity_type: str) -> float:
    return PRESIDIO_SCORE_THRESHOLDS.get(entity_type, DEFAULT_SCORE_THRESHOLD)
logger = logging.getLogger(__name__)
SPACY_TO_PRESIDIO_ENTITY_MAPPING = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    "NORP": "NRP",
}
SPACY_LABELS_TO_IGNORE = [
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "PERCENT",
    "MONEY",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    # German/French spaCy models emit a generic "MISC" label that has no
    # Presidio mapping; without this entry Presidio logs a warning per
    # occurrence ("Entity MISC is not mapped to a Presidio entity").
    "MISC",
]
# Column-name token policy lives in its own module so deployments can
# extend or replace it without touching the recognizers.  The names are
# re-exported here for backwards compatibility with existing callers / tests.
from .column_policy import (  # noqa: E402
    CREDIT_CARD_COLUMN_TOKENS,
    IBAN_COLUMN_TOKENS,
    PHONE_COLUMN_TOKENS,
    RESIDUAL_METADATA_COLUMN_TOKENS,
)
PSEUDONYM_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]*_\d+\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>'\"]+", re.IGNORECASE)
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.IGNORECASE)
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{6,}\d)(?!\w)")
IP_CANDIDATE_RE = re.compile(r"\b(?:[0-9A-Fa-f:.]{3,})\b")
STRUCTURED_SCALAR_RE = re.compile(r"^[A-Za-z0-9_.:/\\-]+$")
SAFE_JSON_PATH_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


@lru_cache(maxsize=50000)
def _detect_language(text: str) -> str:
    """Return a language code from SUPPORTED_LANGUAGES; defaults to 'en' on failure."""
    try:
        from langdetect import detect
        detected = detect(text)
        return detected if detected in SUPPORTED_LANGUAGES else "en"
    except Exception:
        return "en"


_COLUMN_LANGUAGE_HOMOGENEITY_THRESHOLD = 0.8


def _detect_column_language(series: pd.Series, n_samples: int = 20) -> str | None:
    """Sample up to *n_samples* non-null strings from *series* and return the
    dominant language if ≥ 80 % of the sample agrees, otherwise ``None``.

    ``None`` signals that the column is multilingual (e.g. a Luxembourg comment
    column mixing en/fr/de/lb) and the caller should fall back to per-value
    detection so no row's language is misidentified.

    A homogeneous column (all English, all French, …) detects its language
    in O(n_samples) calls instead of O(unique_values) calls, eliminating the
    dominant langdetect cost for large free-text columns.
    """
    strings = [v for v in series if isinstance(v, str) and v.strip()]
    if not strings:
        return "en"
    step = max(1, len(strings) // n_samples)
    sample = strings[::step][:n_samples]
    counts: dict[str, int] = {}
    for s in sample:
        lang = _detect_language(s)
        counts[lang] = counts.get(lang, 0) + 1
    top_lang = max(counts, key=counts.__getitem__)
    if counts[top_lang] / len(sample) >= _COLUMN_LANGUAGE_HOMOGENEITY_THRESHOLD:
        return top_lang
    return None  # mixed-language column — caller will detect per value


def _analyze(text: str, analyzer: Any, language: str | None = None) -> list:
    # Pass the minimum threshold across all entity-specific thresholds to
    # Presidio so it can short-circuit obvious non-matches, then apply the
    # per-entity threshold ourselves.  This keeps high-precision entities
    # (CREDIT_CARD, IBAN_CODE) unfiltered while leaving the rest constrained.
    min_threshold = min(
        [DEFAULT_SCORE_THRESHOLD, *PRESIDIO_SCORE_THRESHOLDS.values()],
        default=DEFAULT_SCORE_THRESHOLD,
    )
    # When language is None (direct callers, e.g. tests), detect per-value.
    # When called from anonymize_dataframe it is pre-resolved at column level,
    # so detection is skipped for all values in that column.
    if language is None:
        language = _detect_language(text)
    findings = analyzer.analyze(
        text=text,
        entities=_entities_supported_in(analyzer, language),
        language=language,
        score_threshold=min_threshold,
    )
    return [f for f in findings if f.score >= _score_threshold_for(f.entity_type)]


def _entities_supported_in(analyzer: Any, language: str) -> list[str] | None:
    """Intersect `GDPR_ENTITIES` with the entity types this analyzer actually
    supports for ``language``.

    Without this filter Presidio logs one WARNING per missing entity per
    `analyze()` call (e.g. "Entity US_SSN doesn't have the corresponding
    recognizer in language : de") because Presidio's US-specific built-ins
    only register for English.  The pipeline already skipped those entities
    in non-English text — this just stops asking for them, eliminating the
    log spam without changing detection behaviour.

    The supported-entity set is cached per analyzer instance via a private
    attribute, so the introspection cost is paid once per language.
    """
    if GDPR_ENTITIES is None or not hasattr(analyzer, "get_supported_entities"):
        return GDPR_ENTITIES
    cache = getattr(analyzer, "_data_hide_supported_entities", None)
    if cache is None:
        cache = {}
        for lang in SUPPORTED_LANGUAGES:
            try:
                cache[lang] = frozenset(analyzer.get_supported_entities(language=lang))
            except Exception:
                cache[lang] = None
        try:
            analyzer._data_hide_supported_entities = cache
        except (AttributeError, TypeError):
            pass  # immutable analyzer mock — fall through to unfiltered path
    supported = cache.get(language)
    if supported is None:
        return GDPR_ENTITIES
    return [entity for entity in GDPR_ENTITIES if entity in supported]


@lru_cache(maxsize=1)
def build_engines() -> Any:
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.recognizer_registry.recognizers_loader_utils import (
        RecognizerConfigurationLoader,
        RecognizerListLoader,
    )

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": lang, "model_name": model} for lang, model in SPACY_MODELS.items()],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": SPACY_TO_PRESIDIO_ENTITY_MAPPING,
            "labels_to_ignore": SPACY_LABELS_TO_IGNORE,
        },
    })
    nlp_engine = provider.create_engine()
    registry = _build_recognizer_registry(nlp_engine, RecognizerConfigurationLoader, RecognizerListLoader, RecognizerRegistry)
    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=SUPPORTED_LANGUAGES)


def _build_recognizer_registry(
    nlp_engine: Any,
    configuration_loader: Any,
    list_loader: Any,
    registry_cls: Any,
) -> Any:
    config = _filter_recognizer_config(configuration_loader.get(), SUPPORTED_LANGUAGES)
    recognizers = list_loader.get(
        config["recognizers"],
        config["supported_languages"],
        config["global_regex_flags"],
    )
    registry = registry_cls(
        recognizers=recognizers,
        supported_languages=config["supported_languages"],
        global_regex_flags=config["global_regex_flags"],
    )
    registry.add_nlp_recognizer(nlp_engine=nlp_engine)
    _install_custom_recognizers(registry, nlp_engine)
    return registry


def _install_custom_recognizers(registry: Any, nlp_engine: Any = None) -> None:
    """Install GDPR-special-category recognizers (LU CCSS, salary, semantic
    Art. 9 / Art. 10 detection, …).

    Imported lazily so the build_engines unit tests that monkeypatch
    presidio_analyzer don't pull in the real PatternRecognizer class.  The
    `nlp_engine` is forwarded so the semantic-concept recognizers can embed
    their seed anchors through the same spaCy models the analyzer uses.
    """
    try:
        from .recognizers import install_custom_recognizers
    except Exception:
        return
    try:
        install_custom_recognizers(registry, nlp_engine)
    except Exception as exc:
        logger.warning("Failed to install custom recognizers: %s", exc)


def _filter_recognizer_config(config: dict, supported_languages: list[str]) -> dict:
    filtered = copy.deepcopy(config)
    filtered["supported_languages"] = supported_languages
    recognizers = []

    for recognizer in filtered.get("recognizers", []):
        languages = recognizer.get("supported_languages") if isinstance(recognizer, dict) else None
        if not languages:
            recognizers.append(recognizer)
            continue

        kept_languages = [
            language
            for language in languages
            if _recognizer_language_code(language) in supported_languages
        ]
        if not kept_languages:
            continue

        recognizer["supported_languages"] = kept_languages
        recognizers.append(recognizer)

    filtered["recognizers"] = recognizers
    return filtered


def _recognizer_language_code(language: object) -> str | None:
    if isinstance(language, str):
        return language
    if isinstance(language, dict):
        value = language.get("language")
        return value if isinstance(value, str) else None
    return None


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


def _resolve_overlapping_findings(findings: list) -> list:
    """Collapse Presidio findings so no two cover the same character range.

    Presidio's recognizers fire independently, so a string like
    ``bob.smith@company.com`` produces both an EMAIL_ADDRESS finding for the
    full span and a URL finding for a substring of it.  Applying both as
    inline token substitutions corrupts the output (``URL_1ADDRESS_0``) and
    inflates the returned finding count.

    Resolution priority (descending):
      1. **Score** — a high-confidence deny-list match (score 1.0) beats a
         lower-confidence spaCy PERSON guess (score 0.85) even when the
         PERSON span is wider.  This matters for multilingual keyword
         categories where Luxembourgish word combinations are routinely
         mis-tagged as PERSON by an English spaCy model.
      2. **Span width** — on score ties, the broader span wins so
         EMAIL_ADDRESS@1.0 absorbs URL@1.0 sub-spans.
      3. **Start position** — earliest match acts as a final, stable
         tiebreaker.
    """
    if not findings:
        return findings
    ordered = sorted(
        findings,
        key=lambda f: (-getattr(f, "score", 0.0), -(f.end - f.start), f.start),
    )
    kept: list = []
    for f in ordered:
        if any(f.start < k.end and k.start < f.end for k in kept):
            continue
        kept.append(f)
    return sorted(kept, key=lambda f: f.start)


def _anonymize_text(text: str, analyzer: Any, registry: EntityRegistry, language: str | None = None) -> tuple[str, list]:
    findings = _resolve_overlapping_findings(_analyze(text, analyzer, language))
    if not findings:
        return text, []
    result = text
    for r in sorted(findings, key=lambda x: x.start, reverse=True):
        token = registry.token_for(r.entity_type, text[r.start:r.end])
        result = result[:r.start] + token + result[r.end:]
    return result, findings


def _is_pseudonymized_text(text: str) -> bool:
    return bool(PSEUDONYM_TOKEN_RE.fullmatch(text.strip()))


def anonymize_dataframe(
    df: pd.DataFrame,
    analyzer: Any,
    registry: EntityRegistry | None = None,
    scan_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    if registry is None:
        registry = EntityRegistry()

    df = df.copy()
    if scan_columns is not None:
        text_cols = [c for c in scan_columns if c in df.columns and _is_text_column(df[c].dtype)]
    else:
        text_cols = [c for c in df.columns if _is_text_column(df[c].dtype)]
    entity_counts: dict[str, int] = {}
    cols_hit: list[str] = []
    column_stats: list[dict] = []
    cache: dict[object, tuple[object, list]] = {}

    for col in text_cols:
        language = _detect_column_language(df[col])
        col_detections = 0
        col_entity_counts: dict[str, int] = {}
        new_values: list = []
        for val in df[col]:
            anon_val, all_findings = _anonymize_value_cached(val, analyzer, registry, cache, language)
            new_values.append(anon_val)

            for f in all_findings:
                col_detections += 1
                entity_counts[f.entity_type] = entity_counts.get(f.entity_type, 0) + 1
                col_entity_counts[f.entity_type] = col_entity_counts.get(f.entity_type, 0) + 1

        df[col] = new_values
        if col_detections:
            cols_hit.append(col)
        column_stats.append({"column": col, "detections": col_detections, "entity_counts": col_entity_counts})

    return df, {
        "text_columns_scanned": text_cols,
        "columns_with_detections": cols_hit,
        "entity_counts": entity_counts,
        "total_entities_detected": sum(entity_counts.values()),
        "column_stats": column_stats,
    }


def _anonymize_value_cached(
    val: object,
    analyzer: Any,
    registry: EntityRegistry,
    cache: dict[object, tuple[object, list]],
    language: str | None = None,
) -> tuple[object, list]:
    raw_key = _cache_key(val)
    key = (raw_key, language) if raw_key is not None else None
    if key is not None and key in cache:
        return cache[key]

    result = _anonymize_value(val, analyzer, registry, language)
    if key is not None:
        cache[key] = result
    return result


def _cache_key(val: object) -> object | None:
    if isinstance(val, str):
        return ("str", val)
    if isinstance(val, (dict, list)):
        try:
            return ("json", json.dumps(val, sort_keys=True, default=str, ensure_ascii=False))
        except (TypeError, ValueError):
            return None
    return None


def _anonymize_value(val: object, analyzer: Any, registry: EntityRegistry, language: str | None = None) -> tuple[object, list]:
    if isinstance(val, (dict, list)):
        return _anonymize_json(val, analyzer, registry, language)
    if isinstance(val, str):
        if _looks_like_json(val):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, (dict, list)):
                    anon_obj, all_findings = _anonymize_json(parsed, analyzer, registry, language)
                    return json.dumps(anon_obj, ensure_ascii=False), all_findings
                raise ValueError("JSON primitive")
            except (json.JSONDecodeError, ValueError):
                pass
        return _anonymize_text(val, analyzer, registry, language)
    return val, []


def enforce_k_anonymity(df: pd.DataFrame, quasi_cols: list[str], k: int) -> tuple[pd.DataFrame, dict]:
    if not quasi_cols:
        return df, {"suppressed_rows": 0, "k": k}
    present = [c for c in quasi_cols if c in df.columns]
    if not present:
        return df, {"suppressed_rows": 0, "k": k}
    group_sizes = df.groupby(present, dropna=False)[present[0]].transform("count")
    filtered = df[group_sizes >= k].reset_index(drop=True)
    return filtered, {"suppressed_rows": len(df) - len(filtered), "k": k}


def pseudonymize_identifier_columns(
    df: pd.DataFrame,
    id_cols: list[str],
    pseudonymizer: Callable[[object], object],
) -> tuple[pd.DataFrame, list[str]]:
    """Replace identifier values with Key Vault-bound pseudonym tokens.

    ``pseudonymizer`` is a callable that maps each non-null value to a
    deterministic token (see ``app.keyvault.KeyVaultPseudonymizer``).  Nulls
    are preserved.  Missing columns are silently skipped so callers can pass
    a superset of column names.
    """
    if not id_cols:
        return df, []
    if pseudonymizer is None:
        raise ValueError(
            "pseudonymizer is required to anonymize identifier columns; "
            "configure KEY_VAULT_URL and KEY_VAULT_RSA_KEY_NAME."
        )
    df = df.copy()
    pseudonymized: list[str] = []
    for col in id_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].map(pseudonymizer)
        pseudonymized.append(col)
    return df, pseudonymized


def _anonymize_json(obj: object, analyzer: Any, registry: EntityRegistry, language: str | None = None) -> tuple[object, list]:
    if isinstance(obj, dict):
        result: dict = {}
        all_findings: list = []
        for k, v in obj.items():
            anon_k, key_findings = _anonymize_text(k, analyzer, registry, language) if isinstance(k, str) else (k, [])
            anon_v, f = _anonymize_json(v, analyzer, registry, language)
            result[anon_k] = anon_v
            all_findings.extend(key_findings)
            all_findings.extend(f)
        return result, all_findings
    if isinstance(obj, list):
        result_list: list = []
        all_findings = []
        for item in obj:
            anon_item, f = _anonymize_json(item, analyzer, registry, language)
            result_list.append(anon_item)
            all_findings.extend(f)
        return result_list, all_findings
    if isinstance(obj, str):
        return _anonymize_text(obj, analyzer, registry, language)
    return obj, []


def _looks_like_json(s: str) -> bool:
    ch = s.lstrip()
    return bool(ch) and ch[0] in ("{", "[")


def residual_pii_findings(df: pd.DataFrame) -> list[dict]:
    findings: list[dict] = []
    for col in df.columns:
        if not _is_text_column(df[col].dtype):
            continue
        for val in df[col]:
            if isinstance(val, (dict, list)):
                findings.extend(_scan_json_for_residuals(val, column=col, path="$"))
            elif isinstance(val, str):
                if _looks_like_json(val):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (dict, list)):
                            findings.extend(_scan_json_for_residuals(parsed, column=col, path="$"))
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                findings.extend(_structured_residuals(col, None, val))
    return findings


def _scan_json_for_residuals(obj: object, column: str, path: str) -> list[dict]:
    if isinstance(obj, dict):
        results: list[dict] = []
        for key, value in obj.items():
            key_path = _json_child_path(path, key)
            if isinstance(key, str):
                results.extend(_structured_residuals(column, key_path, key))
            results.extend(_scan_json_for_residuals(value, column, key_path))
        return results
    if isinstance(obj, list):
        results = []
        for index, item in enumerate(obj):
            results.extend(_scan_json_for_residuals(item, column, f"{path}[{index}]"))
        return results
    if isinstance(obj, str):
        return _structured_residuals(column, path, obj)
    return []


def _json_child_path(path: str, key: object) -> str:
    if isinstance(key, str) and SAFE_JSON_PATH_KEY_RE.fullmatch(key):
        return f"{path}.{key}"
    return f"{path}.<key>"


def _structured_residuals(column: str, path: str | None, text: str) -> list[dict]:
    if _is_pseudonymized_text(text):
        return []
    column_tokens = _column_tokens(column)
    top_level_scalar = path is None
    counts: dict[str, int] = {}
    if not (top_level_scalar and _is_structured_metadata_column_tokens(column_tokens)):
        counts["EMAIL_ADDRESS"] = len(EMAIL_RE.findall(text))
        counts["URL"] = len(URL_RE.findall(text))
        counts["IP_ADDRESS"] = sum(1 for match in IP_CANDIDATE_RE.findall(text) if _is_ip_address(match))
    counts["IBAN_CODE"] = sum(
        1 for match in IBAN_RE.findall(text)
        if _valid_iban(match) and _should_count_structured_residual(column_tokens, text, "IBAN_CODE", top_level_scalar)
    )
    counts["CREDIT_CARD"] = sum(
        1 for match in CREDIT_CARD_RE.findall(text)
        if _valid_luhn(match) and _should_count_structured_residual(column_tokens, text, "CREDIT_CARD", top_level_scalar)
    )
    counts["PHONE_NUMBER"] = sum(
        1 for match in PHONE_RE.findall(text)
        if _is_plausible_phone_text(match) and _should_count_structured_residual(column_tokens, text, "PHONE_NUMBER", top_level_scalar)
    )
    return [
        {"column": column, "path": path, "entity_type": entity_type, "count": count}
        for entity_type, count in counts.items()
        if count
    ]


def _column_tokens(column: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9]+", column.lower()) if token}


def _is_structured_metadata_column(column: str) -> bool:
    return _is_structured_metadata_column_tokens(_column_tokens(column))


def _is_structured_metadata_column_tokens(tokens: set[str]) -> bool:
    return bool(tokens & RESIDUAL_METADATA_COLUMN_TOKENS)


def _column_explicitly_allows_entity(tokens: set[str], entity_type: str) -> bool:
    if entity_type == "PHONE_NUMBER":
        return bool(tokens & PHONE_COLUMN_TOKENS)
    if entity_type == "CREDIT_CARD":
        return bool(tokens & CREDIT_CARD_COLUMN_TOKENS)
    if entity_type == "IBAN_CODE":
        return bool(tokens & IBAN_COLUMN_TOKENS)
    return False


def _is_structured_scalar_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        float(stripped)
        return True
    except ValueError:
        pass
    try:
        if pd.notna(pd.to_datetime(stripped, errors="coerce")):
            return True
    except Exception:
        pass
    return bool(STRUCTURED_SCALAR_RE.fullmatch(stripped))


def _should_count_structured_residual(
    column_tokens: set[str],
    text: str,
    entity_type: str,
    top_level_scalar: bool,
) -> bool:
    if _column_explicitly_allows_entity(column_tokens, entity_type):
        return True
    if top_level_scalar and _is_structured_metadata_column_tokens(column_tokens):
        return False
    return not _is_structured_scalar_text(text)


def _is_plausible_phone_text(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    return len(digits) >= 7 and not re.search(r"[A-Za-z]", text)


def _valid_luhn(value: str) -> bool:
    digits = [int(ch) for ch in re.sub(r"\D", "", value)]
    if len(digits) < 13:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _valid_iban(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    converted = "".join(str(int(ch, 36)) for ch in rearranged)
    try:
        return int(converted) % 97 == 1
    except ValueError:
        return False


def _is_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _merge_residual_summaries(findings: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str | None, str], int] = {}
    for finding in findings:
        key = (finding["column"], finding.get("path"), finding["entity_type"])
        merged[key] = merged.get(key, 0) + finding["count"]
    return [
        {"column": col, "path": path, "entity_type": entity_type, "count": count}
        for (col, path, entity_type), count in sorted(merged.items())
    ]


def _round_wkt(wkt: str, precision: int) -> str:
    """Round all decimal numbers embedded in a WKT geometry string."""
    return re.sub(r"-?\d+\.\d+", lambda m: str(round(float(m.group()), precision)), wkt)


def _is_numeric_like_series(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    return not pd.to_numeric(non_null, errors="coerce").isna().any()


def _round_numeric_like_value(value: object, precision: int) -> object:
    if pd.isna(value):
        return value
    if isinstance(value, Decimal):
        return round(float(value), precision)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return round(float(stripped), precision)
        except ValueError:
            return value
    try:
        return round(value, precision)
    except TypeError:
        return value


def anonymize_gps_columns(df: pd.DataFrame, gps_cols: list[str], precision: int = 2) -> tuple[pd.DataFrame, list[str]]:
    """Reduce spatial precision of GPS columns by rounding to `precision` decimal places.

    Default precision is 2 (about 1 km cells), matching
    GPS_PRECISION in .env.example.  Numeric columns (lat/lon floats) are
    rounded directly.  String columns containing WKT POINT geometries have
    their embedded coordinate values rounded.  Null values are preserved
    unchanged.
    """
    if not gps_cols:
        return df, []
    df = df.copy()
    anonymized: list[str] = []
    for col in gps_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series.dtype):
            df[col] = series.round(precision)
        elif _is_numeric_like_series(series):
            df[col] = series.map(lambda v, p=precision: _round_numeric_like_value(v, p))
        else:
            df[col] = series.map(lambda v, p=precision: _round_wkt(v, p) if isinstance(v, str) else v)
        anonymized.append(col)
    return df, anonymized


def bin_timestamp_columns(df: pd.DataFrame, ts_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Floor timestamp columns to daily granularity (temporal generalisation).

    Sub-day precision combined with rounded GPS coordinates is near-unique per
    person in a city.  Flooring to midnight removes the time-of-day signal
    while preserving the date for analytics.  Null values are preserved.
    Only columns with an actual datetime64 dtype are processed; ambiguous
    string timestamps are left untouched to avoid silent format changes.
    """
    if not ts_cols:
        return df, []
    df = df.copy()
    binned: list[str] = []
    for col in ts_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            df[col] = series.dt.floor("D")
            binned.append(col)
    return df, binned


def _bin_label(lo: float, hi: float) -> str:
    def fmt(v: float) -> str:
        return str(int(v)) if v == int(v) else f"{v:.4g}"
    return f"{fmt(lo)}–{fmt(hi)}"


def bin_numeric_columns(df: pd.DataFrame, cols: list[str], n_bins: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Replace numeric quasi-identifier columns with quantile-range labels.

    Converts e.g. hours_worked=38.5 → "35–40" so k-anonymity groups more rows
    together and suppresses fewer records.  Columns with fewer than two unique
    values are left untouched.  Nulls are preserved as NaN.
    """
    if not cols:
        return df, []
    df = df.copy()
    binned: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        non_null = df[col].dropna()
        if non_null.nunique() < 2:
            continue
        try:
            _, edges = pd.qcut(non_null, q=n_bins, retbins=True, duplicates="drop")
        except ValueError:
            continue
        if len(edges) < 2:
            continue
        labels = [_bin_label(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
        df[col] = pd.cut(df[col], bins=edges, labels=labels, include_lowest=True).astype(object)
        binned.append(col)
        logger.info("Binned numeric column %r into %d quantile ranges", col, len(labels))
    return df, binned


def validate_residual_pii(df: pd.DataFrame) -> int:
    residuals = _merge_residual_summaries(residual_pii_findings(df))
    return _raise_for_residuals(residuals)


def _raise_for_residuals(residuals: list[dict]) -> int:
    total = sum(item["count"] for item in residuals)
    if total:
        detail = ", ".join(
            f"{item['column']}{':' + item['path'] if item.get('path') else ''}.{item['entity_type']}={item['count']}"
            for item in residuals[:10]
        )
        if len(residuals) > 10:
            detail += f", ... {len(residuals) - 10} more"
        raise RuntimeError(
            f"Residual PII detected after anonymization: {total} finding(s): {detail}. "
            "Pipeline aborted; target table was NOT written."
        )
    return 0
