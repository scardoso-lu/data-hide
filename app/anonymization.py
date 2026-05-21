"""GDPR-oriented anonymization and privacy transformations."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from typing import Any, Optional

import pandas as pd

from .classification import _is_text_column

# spaCy models for each supported language.
# Luxembourgish (lb) has no dedicated spaCy model; the German model is the
# closest approximation given the linguistic relationship between the two.
SPACY_MODELS: dict[str, str] = {
    "en": "en_core_web_lg",
    "fr": "fr_core_news_lg",
    "de": "de_core_news_lg",
    "lb": "de_core_news_lg",
}
SUPPORTED_LANGUAGES: list[str] = list(SPACY_MODELS.keys())
GDPR_ENTITIES: list[str] | None = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "CRYPTO",
]
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
]
RESIDUAL_BLOCKING_ENTITIES = {
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "CRYPTO",
}
PSEUDONYM_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]*_\d+\b")


def _detect_language(text: str) -> str:
    """Return a language code from SUPPORTED_LANGUAGES; defaults to 'en' on failure."""
    try:
        from langdetect import detect
        detected = detect(text)
        return detected if detected in SUPPORTED_LANGUAGES else "en"
    except Exception:
        return "en"


def _analyze(text: str, analyzer: Any):
    return analyzer.analyze(text=text, entities=GDPR_ENTITIES, language=_detect_language(text))


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
    return registry


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


def _anonymize_text(text: str, analyzer: Any, registry: EntityRegistry) -> tuple[str, list]:
    findings = _analyze(text, analyzer)
    if not findings:
        return text, []
    result = text
    for r in sorted(findings, key=lambda x: x.start, reverse=True):
        token = registry.token_for(r.entity_type, text[r.start:r.end])
        result = result[:r.start] + token + result[r.end:]
    return result, findings


def _is_pseudonymized_text(text: str) -> bool:
    return bool(PSEUDONYM_TOKEN_RE.fullmatch(text.strip()))


def _finding_overlaps_pseudonym(text: str, finding: Any) -> bool:
    return any(finding.start < match.end() and finding.end > match.start() for match in PSEUDONYM_TOKEN_RE.finditer(text))


def _residual_findings(text: str, analyzer: Any) -> list:
    if _is_pseudonymized_text(text):
        return []
    return [finding for finding in _analyze(text, analyzer) if not _finding_overlaps_pseudonym(text, finding)]


def anonymize_dataframe(
    df: pd.DataFrame,
    analyzer: Any,
    registry: Optional[EntityRegistry] = None,
) -> tuple[pd.DataFrame, dict]:
    if registry is None:
        registry = EntityRegistry()

    df = df.copy()
    text_cols = [c for c in df.columns if _is_text_column(df[c].dtype)]
    entity_counts: dict[str, int] = {}
    cols_hit: list[str] = []
    column_stats: list[dict] = []

    for col in text_cols:
        col_detections = 0
        col_entity_counts: dict[str, int] = {}
        new_values: list = []
        for val in df[col]:
            all_findings: list = []
            if isinstance(val, (dict, list)):
                anon_val, all_findings = _anonymize_json(val, analyzer, registry)
                new_values.append(anon_val)
            elif isinstance(val, str):
                if _looks_like_json(val):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (dict, list)):
                            anon_obj, all_findings = _anonymize_json(parsed, analyzer, registry)
                            new_values.append(json.dumps(anon_obj, ensure_ascii=False))
                        else:
                            raise ValueError("JSON primitive")
                    except (json.JSONDecodeError, ValueError):
                        anon_val, all_findings = _anonymize_text(val, analyzer, registry)
                        new_values.append(anon_val)
                else:
                    anon_val, all_findings = _anonymize_text(val, analyzer, registry)
                    new_values.append(anon_val)
            else:
                new_values.append(val)

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


def enforce_k_anonymity(df: pd.DataFrame, quasi_cols: list[str], k: int) -> tuple[pd.DataFrame, dict]:
    if not quasi_cols:
        return df, {"suppressed_rows": 0, "k": k}
    present = [c for c in quasi_cols if c in df.columns]
    if not present:
        return df, {"suppressed_rows": 0, "k": k}
    group_sizes = df.groupby(present, dropna=False)[present[0]].transform("count")
    filtered = df[group_sizes >= k].reset_index(drop=True)
    return filtered, {"suppressed_rows": len(df) - len(filtered), "k": k}


def _hash_value(val: object, salt_bytes: bytes) -> object:
    try:
        if pd.isna(val):
            return val
    except (TypeError, ValueError):
        pass
    raw = val if isinstance(val, str) else str(val)
    return hashlib.sha256(salt_bytes + raw.encode("utf-8")).hexdigest()[:24]


def hash_identifier_columns(df: pd.DataFrame, id_cols: list[str], salt: str = "") -> tuple[pd.DataFrame, list[str]]:
    if not id_cols:
        return df, []
    df = df.copy()
    hashed: list[str] = []
    salt_bytes = salt.encode("utf-8")
    for col in id_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].map(lambda v, _sb=salt_bytes: _hash_value(v, _sb))
        hashed.append(col)
    return df, hashed


def _anonymize_json(obj: object, analyzer: Any, registry: EntityRegistry) -> tuple[object, list]:
    if isinstance(obj, dict):
        result: dict = {}
        all_findings: list = []
        for k, v in obj.items():
            anon_v, f = _anonymize_json(v, analyzer, registry)
            result[k] = anon_v
            all_findings.extend(f)
        return result, all_findings
    if isinstance(obj, list):
        result_list: list = []
        all_findings = []
        for item in obj:
            anon_item, f = _anonymize_json(item, analyzer, registry)
            result_list.append(anon_item)
            all_findings.extend(f)
        return result_list, all_findings
    if isinstance(obj, str):
        return _anonymize_text(obj, analyzer, registry)
    return obj, []


def _scan_json_for_pii(obj: object, analyzer: Any) -> int:
    return sum(item["count"] for item in _scan_json_for_residuals(obj, analyzer, column="", path="$"))


def _scan_json_for_residuals(obj: object, analyzer: Any, column: str, path: str) -> list[dict]:
    if isinstance(obj, dict):
        results: list[dict] = []
        for key, value in obj.items():
            results.extend(_scan_json_for_residuals(value, analyzer, column, f"{path}.{key}"))
        return results
    if isinstance(obj, list):
        results = []
        for index, item in enumerate(obj):
            results.extend(_scan_json_for_residuals(item, analyzer, column, f"{path}[{index}]"))
        return results
    if isinstance(obj, str):
        return _summarize_findings(column, path, obj, _residual_findings(obj, analyzer))
    return []


def _looks_like_json(s: str) -> bool:
    ch = s.lstrip()
    return bool(ch) and ch[0] in ("{", "[")


def _summarize_findings(column: str, path: str | None, text: str, findings: list) -> list[dict]:
    counts: dict[str, int] = {}
    for finding in findings:
        if not _is_blocking_residual(text, finding):
            continue
        counts[finding.entity_type] = counts.get(finding.entity_type, 0) + 1
    return [
        {"column": column, "path": path, "entity_type": entity_type, "count": count}
        for entity_type, count in counts.items()
    ]


def _is_plausible_phone_residual(text: str, finding: Any) -> bool:
    value = text[finding.start:finding.end]
    digits = re.sub(r"\D", "", value)
    return len(digits) >= 7 and not re.search(r"[A-Za-z]", value)


def _is_blocking_residual(text: str, finding: Any) -> bool:
    if finding.entity_type not in RESIDUAL_BLOCKING_ENTITIES:
        return False
    if finding.entity_type == "PHONE_NUMBER":
        return _is_plausible_phone_residual(text, finding)
    return True


def residual_pii_findings(df: pd.DataFrame, analyzer: Any) -> list[dict]:
    findings: list[dict] = []
    for col in df.columns:
        if not _is_text_column(df[col].dtype):
            continue
        for val in df[col]:
            if isinstance(val, (dict, list)):
                findings.extend(_scan_json_for_residuals(val, analyzer, column=col, path="$"))
            elif isinstance(val, str):
                if _looks_like_json(val):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (dict, list)):
                            findings.extend(_scan_json_for_residuals(parsed, analyzer, column=col, path="$"))
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                findings.extend(_summarize_findings(col, None, val, _residual_findings(val, analyzer)))
    return findings


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


def anonymize_gps_columns(df: pd.DataFrame, gps_cols: list[str], precision: int = 2) -> tuple[pd.DataFrame, list[str]]:
    """Reduce spatial precision of GPS columns by rounding to `precision` decimal places.

    Numeric columns (lat/lon floats) are rounded directly.
    String columns containing WKT POINT geometries have their embedded
    coordinate values rounded.  Null values are preserved unchanged.
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


def validate_residual_pii(df: pd.DataFrame, analyzer: Any) -> int:
    residuals = _merge_residual_summaries(residual_pii_findings(df, analyzer))
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
