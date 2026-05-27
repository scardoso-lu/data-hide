import sys
import types
from types import SimpleNamespace

import pandas as pd
import pytest

from app.anonymization import (
    EntityRegistry,
    SPACY_LABELS_TO_IGNORE,
    SPACY_TO_PRESIDIO_ENTITY_MAPPING,
    SUPPORTED_LANGUAGES,
    _resolve_overlapping_findings,
    anonymize_dataframe,
    build_engines,
    validate_residual_pii,
)


def _f(entity_type: str, start: int, end: int, score: float = 1.0):
    return SimpleNamespace(entity_type=entity_type, start=start, end=end, score=score)


class TestResolveOverlappingFindings:
    """Regression coverage for the Presidio EMAIL_ADDRESS / URL overlap.

    Presidio's recognizers fire independently, so the same character range
    can be returned twice with different entity types.  Without resolution
    the inline-token substitution in ``_anonymize_text`` corrupts the output
    (``URL_1ADDRESS_0`` for ``bob.smith@company.com``) and the returned
    finding count is inflated.
    """

    def test_email_swallows_subspan_url(self):
        # The shape that broke ``bob.smith@company.com``:
        # EMAIL_ADDRESS covers the whole string; URL is a subspan starting
        # at the same offset with a lower score.  Without resolution both
        # would survive and the second substitution would corrupt the first.
        findings = [_f("EMAIL_ADDRESS", 0, 21, 1.0), _f("URL", 0, 6, 0.5)]
        kept = _resolve_overlapping_findings(findings)
        assert [(k.entity_type, k.start, k.end) for k in kept] == [("EMAIL_ADDRESS", 0, 21)]

    def test_non_overlapping_findings_all_kept(self):
        findings = [_f("PERSON", 0, 5, 0.9), _f("EMAIL_ADDRESS", 10, 25, 1.0)]
        kept = _resolve_overlapping_findings(findings)
        assert len(kept) == 2

    def test_adjacent_findings_are_not_overlapping(self):
        # Touching at the boundary (end == next.start) must not be treated
        # as overlap — both spans contribute distinct tokens.
        findings = [_f("PERSON", 0, 10, 0.9), _f("EMAIL_ADDRESS", 10, 25, 1.0)]
        kept = _resolve_overlapping_findings(findings)
        assert len(kept) == 2

    def test_broader_span_wins_when_same_start(self):
        findings = [_f("URL", 0, 6, 0.5), _f("EMAIL_ADDRESS", 0, 21, 1.0)]
        kept = _resolve_overlapping_findings(findings)
        assert kept[0].entity_type == "EMAIL_ADDRESS"

    def test_higher_score_wins_when_spans_equal(self):
        findings = [_f("URL", 0, 21, 0.5), _f("EMAIL_ADDRESS", 0, 21, 1.0)]
        kept = _resolve_overlapping_findings(findings)
        assert kept[0].entity_type == "EMAIL_ADDRESS"

    def test_empty_input_returns_empty(self):
        assert _resolve_overlapping_findings([]) == []


def test_anonymize_dataframe_caches_repeated_text_values():
    class FakeAnalyzer:
        def __init__(self):
            self.calls = 0

        def analyze(self, text, entities=None, language=None, score_threshold=None):
            self.calls += 1
            start = text.find("Alice")
            if start < 0:
                return []
            return [_f("PERSON", start, start + len("Alice"))]

    analyzer = FakeAnalyzer()
    df = pd.DataFrame({"name": ["Alice", "Alice", "Bob"]})

    result, stats = anonymize_dataframe(df, analyzer, EntityRegistry())

    assert analyzer.calls == 2
    assert list(result["name"]) == ["PERSON_0", "PERSON_0", "Bob"]
    assert stats["entity_counts"] == {"PERSON": 2}


def test_residual_validation_catches_deterministic_email():
    df = pd.DataFrame({"email": ["alice@example.com"]})

    with pytest.raises(RuntimeError, match="email.EMAIL_ADDRESS=1"):
        validate_residual_pii(df)


def test_residual_validation_ignores_structured_metadata_false_positives():
    df = pd.DataFrame({
        "dataset_last_update": ["2026-05-23T14:45:20"],
        "record_key": ["LU280019400644750000"],
        "resource_last_modified": ["2026-05-23 14:45:20"],
        "source_file": ["annonces_20260523144520.csv"],
    })

    assert validate_residual_pii(df) == 0


def test_residual_validation_ignores_numeric_measure_false_positives():
    df = pd.DataFrame({
        "announced_price_eur_current_raw": ["4111111111111111", "1640000"],
        "announced_price_m2_eur_current_raw": ["4999123", "5500000000000004"],
    })

    assert validate_residual_pii(df) == 0


def test_residual_validation_still_blocks_explicit_structured_pii_columns():
    df = pd.DataFrame({
        "phone": ["+352 621 123 456"],
        "credit_card": ["4111111111111111"],
        "iban": ["GB29NWBK60161331926819"],
    })

    with pytest.raises(RuntimeError) as exc_info:
        validate_residual_pii(df)

    message = str(exc_info.value)
    assert "phone.PHONE_NUMBER=1" in message
    assert "credit_card.CREDIT_CARD=1" in message
    assert "iban.IBAN_CODE=1" in message


def test_build_engines_ignores_unmapped_cardinal_spacy_label(monkeypatch):
    captured = {}

    class FakeNlpEngineProvider:
        def __init__(self, nlp_configuration):
            captured["config"] = nlp_configuration

        def create_engine(self):
            return object()

    class FakeRecognizerConfigurationLoader:
        @staticmethod
        def get():
            return {
                "supported_languages": ["en"],
                "global_regex_flags": 26,
                "recognizers": [
                    {
                        "name": "CreditCardRecognizer",
                        "type": "predefined",
                        "supported_languages": [
                            {"language": "en", "context": ["credit", "card"]},
                            {"language": "es", "context": ["tarjeta"]},
                            {"language": "it"},
                            {"language": "pl"},
                        ],
                    },
                    {
                        "name": "EsNifRecognizer",
                        "type": "predefined",
                        "supported_languages": ["es"],
                    },
                ],
            }

    class FakeRecognizerListLoader:
        @staticmethod
        def get(recognizers, supported_languages, global_regex_flags):
            captured["recognizers"] = recognizers
            captured["supported_languages"] = supported_languages
            captured["global_regex_flags"] = global_regex_flags
            return ["credit-card-en"]

    class FakeRecognizerRegistry:
        def __init__(self, recognizers, supported_languages, global_regex_flags):
            self.recognizers = recognizers
            self.supported_languages = supported_languages
            self.global_regex_flags = global_regex_flags

        def add_nlp_recognizer(self, nlp_engine):
            captured["nlp_recognizer_added"] = nlp_engine

    class FakeAnalyzerEngine:
        def __init__(self, registry, nlp_engine, supported_languages):
            self.registry = registry
            self.nlp_engine = nlp_engine
            self.supported_languages = supported_languages

    fake_presidio = types.ModuleType("presidio_analyzer")
    fake_presidio.AnalyzerEngine = FakeAnalyzerEngine
    fake_presidio.RecognizerRegistry = FakeRecognizerRegistry
    fake_nlp_engine = types.ModuleType("presidio_analyzer.nlp_engine")
    fake_nlp_engine.NlpEngineProvider = FakeNlpEngineProvider
    fake_registry_utils = types.ModuleType("presidio_analyzer.recognizer_registry.recognizers_loader_utils")
    fake_registry_utils.RecognizerConfigurationLoader = FakeRecognizerConfigurationLoader
    fake_registry_utils.RecognizerListLoader = FakeRecognizerListLoader

    monkeypatch.setitem(sys.modules, "presidio_analyzer", fake_presidio)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", fake_nlp_engine)
    monkeypatch.setitem(
        sys.modules,
        "presidio_analyzer.recognizer_registry.recognizers_loader_utils",
        fake_registry_utils,
    )

    build_engines()

    ner_config = captured["config"]["ner_model_configuration"]
    assert ner_config["model_to_presidio_entity_mapping"] == SPACY_TO_PRESIDIO_ENTITY_MAPPING
    assert ner_config["labels_to_ignore"] == SPACY_LABELS_TO_IGNORE
    assert SPACY_TO_PRESIDIO_ENTITY_MAPPING["GPE"] == "LOCATION"
    assert SPACY_TO_PRESIDIO_ENTITY_MAPPING["PERSON"] == "PERSON"
    assert "CARDINAL" in SPACY_LABELS_TO_IGNORE
    assert captured["supported_languages"] == SUPPORTED_LANGUAGES
    assert captured["recognizers"] == [
        {
            "name": "CreditCardRecognizer",
            "type": "predefined",
            "supported_languages": [{"language": "en", "context": ["credit", "card"]}],
        }
    ]
    assert captured["nlp_recognizer_added"] is not None


class FakeAnalyzerPersonOnly:
    """Detects PERSON in any value that contains the literal word 'Alice'."""

    def analyze(self, text, entities=None, language=None, score_threshold=None):
        start = text.find("Alice")
        if start < 0:
            return []
        return [SimpleNamespace(entity_type="PERSON", start=start, end=start + len("Alice"), score=1.0)]


class TestAnonymizeDataframeScanColumns:
    """anonymize_dataframe must respect the scan_columns allow-list."""

    def test_only_listed_columns_are_scanned(self):
        analyzer = FakeAnalyzerPersonOnly()
        df = pd.DataFrame({
            "notes": ["Alice went to Paris"],
            "status": ["Alice"],
        })
        result, stats = anonymize_dataframe(df, analyzer, EntityRegistry(), scan_columns=["notes"])

        assert "PERSON_0" in result["notes"].iloc[0]
        assert result["status"].iloc[0] == "Alice"
        assert stats["text_columns_scanned"] == ["notes"]

    def test_none_scan_columns_scans_all_text_columns(self):
        analyzer = FakeAnalyzerPersonOnly()
        df = pd.DataFrame({
            "notes": ["Alice went to Paris"],
            "status": ["Alice"],
        })
        result, stats = anonymize_dataframe(df, analyzer, EntityRegistry(), scan_columns=None)

        assert "PERSON_0" in result["notes"].iloc[0]
        assert "PERSON_0" in result["status"].iloc[0]
        assert set(stats["text_columns_scanned"]) == {"notes", "status"}

    def test_empty_scan_columns_skips_all(self):
        analyzer = FakeAnalyzerPersonOnly()
        df = pd.DataFrame({"notes": ["Alice went to Paris"]})
        result, stats = anonymize_dataframe(df, analyzer, EntityRegistry(), scan_columns=[])

        assert result["notes"].iloc[0] == "Alice went to Paris"
        assert stats["text_columns_scanned"] == []

    def test_nonexistent_scan_column_is_ignored(self):
        analyzer = FakeAnalyzerPersonOnly()
        df = pd.DataFrame({"notes": ["Alice went to Paris"]})
        result, stats = anonymize_dataframe(
            df, analyzer, EntityRegistry(), scan_columns=["notes", "nonexistent"]
        )
        assert "PERSON_0" in result["notes"].iloc[0]
        assert stats["text_columns_scanned"] == ["notes"]
