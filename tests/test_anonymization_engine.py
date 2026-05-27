import sys
import types
from types import SimpleNamespace

import pandas as pd
import pytest

from app.classification import (
    ACTION_BIN,
    ACTION_HASH,
    ACTION_SCAN,
    ColumnPolicy,
    FREE_TEXT,
    classify_pii_columns,
    free_text_columns_from_policies,
    _looks_like_free_text,
    _tier_c_fallback,
)
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


class TestLooksLikeFreeText:
    """Boundary conditions for the _looks_like_free_text heuristic."""

    # ── avg_len >= 32 ────────────────────────────────────────────────────────

    def test_avg_len_at_threshold_is_free_text(self):
        assert _looks_like_free_text(["x" * 32]) is True

    def test_avg_len_below_threshold_alone_is_not(self):
        # 31 chars, 1 word, not JSON → all three checks miss
        assert _looks_like_free_text(["x" * 31]) is False

    def test_avg_len_across_multiple_values(self):
        # average is (40 + 24) / 2 = 32 → True
        assert _looks_like_free_text(["x" * 40, "y" * 24]) is True

    # ── avg_words >= 5 ───────────────────────────────────────────────────────

    def test_exactly_five_words_is_free_text(self):
        assert _looks_like_free_text(["one two three four five"]) is True

    def test_four_words_is_not_free_text(self):
        assert _looks_like_free_text(["one two three four"]) is False

    def test_avg_words_across_multiple_values(self):
        # (6 + 4) / 2 = 5.0 → True
        assert _looks_like_free_text(["a b c d e f", "w x y z"]) is True

    # ── jsonish >= 0.5 ───────────────────────────────────────────────────────

    def test_half_json_values_is_free_text(self):
        assert _looks_like_free_text(['{"a": 1}', "plain"]) is True

    def test_array_json_values_is_free_text(self):
        assert _looks_like_free_text(['["x", "y"]', "plain"]) is True

    def test_less_than_half_json_is_not(self):
        # 1 out of 3 → 0.33 < 0.5
        assert _looks_like_free_text(['{"a": 1}', "plain", "also plain"]) is False

    # ── edge cases ───────────────────────────────────────────────────────────

    def test_empty_list_returns_false(self):
        assert _looks_like_free_text([]) is False

    def test_all_non_string_values_returns_false(self):
        assert _looks_like_free_text([None, 42, True, 3.14]) is False

    def test_blank_strings_ignored(self):
        assert _looks_like_free_text(["", "   ", None]) is False

    def test_non_string_values_are_filtered_out(self):
        # only the one real string; it's short and one word → False
        assert _looks_like_free_text([None, 42, "short"]) is False


class TestTierCFallback:
    """_tier_c_fallback gates row-by-row scanning to genuinely free-text columns."""

    class _NullAnalyzer:
        def analyze(self, text, entities=None, language=None, score_threshold=None):
            return []

    def _classify(self, df: pd.DataFrame) -> dict[str, ColumnPolicy]:
        return classify_pii_columns(
            df,
            analyzer=self._NullAnalyzer(),
            similarity_models={},   # skip Tier B2
        )

    # ── ACTION_SCAN paths ────────────────────────────────────────────────────

    def test_long_text_column_gets_action_scan(self):
        long_values = [
            "Customer called to report a problem with their recent order and requested a refund.",
            "Follow-up needed: the delivery was delayed by three days due to a warehouse issue.",
            "Agent noted the account has been flagged for unusual login activity in the past week.",
        ]
        df = pd.DataFrame({"notes": long_values})
        policies = self._classify(df)

        assert "notes" in policies
        assert policies["notes"].action == ACTION_SCAN
        assert "notes" in free_text_columns_from_policies(policies)

    def test_many_words_column_gets_action_scan(self):
        # avg_words >= 5 path
        df = pd.DataFrame({"remarks": ["one two three four five six", "a b c d e f g"]})
        policies = self._classify(df)

        assert "remarks" in policies
        assert policies["remarks"].action == ACTION_SCAN

    def test_json_blob_column_gets_action_scan(self):
        # jsonish >= 0.5 path
        df = pd.DataFrame({"payload": ['{"event": "click", "user": "u1"}', '{"event": "view"}']})
        policies = self._classify(df)

        assert "payload" in policies
        assert policies["payload"].action == ACTION_SCAN

    # ── ACTION_BIN paths ─────────────────────────────────────────────────────

    def test_short_structured_column_gets_action_bin(self):
        df = pd.DataFrame({"status": ["Active", "Inactive", "Pending"]})
        policies = self._classify(df)

        assert "status" in policies
        assert policies["status"].action == ACTION_BIN
        assert "status" not in free_text_columns_from_policies(policies)

    def test_single_word_enum_column_gets_action_bin(self):
        df = pd.DataFrame({"type": ["A", "B", "C", "D"]})
        policies = self._classify(df)

        assert "type" in policies
        assert policies["type"].action == ACTION_BIN

    # ── Metadata ─────────────────────────────────────────────────────────────

    def test_fallback_policy_source_is_fallback(self):
        df = pd.DataFrame({"notes": ["short"]})
        policies = self._classify(df)

        assert policies["notes"].source == "fallback"

    def test_fallback_policy_entity_type_is_free_text(self):
        df = pd.DataFrame({"notes": ["short"]})
        policies = self._classify(df)

        assert policies["notes"].entity_type == FREE_TEXT

    # ── Non-text columns skipped ─────────────────────────────────────────────

    def test_numeric_column_not_added_to_policies(self):
        df = pd.DataFrame({"age": [25, 30, 35]})
        policies = self._classify(df)

        assert "age" not in policies

    # ── Pre-classified columns not overwritten ───────────────────────────────

    def test_already_classified_column_not_overwritten(self):
        df = pd.DataFrame({"email": ["alice@example.com", "bob@example.com"]})
        existing = ColumnPolicy(
            column="email", entity_type="EMAIL_ADDRESS",
            action=ACTION_HASH, source="presidio_structured", score=0.9,
        )
        policies: dict[str, ColumnPolicy] = {"email": existing}
        _tier_c_fallback(df, policies)

        assert policies["email"] is existing


class TestScanColumnsIntegration:
    """End-to-end: classify → free_text_columns_from_policies → anonymize_dataframe.

    Verifies that only free-text columns are passed to the row-by-row scanner
    and that structured columns are left untouched.
    """

    class _PersonInNotes:
        """Finds PERSON in any string containing 'Alice'."""
        def analyze(self, text, entities=None, language=None, score_threshold=None):
            start = text.find("Alice")
            if start < 0:
                return []
            return [SimpleNamespace(entity_type="PERSON", start=start, end=start + len("Alice"), score=1.0)]

    def test_only_free_text_columns_scanned(self):
        free_text = [
            "Alice contacted support about a billing issue with her account last Tuesday.",
            "The customer Alice reported that her order was missing two items from the shipment.",
        ]
        df = pd.DataFrame({
            "notes": free_text,
            "status": ["Open", "Closed"],          # short structured → ACTION_BIN
        })

        analyzer = self._PersonInNotes()
        policies = classify_pii_columns(df, analyzer=analyzer, similarity_models={})
        scan_cols = free_text_columns_from_policies(policies)

        assert "notes" in scan_cols
        assert "status" not in scan_cols

        result, stats = anonymize_dataframe(df, analyzer, EntityRegistry(), scan_columns=scan_cols)

        # notes column: Alice must be masked
        for val in result["notes"]:
            assert "Alice" not in val

        # status column: unchanged — not scanned
        assert list(result["status"]) == ["Open", "Closed"]
        assert "status" not in stats["text_columns_scanned"]
