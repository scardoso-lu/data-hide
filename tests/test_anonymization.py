"""
Tests for the PII anonymization core.

Five sections
-------------
1. PII that MUST be detected and masked (most popular real-world patterns)
2. Non-PII text that must pass through UNCHANGED (no false positives)
3. Non-string Python objects that must pass through UNCHANGED (P1 regression)
4. DataFrame-level behaviour and stats
5. EntityRegistry — consistent pseudonym tokenisation
6. JSON and nested-document anonymization
"""

import json
import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from main import (
    EntityRegistry,
    _anonymize_json,
    _anonymize_text,
    anonymize_dataframe,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PII data — every pattern here must be detected and replaced
# ─────────────────────────────────────────────────────────────────────────────
# Rule-based recognisers (email, credit-card, IBAN, phone) are deterministic.
# NLP-based recognisers (PERSON) require sentence context for reliable recall;
# examples have been chosen to score highly with en_core_web_lg.

PII_CASES = [
    # ── EMAIL ADDRESS ──────────────────────────────────────────────────────
    ("email_simple",        "Please reach me at alice@example.com."),
    ("email_plus_tag",      "Filtered inbox: user+reports@mail.example.com"),
    ("email_subdomain",     "Open a ticket at helpdesk@support.acme.org"),
    ("email_country_tld",   "Send invoice to billing@company.co.uk"),
    ("email_standalone",    "bob.smith@company.com"),
    # ── PHONE NUMBER ──────────────────────────────────────────────────────
    ("phone_us_dashes",     "Call our hotline at +1-800-555-0199."),
    ("phone_us_parens",     "Appointment line: (212) 555-0147"),
    ("phone_e164",          "Registered mobile: +15005550006"),
    ("phone_international", "UK contact: +44 20 7946 0958"),
    # ── CREDIT CARD ───────────────────────────────────────────────────────
    ("cc_visa_spaced",      "Charge card 4111 1111 1111 1111 for the order."),
    ("cc_visa_compact",     "Stored card: 4111111111111111"),
    ("cc_mastercard",       "MC ending: 5500 0000 0000 0004"),
    ("cc_amex",             "Amex on file: 378282246310005"),
    ("cc_discover",         "Discover card 6011 1111 1111 1117 declined."),
    # ── IBAN CODE ─────────────────────────────────────────────────────────
    ("iban_uk",             "Wire to GB29 NWBK 6016 1331 9268 19."),
    ("iban_germany",        "German IBAN: DE89370400440532013000"),
    ("iban_france",         "Beneficiary IBAN: FR7630006000011234567890189"),
    ("iban_spain",          "Account ES9121000418450200051332"),
    # ── US BANK NUMBER ────────────────────────────────────────────────────
    ("bank_in_sentence",    "Debit account 122105155 routing 021000021."),
    # ── PERSON ────────────────────────────────────────────────────────────
    ("person_full",         "The account holder is John Smith."),
    ("person_formal",       "Best regards, Robert Johnson, CFO"),
    ("person_titled",       "Approved by Dr. Jane Doe."),
    ("person_multi",        "Contract signed by Alice Brown and Michael Davis."),
    ("person_possessive",   "Emily Clark's policy number is on file."),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Non-PII strings — Presidio must produce ZERO detections
# ─────────────────────────────────────────────────────────────────────────────

NON_PII_STRINGS = [
    "The shipment arrived on schedule.",
    "Order reference: ORD-2024-98765",
    "Status: COMPLETED",
    "SKU: WIDGET-XL-RED-42",
    "Temperature: 22.5 degrees Celsius.",
    "The quarterly review is next Tuesday.",
    "Version 3.14.0 released.",
    "Category: Home and Garden",
    "ISO date: 2024-01-15",
    "Hex colour: #ff5733",
    "Discount code: SUMMER20",
    "Country code: US",
    "Coordinates: 40.7128 N 74.0060 W",
    "",  # empty string — valid no-op input
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Non-string Python objects — must survive anonymize_dataframe UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

NON_STRING_VALUES = [
    ("none",          None),
    ("nan",           float("nan")),
    ("integer",       42),
    ("negative_int",  -7),
    ("zero",          0),
    ("float_pi",      3.14159),
    ("bool_true",     True),
    ("bool_false",    False),
    # Decimal looks like a credit-card number when str()-coerced; must be untouched
    ("decimal_cc",    Decimal("4111111111111111")),
    ("decimal_iban",  Decimal("29060161331926819")),
    # Tuples are not dict/list — pass through unchanged
    ("tuple_mixed",   (1, "alice@example.com")),
]
# Note: dict and list values ARE now anonymized recursively (see TestJSONAndDictAnonymization).


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPIIDetection:
    """Every pattern in PII_CASES must be detected and the original fragment
    must be absent from the output."""

    @pytest.mark.parametrize("case_id,text", PII_CASES)
    def test_entity_detected(self, analyzer, case_id, text):
        _result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert findings, f"[{case_id}] No entity found in: {text!r}"

    @pytest.mark.parametrize("case_id,text", PII_CASES)
    def test_token_in_output(self, analyzer, case_id, text):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        if findings:
            entity_type = findings[0].entity_type
            assert f"{entity_type}_0" in result, (
                f"[{case_id}] Token '{entity_type}_0' not found in output: {result!r}"
            )

    @pytest.mark.parametrize("case_id,text", PII_CASES)
    def test_original_pii_fragment_removed(self, analyzer, case_id, text):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        for r in findings:
            fragment = text[r.start:r.end]
            assert fragment not in result, (
                f"[{case_id}] Original fragment still present: {fragment!r} in {result!r}"
            )


class TestNoPIIPassthrough:
    """Strings with no PII must produce zero findings and be returned unchanged."""

    @pytest.mark.parametrize("text", NON_PII_STRINGS)
    def test_no_detections(self, analyzer, text):
        _result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert not findings, (
            f"Unexpected detection(s) {findings!r} in non-PII text: {text!r}"
        )

    @pytest.mark.parametrize("text", NON_PII_STRINGS)
    def test_text_returned_unchanged(self, analyzer, text):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        if not findings:
            assert result == text, (
                f"Text mutated without detections: {result!r} != {text!r}"
            )


class TestNonStringPassthrough:
    """Non-string values in object-dtype columns must pass through untouched."""

    @pytest.mark.parametrize("desc,value", NON_STRING_VALUES)
    def test_value_unchanged_in_dataframe(self, analyzer, desc, value):
        df = pd.DataFrame({"col": [value]})
        result_df, _stats = anonymize_dataframe(df, analyzer)
        actual = result_df["col"].iloc[0]

        if isinstance(value, float) and math.isnan(value):
            assert isinstance(actual, float) and math.isnan(actual), (
                f"[{desc}] NaN was mutated to {actual!r}"
            )
        else:
            assert actual == value, (
                f"[{desc}] Value changed: {value!r} → {actual!r}"
            )

    def test_mixed_column_strings_masked_non_strings_intact(self, analyzer):
        """Strings with PII are masked; non-string scalars pass through unchanged;
        dict without PII is returned structurally identical."""
        df = pd.DataFrame({
            "data": [
                "contact jane@example.com",  # str with PII       → masked
                42,                           # int                → unchanged
                None,                         # None               → unchanged
                "No PII here at all.",        # str no PII         → unchanged
                {"key": "value"},             # dict without PII   → unchanged content
                Decimal("4111111111111111"),  # Decimal CC         → unchanged
            ]
        })
        result_df, stats = anonymize_dataframe(df, analyzer)

        assert "EMAIL_ADDRESS_0" in str(result_df["data"].iloc[0])  # email masked
        assert result_df["data"].iloc[1] == 42                       # int intact
        assert result_df["data"].iloc[2] is None                     # None intact
        assert result_df["data"].iloc[3] == "No PII here at all."    # no-PII intact
        assert result_df["data"].iloc[4] == {"key": "value"}         # dict intact (no PII)
        assert result_df["data"].iloc[5] == Decimal("4111111111111111")  # Decimal intact


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DataFrame-level behaviour and stats
# ─────────────────────────────────────────────────────────────────────────────

class TestDataFrameAnonymization:

    def test_email_column_masked(self, analyzer):
        df = pd.DataFrame({"email": ["alice@example.com", "bob@company.org"]})
        result_df, stats = anonymize_dataframe(df, analyzer)

        assert "alice@example.com" not in result_df["email"].values
        assert "bob@company.org"   not in result_df["email"].values
        for val in result_df["email"]:
            assert "@" not in val
            assert "example.com" not in val

    def test_multiple_pii_types_in_one_cell(self, analyzer):
        df = pd.DataFrame({
            "note": [
                "Email alice@example.com, card 4111 1111 1111 1111."
            ]
        })
        result_df, stats = anonymize_dataframe(df, analyzer)

        assert stats["total_entities_detected"] >= 2
        assert "alice@example.com" not in result_df["note"].iloc[0]
        assert "4111 1111 1111 1111" not in result_df["note"].iloc[0]

    def test_non_object_dtype_columns_skipped_entirely(self, analyzer):
        df = pd.DataFrame({
            "id":     pd.array([1, 2, 3], dtype="int64"),
            "score":  pd.array([0.1, 0.2, 0.3], dtype="float64"),
            "active": pd.array([True, False, True], dtype="bool"),
        })
        result_df, stats = anonymize_dataframe(df, analyzer)

        pd.testing.assert_frame_equal(result_df, df)
        assert stats["text_columns_scanned"] == []
        assert stats["total_entities_detected"] == 0

    def test_pii_column_appears_in_stats(self, analyzer):
        df = pd.DataFrame({
            "email":       ["a@example.com", "b@example.com"],
            "description": ["Widget A", "Widget B"],
            "qty":         [1, 2],
        })
        _, stats = anonymize_dataframe(df, analyzer)

        assert "email" in stats["columns_with_detections"]
        assert "description" not in stats["columns_with_detections"]
        assert stats["entity_counts"]["EMAIL_ADDRESS"] >= 2
        assert stats["total_entities_detected"] >= 2

    def test_no_pii_column_absent_from_detections(self, analyzer):
        df = pd.DataFrame({
            "category": ["Electronics", "Home & Garden", "Sports"],
        })
        _, stats = anonymize_dataframe(df, analyzer)

        assert stats["columns_with_detections"] == []
        assert stats["total_entities_detected"] == 0

    def test_column_stats_list_length_matches_text_cols(self, analyzer):
        df = pd.DataFrame({
            "name":  ["Alice Smith"],
            "qty":   [5],
            "notes": ["No issues found."],
        })
        _, stats = anonymize_dataframe(df, analyzer)

        assert len(stats["column_stats"]) == 2
        col_names = [s["column"] for s in stats["column_stats"]]
        assert "name" in col_names
        assert "notes" in col_names

    def test_original_dataframe_not_mutated(self, analyzer):
        """anonymize_dataframe must operate on a copy, not the original."""
        df = pd.DataFrame({"email": ["alice@example.com"]})
        original = df["email"].iloc[0]
        anonymize_dataframe(df, analyzer)
        assert df["email"].iloc[0] == original

    def test_empty_dataframe_returns_zero_stats(self, analyzer):
        df = pd.DataFrame({"email": pd.Series([], dtype=object)})
        result_df, stats = anonymize_dataframe(df, analyzer)

        assert len(result_df) == 0
        assert stats["total_entities_detected"] == 0
        assert stats["columns_with_detections"] == []

    def test_all_null_column_leaves_nulls_intact(self, analyzer):
        df = pd.DataFrame({"email": [None, None, None]})
        result_df, stats = anonymize_dataframe(df, analyzer)

        assert result_df["email"].isna().all()
        assert stats["total_entities_detected"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EntityRegistry — consistent pseudonym tokenisation
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityRegistry:

    def test_token_format(self):
        r = EntityRegistry()
        assert r.token_for("EMAIL_ADDRESS", "alice@example.com") == "EMAIL_ADDRESS_0"

    def test_same_value_same_token(self):
        r = EntityRegistry()
        t1 = r.token_for("PERSON", "Alice Smith")
        t2 = r.token_for("PERSON", "Alice Smith")
        assert t1 == t2

    def test_different_values_different_tokens(self):
        r = EntityRegistry()
        t1 = r.token_for("PERSON", "Alice Smith")
        t2 = r.token_for("PERSON", "Bob Jones")
        assert t1 != t2

    def test_counter_increments_per_entity_type(self):
        r = EntityRegistry()
        r.token_for("PERSON", "Alice")
        t2 = r.token_for("PERSON", "Bob")
        assert t2 == "PERSON_1"

    def test_counters_independent_across_entity_types(self):
        r = EntityRegistry()
        r.token_for("PERSON", "Alice")
        r.token_for("PERSON", "Bob")
        t_email = r.token_for("EMAIL_ADDRESS", "alice@example.com")
        assert t_email == "EMAIL_ADDRESS_0"

    def test_case_insensitive_matching(self):
        r = EntityRegistry()
        t1 = r.token_for("PERSON", "ALICE SMITH")
        t2 = r.token_for("PERSON", "alice smith")
        assert t1 == t2

    def test_whitespace_trimmed(self):
        r = EntityRegistry()
        t1 = r.token_for("PERSON", "  Alice  ")
        t2 = r.token_for("PERSON", "Alice")
        assert t1 == t2

    def test_unique_counts_reflects_distinct_values(self):
        r = EntityRegistry()
        r.token_for("PERSON", "Alice")
        r.token_for("PERSON", "Bob")
        r.token_for("PERSON", "Alice")  # duplicate — no new counter increment
        r.token_for("EMAIL_ADDRESS", "alice@example.com")
        counts = r.unique_counts()
        assert counts["PERSON"] == 2
        assert counts["EMAIL_ADDRESS"] == 1

    def test_consistent_across_rows_in_dataframe(self, analyzer):
        """The same email address in two rows maps to the same token."""
        registry = EntityRegistry()
        df = pd.DataFrame({"email": ["alice@example.com", "alice@example.com"]})
        result_df, _ = anonymize_dataframe(df, analyzer, registry)
        assert result_df["email"].iloc[0] == result_df["email"].iloc[1]

    def test_different_emails_get_different_tokens(self, analyzer):
        registry = EntityRegistry()
        df = pd.DataFrame({"email": ["alice@example.com", "bob@example.com"]})
        result_df, _ = anonymize_dataframe(df, analyzer, registry)
        assert result_df["email"].iloc[0] != result_df["email"].iloc[1]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  JSON and nested-document anonymization
# ─────────────────────────────────────────────────────────────────────────────

class TestJSONAndDictAnonymization:
    """anonymize_dataframe must recurse into JSON strings and native dicts/lists."""

    def test_json_string_email_anonymized(self, analyzer):
        df = pd.DataFrame({"payload": ['{"name": "Alice", "email": "alice@example.com"}']})
        result_df, _ = anonymize_dataframe(df, analyzer)
        parsed = json.loads(result_df["payload"].iloc[0])
        assert "alice@example.com" not in parsed["email"]
        assert "@" not in parsed["email"]

    def test_json_output_is_valid_json(self, analyzer):
        df = pd.DataFrame({"payload": ['{"score": 100, "note": "Contact bob@company.com"}']})
        result_df, _ = anonymize_dataframe(df, analyzer)
        parsed = json.loads(result_df["payload"].iloc[0])
        assert isinstance(parsed, dict)
        assert parsed["score"] == 100

    def test_nested_json_pii_anonymized(self, analyzer):
        payload = '{"contact": {"name": "John Smith", "email": "john@example.com"}}'
        df = pd.DataFrame({"payload": [payload]})
        result_df, _ = anonymize_dataframe(df, analyzer)
        parsed = json.loads(result_df["payload"].iloc[0])
        assert "john@example.com" not in parsed["contact"]["email"]

    def test_json_array_pii_anonymized(self, analyzer):
        df = pd.DataFrame({"contacts": ['["alice@example.com", "bob@example.com"]']})
        result_df, _ = anonymize_dataframe(df, analyzer)
        parsed = json.loads(result_df["contacts"].iloc[0])
        for item in parsed:
            assert "example.com" not in item

    def test_native_dict_pii_anonymized(self, analyzer):
        df = pd.DataFrame({"data": [{"email": "alice@example.com", "score": 10}]})
        result_df, _ = anonymize_dataframe(df, analyzer)
        result = result_df["data"].iloc[0]
        assert isinstance(result, dict)
        assert "alice@example.com" not in result["email"]
        assert result["score"] == 10  # numeric value preserved

    def test_native_list_pii_anonymized(self, analyzer):
        df = pd.DataFrame({"emails": [["alice@example.com", "bob@example.com"]]})
        result_df, _ = anonymize_dataframe(df, analyzer)
        result = result_df["emails"].iloc[0]
        assert isinstance(result, list)
        for item in result:
            assert "example.com" not in item

    def test_json_stats_counted(self, analyzer):
        df = pd.DataFrame({"payload": ['{"email": "alice@example.com"}']})
        _, stats = anonymize_dataframe(df, analyzer)
        assert stats["total_entities_detected"] >= 1
        assert "payload" in stats["columns_with_detections"]

    def test_non_pii_json_structure_preserved(self, analyzer):
        df = pd.DataFrame({"payload": ['{"status": "completed", "score": 42}']})
        result_df, _ = anonymize_dataframe(df, analyzer)
        parsed = json.loads(result_df["payload"].iloc[0])
        assert parsed["status"] == "completed"
        assert parsed["score"] == 42

    def test_numeric_json_values_preserved(self, analyzer):
        df = pd.DataFrame({"payload": ['{"id": 123, "ratio": 9.5, "active": true}']})
        result_df, _ = anonymize_dataframe(df, analyzer)
        parsed = json.loads(result_df["payload"].iloc[0])
        assert parsed["id"] == 123
        assert abs(parsed["ratio"] - 9.5) < 1e-9

    def test_anonymize_json_function_directly(self, analyzer):
        """Unit test for _anonymize_json helper."""
        registry = EntityRegistry()
        obj = {"email": "alice@example.com", "score": 10}
        result, findings = _anonymize_json(obj, analyzer, registry)
        assert isinstance(result, dict)
        assert "alice@example.com" not in result["email"]
        assert result["score"] == 10
        assert findings  # at least one finding

    def test_anonymize_json_nested_list(self, analyzer):
        registry = EntityRegistry()
        obj = [{"email": "alice@example.com"}, {"email": "bob@example.com"}]
        result, findings = _anonymize_json(obj, analyzer, registry)
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(findings) == 2
