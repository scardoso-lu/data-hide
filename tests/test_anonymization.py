"""
Tests for the PII anonymization core.

Three sections
--------------
1. PII that MUST be detected and masked (most popular real-world patterns)
2. Non-PII text that must pass through UNCHANGED (no false positives)
3. Non-string Python objects that must pass through UNCHANGED (P1 regression)
4. DataFrame-level behaviour and stats
"""

import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from presidio_anonymizer.entities import OperatorConfig

from main import MASK_VALUE, _process_value, anonymize_dataframe

MASK = MASK_VALUE


def _ops():
    return {"DEFAULT": OperatorConfig("replace", {"new_value": MASK})}


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
# Selected to avoid known Presidio false-positive patterns.

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
# Validates the P1 guard: only isinstance(val, str) values enter Presidio.
# Types like Decimal("4111111111111111") look like a credit-card number when
# coerced to str() but must NOT be modified because they are not strings.

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
    # Containers with PII-like content — must not be processed
    ("dict_with_pii", {"email": "user@example.com"}),
    ("list_of_emails",["alice@example.com", "bob@example.com"]),
    ("tuple_mixed",   (1, "alice@example.com")),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPIIDetection:
    """Every pattern in PII_CASES must be detected and the original fragment
    must be absent from the output."""

    @pytest.mark.parametrize("case_id,text", PII_CASES)
    def test_entity_detected(self, presidio_engines, case_id, text):
        analyzer, anonymizer = presidio_engines
        _result, findings = _process_value(text, analyzer, anonymizer, _ops())
        assert findings, (
            f"[{case_id}] No entity found in: {text!r}"
        )

    @pytest.mark.parametrize("case_id,text", PII_CASES)
    def test_mask_in_output(self, presidio_engines, case_id, text):
        analyzer, anonymizer = presidio_engines
        result, findings = _process_value(text, analyzer, anonymizer, _ops())
        if findings:
            assert MASK in result, (
                f"[{case_id}] Mask '{MASK}' not found in output: {result!r}"
            )

    @pytest.mark.parametrize("case_id,text", PII_CASES)
    def test_original_pii_fragment_removed(self, presidio_engines, case_id, text):
        analyzer, anonymizer = presidio_engines
        result, findings = _process_value(text, analyzer, anonymizer, _ops())
        for r in findings:
            fragment = text[r.start : r.end]
            assert fragment not in result, (
                f"[{case_id}] Original fragment still present: {fragment!r} in {result!r}"
            )


class TestNoPIIPassthrough:
    """Strings with no PII must produce zero findings and be returned unchanged."""

    @pytest.mark.parametrize("text", NON_PII_STRINGS)
    def test_no_detections(self, presidio_engines, text):
        analyzer, anonymizer = presidio_engines
        _result, findings = _process_value(text, analyzer, anonymizer, _ops())
        assert not findings, (
            f"Unexpected detection(s) {findings!r} in non-PII text: {text!r}"
        )

    @pytest.mark.parametrize("text", NON_PII_STRINGS)
    def test_text_returned_unchanged(self, presidio_engines, text):
        analyzer, anonymizer = presidio_engines
        result, findings = _process_value(text, analyzer, anonymizer, _ops())
        if not findings:
            assert result == text, (
                f"Text mutated without detections: {result!r} != {text!r}"
            )


class TestNonStringPassthrough:
    """Non-string values in object-dtype columns must pass through untouched."""

    @pytest.mark.parametrize("desc,value", NON_STRING_VALUES)
    def test_value_unchanged_in_dataframe(self, presidio_engines, desc, value):
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({"col": [value]})
        result_df, _stats = anonymize_dataframe(df, analyzer, anonymizer)
        actual = result_df["col"].iloc[0]

        if isinstance(value, float) and math.isnan(value):
            assert isinstance(actual, float) and math.isnan(actual), (
                f"[{desc}] NaN was mutated to {actual!r}"
            )
        else:
            assert actual == value, (
                f"[{desc}] Value changed: {value!r} → {actual!r}"
            )

    def test_mixed_column_strings_masked_non_strings_intact(self, presidio_engines):
        """Within one mixed-type column only string values enter Presidio."""
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({
            "data": [
                "contact jane@example.com",  # str with PII  → must be masked
                42,                           # int           → unchanged
                None,                         # None          → unchanged
                "No PII here at all.",        # str no PII    → unchanged
                {"key": "value"},             # dict          → unchanged
                Decimal("4111111111111111"),  # Decimal CC    → unchanged
            ]
        })
        result_df, stats = anonymize_dataframe(df, analyzer, anonymizer)

        assert MASK in str(result_df["data"].iloc[0])             # email masked
        assert result_df["data"].iloc[1] == 42                    # int intact
        assert result_df["data"].iloc[2] is None                  # None intact
        assert result_df["data"].iloc[3] == "No PII here at all." # no-PII intact
        assert result_df["data"].iloc[4] == {"key": "value"}      # dict intact
        assert result_df["data"].iloc[5] == Decimal("4111111111111111")  # Decimal intact


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DataFrame-level behaviour and stats
# ─────────────────────────────────────────────────────────────────────────────

class TestDataFrameAnonymization:

    def test_email_column_masked(self, presidio_engines):
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({"email": ["alice@example.com", "bob@company.org"]})
        result_df, stats = anonymize_dataframe(df, analyzer, anonymizer)

        assert "alice@example.com" not in result_df["email"].values
        assert "bob@company.org" not in result_df["email"].values
        assert all(MASK in v for v in result_df["email"])

    def test_multiple_pii_types_in_one_cell(self, presidio_engines):
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({
            "note": [
                "Email alice@example.com, card 4111 1111 1111 1111."
            ]
        })
        result_df, stats = anonymize_dataframe(df, analyzer, anonymizer)

        assert stats["total_entities_detected"] >= 2
        assert "alice@example.com" not in result_df["note"].iloc[0]
        assert "4111 1111 1111 1111" not in result_df["note"].iloc[0]

    def test_non_object_dtype_columns_skipped_entirely(self, presidio_engines):
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({
            "id":     pd.array([1, 2, 3], dtype="int64"),
            "score":  pd.array([0.1, 0.2, 0.3], dtype="float64"),
            "active": pd.array([True, False, True], dtype="bool"),
        })
        result_df, stats = anonymize_dataframe(df, analyzer, anonymizer)

        pd.testing.assert_frame_equal(result_df, df)
        assert stats["text_columns_scanned"] == []
        assert stats["total_entities_detected"] == 0

    def test_pii_column_appears_in_stats(self, presidio_engines):
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({
            "email":       ["a@example.com", "b@example.com"],
            "description": ["Widget A", "Widget B"],
            "qty":         [1, 2],
        })
        _, stats = anonymize_dataframe(df, analyzer, anonymizer)

        assert "email" in stats["columns_with_detections"]
        assert "description" not in stats["columns_with_detections"]
        assert stats["entity_counts"]["EMAIL_ADDRESS"] >= 2
        assert stats["total_entities_detected"] >= 2

    def test_no_pii_column_absent_from_detections(self, presidio_engines):
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({
            "category": ["Electronics", "Home & Garden", "Sports"],
        })
        _, stats = anonymize_dataframe(df, analyzer, anonymizer)

        assert stats["columns_with_detections"] == []
        assert stats["total_entities_detected"] == 0

    def test_column_stats_list_length_matches_text_cols(self, presidio_engines):
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({
            "name":  ["Alice Smith"],
            "qty":   [5],                   # int — skipped
            "notes": ["No issues found."],
        })
        _, stats = anonymize_dataframe(df, analyzer, anonymizer)

        # Only 'name' and 'notes' are object dtype
        assert len(stats["column_stats"]) == 2
        col_names = [s["column"] for s in stats["column_stats"]]
        assert "name" in col_names
        assert "notes" in col_names

    def test_original_dataframe_not_mutated(self, presidio_engines):
        """anonymize_dataframe must operate on a copy, not the original."""
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({"email": ["alice@example.com"]})
        original = df["email"].iloc[0]
        anonymize_dataframe(df, analyzer, anonymizer)
        assert df["email"].iloc[0] == original

    def test_empty_dataframe_returns_zero_stats(self, presidio_engines):
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({"email": pd.Series([], dtype=object)})
        result_df, stats = anonymize_dataframe(df, analyzer, anonymizer)

        assert len(result_df) == 0
        assert stats["total_entities_detected"] == 0
        assert stats["columns_with_detections"] == []

    def test_all_null_column_leaves_nulls_intact(self, presidio_engines):
        analyzer, anonymizer = presidio_engines
        df = pd.DataFrame({"email": [None, None, None]})
        result_df, stats = anonymize_dataframe(df, analyzer, anonymizer)

        assert result_df["email"].isna().all()
        assert stats["total_entities_detected"] == 0
