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
import re
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
from app.classification import (
    ACTION_BIN,
    ACTION_HASH,
    ACTION_SCAN,
    ACTION_TOKENIZE,
    apply_column_policies,
    classify_pii_columns,
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
    # "Country code: US" intentionally removed — LOCATION is now a GDPR
    # quasi-identifier and 'US' is correctly flagged as such (iter 3).
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

    def test_json_object_key_pii_anonymized(self, analyzer):
        df = pd.DataFrame({"payload": ['{"alice@example.com": "primary contact"}']})

        result_df, _ = anonymize_dataframe(df, analyzer)

        parsed = json.loads(result_df["payload"].iloc[0])
        assert "alice@example.com" not in parsed
        assert any(key.startswith("EMAIL_ADDRESS_") for key in parsed)

    def test_json_string_primitive_remains_valid_json(self, analyzer):
        df = pd.DataFrame({"payload": ['"alice@example.com"']})

        result_df, _ = anonymize_dataframe(df, analyzer)

        parsed = json.loads(result_df["payload"].iloc[0])
        assert parsed.startswith("EMAIL_ADDRESS_")

    def test_native_dict_with_non_string_keys_preserved(self, analyzer):
        df = pd.DataFrame({"payload": [{1: "alice@example.com", "score": 10}]})

        result_df, _ = anonymize_dataframe(df, analyzer)

        result = result_df["payload"].iloc[0]
        assert result[1].startswith("EMAIL_ADDRESS_")
        assert result["score"] == 10


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Expanded GDPR entity coverage — patterns that previously leaked through
# ─────────────────────────────────────────────────────────────────────────────
# These cases were added during the hardening sweep that converted GDPR_ENTITIES
# from the original 8-item whitelist to the broader catalog covering special
# Article 9 categories (health, ethnicity), national identifiers (SSN, passport,
# driver's licence, CCSS) and financial sensitive data (salary).
#
# Each subclass below pins one category that MUST be masked.  The originals
# were proven to leak through by running anonymize_dataframe against them on
# the previous codebase.


SSN_CASES = [
    ("ssn_us_dashes",       "Member's SSN: 912-34-5678 on file."),
    ("ssn_us_spaces",       "Social security 912 34 5678 verified."),
    ("ssn_us_inline",       "He gave his social security number as 612-34-5678."),
    # Compact 9-digit (e.g. 912345678) is intentionally NOT covered — it is
    # indistinguishable from ZIP+4, bank routing/account numbers and order
    # references.  Detection without context produces high false-positive
    # rates on real datasets.
]

LU_CCSS_CASES = [
    ("ccss_full",           "Matricule: 1985032512345"),
    ("ccss_with_label",     "Numéro CCSS 1985032512345 enregistré."),
    ("ccss_inline_lb",      "D'CCSS-Nummer ass 1985032512345 fir den Employé."),
]


COURT_CASE_CASES = [
    ("court_us",       "Case No. 2024-CV-12345 dismissed.",                  "2024-CV-12345"),
    ("court_fr",       "Affaire n° 23/4567 jugée hier.",                     "23/4567"),
    ("court_de",       "Aktenzeichen 5 C 1234/24 erledigt.",                 "1234/24"),
    ("docket_year",    "Docket 2023-CR-9988 pending.",                       "2023-CR-9988"),
]

INVOICE_CASES = [
    ("invoice_inv",    "Invoice INV-2024-00078 issued.",                     "INV-2024-00078"),
    ("invoice_fact",   "Facture #F-12345 due in 30 days.",                   "F-12345"),
    ("invoice_de",     "Rechnung Nr. R-2024/0099 ausgestellt.",              "R-2024/0099"),
]


class TestCourtCaseDetection:
    @pytest.mark.parametrize("case_id,text,fragment", COURT_CASE_CASES)
    def test_court_case_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Court case ref {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


class TestInvoiceNumberDetection:
    @pytest.mark.parametrize("case_id,text,fragment", INVOICE_CASES)
    def test_invoice_number_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Invoice number {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


POSTAL_CODE_CASES = [
    ("postcode_lu",    "Address: 25 Rue de la Gare, L-1611 Luxembourg.",          "L-1611"),
    ("postcode_uk",    "Office postcode SW1A 1AA London.",                         "SW1A 1AA"),
    ("postcode_de",    "Anschrift: Hauptstrasse 8, 10115 Berlin.",                 "10115"),
    ("postcode_fr",    "Adresse postale: 75008 Paris.",                            "75008"),
    ("postcode_nl",    "Postcode: 1012 AB Amsterdam.",                             "1012 AB"),
]

POSTAL_CODE_NON_PII = [
    # Standalone 5-digit runs without postal/address context should not be
    # masked — they are commonly counts, prices, or unrelated identifiers.
    "Inventory: 75008 units shipped.",
    "Pi to 5 digits: 31415",
]


class TestPostalCodeDetection:
    @pytest.mark.parametrize("case_id,text,fragment", POSTAL_CODE_CASES)
    def test_postal_code_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Postal code fragment {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )

    @pytest.mark.parametrize("text", POSTAL_CODE_NON_PII)
    def test_no_false_positive_without_postal_context(self, analyzer, text):
        _result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        pc = [f for f in findings if f.entity_type == "POSTAL_CODE"]
        assert not pc, f"False positive: {text!r} → {[(f.entity_type, f.score) for f in pc]}"


ART9_ART10_CASES = [
    ("orientation_gay",      "Patient identifies as gay.",                    "gay",        "SEXUAL_ORIENTATION"),
    ("orientation_lgbt",     "Outreach program for LGBT youth.",              "LGBT",       "SEXUAL_ORIENTATION"),
    ("orientation_trans",    "Transgender support group meets weekly.",        "Transgender","SEXUAL_ORIENTATION"),
    ("union_cgt",            "Member of CGT since 1998.",                      "CGT",        "TRADE_UNION"),
    ("union_dgb",            "Affiliated with DGB representative.",            "DGB",        "TRADE_UNION"),
    ("crime_convicted",      "Subject was convicted of fraud in 2019.",        "convicted",  "CRIMINAL_RECORD"),
    ("crime_arrested",       "Suspect arrested on burglary charges.",          "arrested",   "CRIMINAL_RECORD"),
    ("crime_felony",         "Prior felony on the candidate's record.",        "felony",     "CRIMINAL_RECORD"),
]


class TestArt9Art10Detection:
    # See `_ART9_ART10_ENTITY_TYPES` below — the six Art. 9 / Art. 10
    # semantic categories plus spaCy's umbrella NRP label.  The categories
    # overlap in embedding space (Jewish is both religion and ethnicity;
    # "gay" embeds close to both SEXUAL_ORIENTATION and ETHNICITY) so the
    # masking goal does not depend on the exact category label firing.
    _ART9_TYPES_LOCAL = frozenset({
        "HEALTH_CONDITION", "ETHNICITY", "RELIGION", "SEXUAL_ORIENTATION",
        "TRADE_UNION", "CRIMINAL_RECORD", "NRP",
    })

    @pytest.mark.parametrize("case_id,text,fragment,expected_type", ART9_ART10_CASES)
    def test_art9_art10_masked(self, analyzer, case_id, text, fragment, expected_type):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Art. 9/10 fragment {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )
        assert any(f.entity_type in self._ART9_TYPES_LOCAL for f in findings), (
            f"[{case_id}] No Art. 9 / Art. 10 finding (expected at minimum "
            f"{expected_type!r}); got {[(f.entity_type, f.score) for f in findings]}"
        )


HEALTH_INSURANCE_CASES = [
    ("carte_vitale",   "Carte Vitale 185041234567890 enregistrée.",          "185041234567890"),
    ("kvnr_de",        "Krankenversichertennummer A123456789 aktiv.",         "A123456789"),
    ("nhs_number",     "NHS number 943 476 5919 on file.",                    "943 476 5919"),
]


class TestHealthInsuranceDetection:
    @pytest.mark.parametrize("case_id,text,fragment", HEALTH_INSURANCE_CASES)
    def test_health_insurance_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Health-insurance fragment {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


BOOKING_REF_CASES = [
    ("pnr_airline",    "Flight PNR ABC123 confirmed for the passenger.",      "ABC123"),
    ("booking_label",  "Booking BK-2024-7788 received.",                       "BK-2024-7788"),
    ("res_label",      "Reservation RSV-99887 hotel cancelled.",               "RSV-99887"),
]

CUSTOMER_ID_CASES = [
    ("cust_id",        "Customer CUST-12345 escalated the ticket.",            "CUST-12345"),
    ("employee_id",    "Employee #E-45678 transferred to Berlin office.",      "E-45678"),
    ("badge_id",       "Badge 98765 deactivated yesterday.",                   "98765"),
    ("matricule_lu",   "Matricule personnel: M-2024-001 active.",              "M-2024-001"),
]


class TestBookingRefDetection:
    @pytest.mark.parametrize("case_id,text,fragment", BOOKING_REF_CASES)
    def test_booking_ref_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Booking ref {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


class TestCustomerEmployeeIDDetection:
    @pytest.mark.parametrize("case_id,text,fragment", CUSTOMER_ID_CASES)
    def test_customer_employee_id_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Internal ID {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


VEHICLE_PLATE_CASES = [
    ("plate_lu",       "Vehicle plate AB 1234 from Luxembourg.",          "AB 1234"),
    ("plate_de",       "License plate: M-AB 1234 registered.",            "M-AB 1234"),
    ("plate_uk",       "Vehicle UK plate AB12 CDE involved in accident.", "AB12 CDE"),
    ("plate_fr",       "French plate AA-123-BB stationné.",               "AA-123-BB"),
    ("plate_it",       "Italian vehicle plate AB 123 CD identified.",     "AB 123 CD"),
]


class TestVehiclePlateDetection:
    @pytest.mark.parametrize("case_id,text,fragment", VEHICLE_PLATE_CASES)
    def test_vehicle_plate_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Plate fragment {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


INSURANCE_CASES = [
    ("pol_prefix",     "Insurance POL-AB-12345 active.",                          "POL-AB-12345"),
    ("policy_hash",    "Policy #ABC-123456 issued.",                              "ABC-123456"),
    ("policy_fr",      "Police d'assurance n° FR-987-654 valid.",                 "FR-987-654"),
    ("policy_de",      "Versicherungsnummer DE/2024/001234 aktiv.",               "DE/2024/001234"),
]


class TestInsurancePolicyDetection:
    @pytest.mark.parametrize("case_id,text,fragment", INSURANCE_CASES)
    def test_insurance_policy_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Policy fragment {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


SWIFT_BIC_CASES = [
    ("bic_lu",         "BIC: BCEELULL for the wire.",                       "BCEELULL"),
    ("bic_de_11",      "SWIFT code DEUTDEFFXXX provided.",                  "DEUTDEFFXXX"),
    ("bic_fr",         "Bank BIC BNPAFRPP signed off.",                     "BNPAFRPP"),
    ("bic_uk",         "Use BIC BARCGB22 for the SEPA transfer.",           "BARCGB22"),
]


class TestSwiftBICDetection:
    @pytest.mark.parametrize("case_id,text,fragment", SWIFT_BIC_CASES)
    def test_swift_bic_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] SWIFT/BIC fragment {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


NATIONAL_TAX_ID_CASES = [
    ("siren_fr",       "SIREN 732 829 320 is registered.",                  "732 829 320"),
    ("siret_fr",       "SIRET: 73282932000074 active.",                      "73282932000074"),
    ("steuer_de",      "Steuernummer 12/345/67890 enregistré.",              "12/345/67890"),
    ("utr_uk",         "UK UTR 1234567890 filed.",                            "1234567890"),
    ("ein_us",         "Federal EIN 12-3456789 for the entity.",             "12-3456789"),
    ("nir_fr",         "INSEE NIR: 1850412345678",                            "1850412345678"),
]

NATIONAL_TAX_ID_NON_PII = [
    # 9-digit shape without tax context — must not be flagged (could be a
    # product code, version count, or unrelated number).
    "Inventory count: 732829320 units",
    "Random number 73282932000074 unrelated.",
]


class TestNationalTaxIDDetection:
    @pytest.mark.parametrize("case_id,text,fragment", NATIONAL_TAX_ID_CASES)
    def test_national_tax_id_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] National tax ID fragment {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )

    @pytest.mark.parametrize("text", NATIONAL_TAX_ID_NON_PII)
    def test_digit_runs_without_tax_context(self, analyzer, text):
        _result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        tax = [f for f in findings if f.entity_type == "NATIONAL_TAX_ID"]
        assert not tax, (
            f"False positive: {text!r} → {[(f.entity_type, f.score) for f in tax]}"
        )


MULTILINGUAL_KEYWORD_CASES = [
    # ── French (fr) ─────────────────────────────────────────────────────
    ("fr_health_diabete",      "Le patient est atteint de diabète depuis l'enfance.",  "diabète",      "HEALTH_CONDITION"),
    ("fr_health_avc",          "Suite à un AVC l'an dernier.",                          "AVC",          "HEALTH_CONDITION"),
    ("fr_religion_musulman",   "S'identifie comme musulman pratiquant.",                "musulman",     "RELIGION"),
    ("fr_ethnicity_asiatique", "Patientèle principalement asiatique.",                  "asiatique",    "ETHNICITY"),
    ("fr_orientation_lesbienne", "Couple lesbienne adoptant un enfant.",                "lesbienne",    "SEXUAL_ORIENTATION"),
    ("fr_union_cgt",           "Adhérent du syndicat CGT depuis 1998.",                 "syndicat",     "TRADE_UNION"),
    ("fr_crime_condamne",      "Le suspect a été condamné pour fraude.",                "condamné",     "CRIMINAL_RECORD"),
    # ── German (de) ─────────────────────────────────────────────────────
    ("de_health_kriibs",       "Der Patient leidet an Krebs.",                           "Krebs",        "HEALTH_CONDITION"),
    ("de_health_schwanger",    "Die Patientin ist schwanger im zweiten Trimester.",     "schwanger",    "HEALTH_CONDITION"),
    ("de_religion_juedisch",   "Mitglied der jüdischen Gemeinde.",                       "jüdisch",      "RELIGION"),
    ("de_ethnicity_araber",    "Der Kunde ist Araber aus Damaskus.",                    "Araber",       "ETHNICITY"),
    ("de_orientation_schwul",  "Identifiziert sich als schwul.",                         "schwul",       "SEXUAL_ORIENTATION"),
    ("de_union_gewerk",        "Mitglied der Gewerkschaft Ver.di.",                      "Gewerkschaft", "TRADE_UNION"),
    ("de_crime_verurteilt",    "Der Verdächtige wurde verurteilt.",                      "verurteilt",   "CRIMINAL_RECORD"),
    # ── Luxembourgish (lb) ───────────────────────────────────────────────
    ("lb_health_kriibs",       "De Patient huet Kriibs am leschte Joer kritt.",         "Kriibs",       "HEALTH_CONDITION"),
    ("lb_health_schwanger",    "D'Madamm ass schwanger am zweete Mount.",                "schwanger",    "HEALTH_CONDITION"),
    ("lb_religion_kathoulesch","Hien ass kathoulesch erzunn ginn.",                      "kathoulesch",  "RELIGION"),
    ("lb_ethnicity_letzebuerger","Lëtzebuerger Bierger an der Datebank.",                "Lëtzebuerger", "ETHNICITY"),
    ("lb_union_ogbl",          "Member vum OGBL säit 2010.",                             "OGBL",         "TRADE_UNION"),
    ("lb_union_lcgb",          "Affilijéiert mam LCGB.",                                 "LCGB",         "TRADE_UNION"),
    ("lb_crime_verurteelt",    "De Verdächtegen ass verurteelt ginn.",                  "verurteelt",   "CRIMINAL_RECORD"),
    ("lb_crime_prisong",       "Mam Prisongstrof bestrooft.",                            "Prisongstrof", "CRIMINAL_RECORD"),
]


# Categories that overlap heavily in spaCy's vector space — a token like
# `jüdisch` is simultaneously religious and ethnic, and `musulman` is both
# religious and (in some embeddings) ethnic.  The masking goal does not
# depend on the exact label; only that *some* Art. 9 / Art. 10 / NRP
# entity wins on the span so the fragment is removed.
_ART9_ART10_ENTITY_TYPES = frozenset({
    "HEALTH_CONDITION", "ETHNICITY", "RELIGION", "SEXUAL_ORIENTATION",
    "TRADE_UNION", "CRIMINAL_RECORD",
    "NRP",  # spaCy NER's combined nationality/religious/political label
})


class TestMultilingualKeywordDetection:
    """GDPR Art. 9 / Art. 10 keywords must be detected (and masked) in every
    language the pipeline supports (en, fr, de, lb).

    Since the categories overlap semantically in embedding space (e.g. a
    Jewish person is both an ethnicity and a religion), this test asserts:
      1. the fragment is masked (the only security-critical contract), and
      2. *some* Art. 9 / Art. 10 entity fired (any of the six categories or
         the umbrella NRP).
    The specific category label is informational, not security-critical."""

    @pytest.mark.parametrize("case_id,text,fragment,expected_type", MULTILINGUAL_KEYWORD_CASES)
    def test_multilingual_keyword_masked(self, analyzer, case_id, text, fragment, expected_type):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] {expected_type} keyword {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )
        assert any(f.entity_type in _ART9_ART10_ENTITY_TYPES for f in findings), (
            f"[{case_id}] No Art. 9 / Art. 10 finding (expected at minimum "
            f"{expected_type!r}); got {[(f.entity_type, f.score) for f in findings]}"
        )


EMAIL_PHONE_TYPO_CASES = [
    # Email and phone main-number recognizers fire on regex shape regardless
    # of label — these tests guard against future regressions if context
    # were ever made mandatory.
    ("email_label_typo",   "Emial: alice@example.com sent confirmation.", "alice@example.com"),
    ("contact_label_typo", "Cntact: bob@company.org for support.",        "bob@company.org"),
    # Phone-extension label typos — pst*, ext*, exten* must still trigger
    # the extension recognizer and mask the digits after the typo'd label.
    ("ext_pste_typo",      "Bureau +33 1 42 86 82 00 pste 412 ouvert.",   "412"),
    ("ext_exten_typo",     "Call +1-800-555-0199 exten 1234 for support.","1234"),
    ("ext_psote_typo",     "Office +49 30 12345 psote 99 active.",        "99"),
]


class TestEmailPhoneLabelTypos:
    """Email regex and phone-extension recognizers must remain effective when
    the surrounding label or extension keyword has a small typo."""

    @pytest.mark.parametrize("case_id,text,fragment", EMAIL_PHONE_TYPO_CASES)
    def test_email_phone_with_typo_label_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Sensitive value {fragment!r} leaked due to typo'd label. "
            f"Result: {result!r} (findings: {[(f.entity_type, f.score) for f in findings]})"
        )


ART9_TYPO_CASES = [
    ("religion_catholc",    "Member of the local Catholc parish.",      "Catholc"),
    ("religion_musllim",    "Identifies as Musllim faith.",              "Musllim"),
    ("religion_jewsh",      "Jewsh community center.",                   "Jewsh"),
    ("orientation_transgd", "Transgendr support group.",                  "Transgendr"),
    ("union_cgtt",          "Affilated with CGTT representative.",        "CGTT"),
    ("ethnicity_afram",     "African Amrcan customer base.",              "African Amrcan"),
    ("ethnicity_asain",     "Asain market expansion strategy.",           "Asain"),
]


class TestArt9TyposMasked:
    """Special-category Art. 9 keywords with a single mis-typed letter must
    still trigger their deny-list recognizer."""

    @pytest.mark.parametrize("case_id,text,fragment", ART9_TYPO_CASES)
    def test_art9_typo_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Art. 9 typo'd keyword {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


ADDRESS_TYPO_CASES = [
    ("address_typo_adress",   "Adress: 25 Rue de la Gare, L-1611 Luxembourg.",          "L-1611"),
    ("address_typo_anschrif", "Anschrif: Hauptstrasse 8, 10115 Berlin.",                "10115"),
    ("postcode_typo_pstcode", "Pstcode SW1A 1AA London office.",                         "SW1A 1AA"),
    ("postcode_typo_pstal",   "Pstal cd 75008 Paris.",                                   "75008"),
    ("postcode_typo_adres",   "Adres: 1012 AB Amsterdam.",                               "1012 AB"),
]


class TestAddressLabelTypos:
    """Address/postal-code recognizers depend on label context.  A typo'd
    'Adress' or 'Pstcode' label must not defeat the boost."""

    @pytest.mark.parametrize("case_id,text,fragment", ADDRESS_TYPO_CASES)
    def test_address_typo_label_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Address/postal fragment {fragment!r} leaked due to "
            f"typo'd label. Result: {result!r} (findings: "
            f"{[(f.entity_type, f.score) for f in findings]})"
        )


LABEL_TYPO_CASES = [
    ("passport_typo",      "Passprt A12345678 issued in 2020.",          "A12345678"),
    ("contract_typo",      "Contrct CTR-2024-001 was signed.",            "CTR-2024-001"),
    ("dl_two_typos",       "Drivr lcense D12345678 expires.",             "D12345678"),
    ("swift_typo",         "Swft code DEUTDEFFXXX used.",                 "DEUTDEFFXXX"),
    ("agreement_typo",     "Master aggrement #AG-9988-XYZ activated.",    "AG-9988-XYZ"),
]


class TestIdentifierLabelTypos:
    """Context-driven recognizers must remain effective when the user mis-spells
    the label adjacent to the sensitive value."""

    @pytest.mark.parametrize("case_id,text,fragment", LABEL_TYPO_CASES)
    def test_identifier_with_typo_label_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Identifier {fragment!r} leaked because the context label "
            f"was mis-spelled. Result: {result!r} (findings: "
            f"{[(f.entity_type, f.score) for f in findings]})"
        )


HEALTH_TYPO_CASES = [
    ("typo_diabetis",      "Patient has diabetis since childhood.",     "diabetis"),
    ("typo_alzheimers",    "Father has alzheimers diagnosis.",           "alzheimers"),
    ("typo_depresion",     "Long history of depresion noted.",           "depresion"),
    ("typo_anxitey",       "Treated for anxitey and stress.",            "anxitey"),
    ("typo_cancerus",      "Cancerus growth detected last year.",        "Cancerus"),
    ("typo_ashma",         "Severe ashma triggered by pollen.",          "ashma"),
    ("typo_pregant",       "She is pregant in her second trimester.",    "pregant"),
    ("typo_alzhiemer",     "Suspected alzhiemer onset.",                  "alzhiemer"),
]


class TestHealthConditionTypos:
    """Users routinely mis-spell condition names in free-text fields.  A single
    extra/missing letter must NOT bypass the Art. 9 health guard."""

    @pytest.mark.parametrize("case_id,text,fragment", HEALTH_TYPO_CASES)
    def test_typo_health_condition_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Typo'd health condition {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


CONTRACT_CASES = [
    ("ctr_dashed",       "Contract CTR-2024-001 was signed yesterday.",        "CTR-2024-001"),
    ("ctr_fr",           "Numéro de contrat: C-FR-12345678.",                  "C-FR-12345678"),
    ("ctr_de_slash",     "Vertrag Nr. DE-2024/0078 in force.",                 "DE-2024/0078"),
    ("ctr_master",       "Master agreement #AG-9988-XYZ activated.",           "AG-9988-XYZ"),
    ("ctr_id_label",     "Contract ID: 2024-00789-CRM signed.",                "2024-00789-CRM"),
]

CONTRACT_NON_PII_CASES = [
    # Without a contract-related context word these reference shapes are
    # legitimate non-PII (order numbers, SKUs).
    "Order code: ORD-2024-98765",
    "SKU: WIDGET-XL-RED-42",
    "Version 3.14.0 released.",
]


class TestContractNumberDetection:
    @pytest.mark.parametrize("case_id,text,fragment", CONTRACT_CASES)
    def test_contract_number_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Contract number fragment {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )

    @pytest.mark.parametrize("text", CONTRACT_NON_PII_CASES)
    def test_no_false_positive_without_contract_context(self, analyzer, text):
        _result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        ctr = [f for f in findings if f.entity_type == "CONTRACT_NUMBER"]
        assert not ctr, (
            f"False positive on non-contract reference: {text!r} → {[(f.entity_type, f.score) for f in ctr]}"
        )


PHONE_EXTENSION_CASES = [
    ("phone_ext_us",       "Call +1-800-555-0199 ext. 1234 to reach support.", "1234"),
    ("phone_ext_x",        "Reception (212) 555-0147 x789.",                    "x789"),
    ("phone_ext_fr",       "Bureau +33 1 42 86 82 00 poste 412.",               "412"),
]

IPV6_CASES = [
    ("ipv6_full",          "Source: 2001:0db8:85a3:0000:0000:8a2e:0370:7334 logged.",  "2001:0db8:85a3:0000:0000:8a2e:0370:7334"),
    ("ipv6_short",         "IPv6 short: fe80::1ff:fe23:4567:890a is the link-local.",  "fe80::1ff:fe23:4567:890a"),
    ("ipv6_in_log",        "Connection from 2001:db8::1 refused.",                     "2001:db8::1"),
]


class TestPhoneEdgeFormats:
    @pytest.mark.parametrize("case_id,text,fragment", PHONE_EXTENSION_CASES)
    def test_phone_extension_part_masked(self, analyzer, case_id, text, fragment):
        """The extension digits must be masked along with the main number,
        otherwise re-identification is trivial within an organization."""
        result, _ = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Phone extension fragment {fragment!r} leaked in: {result!r}"
        )


IPV6_HEX_GROUP_RE = re.compile(r"\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{0,4}){2,}\b")


class TestIPv6EdgeFormats:
    @pytest.mark.parametrize("case_id,text,fragment", IPV6_CASES)
    def test_ipv6_fully_masked(self, analyzer, case_id, text, fragment):
        result, _ = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] IPv6 fragment {fragment!r} leaked in: {result!r}"
        )
        # Stronger guard: ANY hex-colon chain ≥3 groups long in the result
        # indicates a partial IPv6 leak (Presidio's default IPv6 regex matches
        # only the leading run before "::" — the tail can survive).
        leftover = IPV6_HEX_GROUP_RE.findall(result)
        assert not leftover, (
            f"[{case_id}] Partial IPv6 fragments survived in: {result!r}: {leftover}"
        )

    def test_residual_check_catches_unmasked_ipv6(self):
        """Belt-and-braces: even if the upstream recognizer misses an IPv6,
        the residual safety net must abort the pipeline."""
        df = pd.DataFrame({"log": ["Connection from fe80::1ff:fe23:4567:890a refused."]})
        with pytest.raises(RuntimeError, match="IP_ADDRESS"):
            from main import validate_residual_pii
            validate_residual_pii(df)


TAX_ID_CASES = [
    ("vat_lu",         "Invoice VAT: LU12345678",                       "LU12345678"),
    ("vat_de",         "Lieferant USt-IdNr DE123456789",                "DE123456789"),
    ("vat_fr",         "TVA intracommunautaire FR12345678901",          "FR12345678901"),
    ("vat_be",         "BTW BE0123456789",                              "BE0123456789"),
    ("vat_it",         "Partita IVA IT12345678901",                     "IT12345678901"),
    ("itin_us",        "Tax filing ITIN 912-34-5678 was rejected.",     "912-34-5678"),
]


class TestTaxIDDetection:
    @pytest.mark.parametrize("case_id,text,fragment", TAX_ID_CASES)
    def test_tax_id_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Tax ID fragment {fragment!r} leaked in: {result!r} "
            f"(findings: {[(f.entity_type, f.score) for f in findings]})"
        )


MEDICAL_ID_CASES = [
    ("med_license",         "Dr. Smith, medical license AB1234567 issued 2010.",   "AB1234567"),
    ("mrn",                 "MRN: 12345678 admitted yesterday.",                    "12345678"),
    ("chart_number",        "Chart #X12345 needs review.",                          "X12345"),
    ("dossier",             "Dossier patient: 98765432 archivé.",                   "98765432"),
    ("pid",                 "Patient PID 4567890 transferred.",                      "4567890"),
]


class TestMedicalIDDetection:
    @pytest.mark.parametrize("case_id,text,fragment", MEDICAL_ID_CASES)
    def test_medical_id_masked(self, analyzer, case_id, text, fragment):
        result, _findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Medical identifier fragment {fragment!r} leaked in: {result!r}"
        )


NRP_CASES = [
    ("religion_catholic",   "Member of the local Catholic parish.",      "Catholic"),
    ("religion_muslim",     "Identifies as Muslim.",                      "Muslim"),
    ("religion_jewish",     "Jewish community center membership.",        "Jewish"),
    ("ethnicity_african",   "Marketing campaign for African American customers.", "African American"),
    ("ethnicity_asian",     "Patient is Asian descent.",                  "Asian"),
    ("nationality_italian", "Italian national living in Luxembourg.",     "Italian"),
]


class TestNRPDetection:
    """Nationality / Religion / Political affiliation — GDPR Art. 9 special category."""

    @pytest.mark.parametrize("case_id,text,fragment", NRP_CASES)
    def test_nrp_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] NRP keyword {fragment!r} leaked in: {result!r}"
        )
        assert any(f.entity_type in {"NRP", "RELIGION", "ETHNICITY"} for f in findings), (
            f"[{case_id}] No NRP/RELIGION/ETHNICITY finding; got {[(f.entity_type, f.score) for f in findings]}"
        )


HEALTH_CASES = [
    ("health_hiv",            "Patient is HIV positive since 2010.",       "HIV"),
    ("health_diabetes",       "He has diabetes type 2.",                    "diabetes"),
    ("health_cancer",         "Diagnosed with cancer last year.",           "cancer"),
    ("health_depression",     "Treated for depression and anxiety.",        "depression"),
    ("health_pregnancy",      "She is pregnant in her second trimester.",   "pregnant"),
    ("health_alzheimer",      "Father suffers from Alzheimer disease.",     "Alzheimer"),
    ("health_ptsd",           "Veteran with PTSD claim pending.",           "PTSD"),
]


class TestHealthConditionDetection:
    @pytest.mark.parametrize("case_id,text,fragment", HEALTH_CASES)
    def test_health_condition_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Health condition keyword {fragment!r} leaked in: {result!r}"
        )
        assert any(f.entity_type == "HEALTH_CONDITION" for f in findings), (
            f"[{case_id}] No HEALTH_CONDITION finding; got {[(f.entity_type, f.score) for f in findings]}"
        )

    def test_non_health_words_not_flagged(self, analyzer):
        # Words that merely contain a health-condition substring must not match.
        # "candidate" must NOT match "AIDS"; "diabetic-friendly recipe" is OK
        # but "diabetic" itself remains PII (intentionally — Art. 9 keyword).
        text = "Best candidate for the marketing role."
        _result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        health = [f for f in findings if f.entity_type == "HEALTH_CONDITION"]
        assert not health, f"False positive on candidate: {[(f.entity_type, f.score) for f in findings]}"


SALARY_CASES = [
    ("salary_eur_annual",   "Annual salary: EUR 75000",                  "75000"),
    ("salary_usd_per_year", "Compensation $120,000/year for the role.",   "$120,000"),
    ("salary_monthly",      "She earns 4500 EUR per month.",              "4500 EUR"),
    ("salary_euro_symbol",  "Monthly wage: 3,200 €",                      "3,200 €"),
    ("salary_with_k",       "Base salary: $95k annually.",                "$95k"),
]

SALARY_NON_PII_CASES = [
    "Order total: €1500",
    "Price tag: $5.99",
    "Item cost: EUR 99",
    "Refund of $20 issued.",
]


class TestSalaryDetection:
    @pytest.mark.parametrize("case_id,text,fragment", SALARY_CASES)
    def test_salary_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Salary fragment {fragment!r} leaked in: {result!r}"
        )
        assert any(f.entity_type == "SALARY" for f in findings), (
            f"[{case_id}] No SALARY finding; got {[(f.entity_type, f.score) for f in findings]}"
        )

    @pytest.mark.parametrize("text", SALARY_NON_PII_CASES)
    def test_prices_not_flagged_as_salary(self, analyzer, text):
        # Money amounts without salary/wage/compensation context must NOT
        # be flagged.  Otherwise every product price in a free-text column
        # would get masked, destroying analytics.
        _result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        salary_findings = [f for f in findings if f.entity_type == "SALARY"]
        assert not salary_findings, (
            f"Generic price should not be flagged as SALARY: {text!r}, got {[(f.entity_type, f.score) for f in salary_findings]}"
        )


DOB_CASES = [
    ("dob_iso",         "Date of birth: 1985-03-15.",         "1985-03-15"),
    ("dob_dmy_slash",   "DOB 15/03/1985",                     "15/03/1985"),
    ("dob_us_slash",    "DOB 03/15/1985",                     "03/15/1985"),
    ("dob_written_en",  "Born on June 21, 1990 in Lyon.",     "1990"),
    ("dob_written_fr",  "Né le 15 mars 1985 à Paris.",        "1985"),
    ("dob_geburt_de",   "Geburtsdatum: 1985-03-15",           "1985-03-15"),
]

DOB_NON_PII_CASES = [
    # Generic dates without DOB context must NOT be flagged — they are NOT
    # personal data on their own.
    "ISO date: 2024-01-15",
    "Version 3.14.0 released on 2024-01-15",
    "Q1 review scheduled for 2024-03-01.",
]


class TestDateOfBirthDetection:
    @pytest.mark.parametrize("case_id,text,fragment", DOB_CASES)
    def test_dob_masked(self, analyzer, case_id, text, fragment):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] DOB fragment {fragment!r} leaked in: {result!r}"
        )
        assert any(f.entity_type == "DATE_OF_BIRTH" for f in findings), (
            f"[{case_id}] No DATE_OF_BIRTH finding; got {[(f.entity_type, f.score) for f in findings]}"
        )

    @pytest.mark.parametrize("text", DOB_NON_PII_CASES)
    def test_generic_dates_not_flagged_as_dob(self, analyzer, text):
        _result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        dob_findings = [f for f in findings if f.entity_type == "DATE_OF_BIRTH"]
        assert not dob_findings, (
            f"Generic date should not be flagged as DOB: {text!r}, got {[(f.entity_type, f.score) for f in dob_findings]}"
        )


ADDRESS_CASES = [
    ("addr_us_full",        "Patient lives at 123 Main Street, Springfield IL 62704.",   "123 Main Street"),
    ("addr_fr_avenue",      "Address: 45 Avenue des Champs-Élysées, 75008 Paris.",       "45 Avenue"),
    ("addr_uk_baker",       "Home: 221B Baker Street, London",                            "221B Baker Street"),
    ("addr_de_postnumber",  "Bahnhofstrasse 12, 8001 Zürich.",                            "Bahnhofstrasse 12"),
    ("addr_lu_rue",         "Rue de la Gare 25, 1611 Luxembourg",                         "Rue de la Gare 25"),
]


class TestStreetAddressDetection:
    @pytest.mark.parametrize("case_id,text,fragment", ADDRESS_CASES)
    def test_street_address_masked(self, analyzer, case_id, text, fragment):
        result, _findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert fragment not in result, (
            f"[{case_id}] Street address fragment {fragment!r} leaked in: {result!r}"
        )


DRIVER_LICENSE_CASES = [
    ("dl_with_label",       "My driver license number is D12345678 issued in California."),
    ("dl_state_context",    "Driver's license D1234567 from New York expires soon."),
]

PASSPORT_CASES = [
    ("passport_with_label", "My passport number is A12345678 issued recently."),
    ("passport_us",         "US passport 123456789 reported lost yesterday."),
]


class TestDriverLicensePassport:
    # Recognizer types acceptable for masking a driver-licence-shaped
    # alphanumeric.  Either Presidio's US_DRIVER_LICENSE or our broader
    # LU_PASSPORT (letter + 7-9 digits) is sufficient — both result in the
    # fragment being replaced.  The security goal is masking, not a specific
    # entity label.
    _LICENCE_ENTITIES = {"US_DRIVER_LICENSE", "LU_PASSPORT", "US_PASSPORT"}

    @pytest.mark.parametrize("case_id,text", DRIVER_LICENSE_CASES)
    def test_driver_license_masked(self, analyzer, case_id, text):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert findings, f"[{case_id}] No entity found in: {text!r}"
        for r in findings:
            if r.entity_type in self._LICENCE_ENTITIES:
                assert text[r.start:r.end] not in result, (
                    f"[{case_id}] License fragment {text[r.start:r.end]!r} leaked in: {result!r}"
                )
                return
        pytest.fail(f"[{case_id}] No licence-like finding; got {[(f.entity_type, f.score) for f in findings]}")

    @pytest.mark.parametrize("case_id,text", PASSPORT_CASES)
    def test_passport_masked(self, analyzer, case_id, text):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert findings, f"[{case_id}] No entity found in: {text!r}"
        for r in findings:
            if r.entity_type in {"US_PASSPORT", "LU_PASSPORT"}:
                assert text[r.start:r.end] not in result, (
                    f"[{case_id}] Passport fragment {text[r.start:r.end]!r} leaked in: {result!r}"
                )
                return
        pytest.fail(f"[{case_id}] No passport finding; got {[(f.entity_type, f.score) for f in findings]}")

    def test_hex_colour_not_flagged_as_driver_license(self, analyzer):
        # The Presidio US_DRIVER_LICENSE recognizer fires on any 6-alphanumeric
        # string at score 0.3.  The score_threshold must filter that out so
        # non-PII hex colours and SKUs survive unchanged.
        text = "Hex colour: #ff5733"
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert not findings, f"Unexpected finding(s) on hex colour: {[(f.entity_type, f.score) for f in findings]}"
        assert result == text


class TestSSNDetection:
    @pytest.mark.parametrize("case_id,text", SSN_CASES)
    def test_us_ssn_masked(self, analyzer, case_id, text):
        result, findings = _anonymize_text(text, analyzer, EntityRegistry())
        assert findings, f"[{case_id}] No entity found in: {text!r}"
        for r in findings:
            assert text[r.start:r.end] not in result, (
                f"[{case_id}] SSN fragment {text[r.start:r.end]!r} survived in: {result!r}"
            )

    @pytest.mark.parametrize("case_id,text", LU_CCSS_CASES)
    def test_lu_ccss_masked(self, analyzer, case_id, text):
        result, _ = _anonymize_text(text, analyzer, EntityRegistry())
        assert "1985032512345" not in result, (
            f"[{case_id}] LU CCSS 13-digit identifier leaked: {result!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Column-name PII policy (Phase 5 of the column-aware layer).
#
# These tests pin the contract that motivated the entire column-policy work:
# Presidio's row-by-row PERSON recognizer alone is too weak — it misses
# `Jimmy`, `Michael`, `Anna` in a `first_name` column.  The column-policy
# layer aggregates evidence across rows (presidio-structured) and falls back
# to spaCy embedding similarity on the column name, so the masking decision
# can be made for the WHOLE column.
# ─────────────────────────────────────────────────────────────────────────────


# Pseudonymizer stub used in the locked tests: deterministic, joinable, and
# trivially inspectable in assertions.  Production deployments use the
# Key Vault RSA-bound pseudonymizer; the contract this layer relies on is
# only that equal inputs produce equal outputs.
class _LockedTestPseudonymizer:
    def __call__(self, value):
        return f"HASH<{value}>"


COLUMN_NAME_POLICY_CASES = [
    # ── Person-name columns across all four supported languages ────────
    ("first_name_en", "first_name", ["Jimmy", "Michael", "Anna"],            "PERSON",       ACTION_TOKENIZE),
    ("first_name_fr", "prenom",     ["Jacques", "Herve", "Marie"],           "PERSON",       ACTION_TOKENIZE),
    ("last_name_de",  "nachname",   ["Mueller", "Schmidt", "Weber"],         "PERSON",       ACTION_TOKENIZE),
    ("name_lb",       "numm",       ["Jean Kohnen", "Anna Weber"],           "PERSON",       ACTION_TOKENIZE),
    # ── Contact columns ────────────────────────────────────────────────
    ("email_col",     "email",      ["a@x.com", "b@y.com", "c@z.com"],       "EMAIL_ADDRESS", ACTION_TOKENIZE),
    # ── Identifier columns ─ hash via the pseudonymizer ─────────────────
    ("customer_id",   "customer_id", ["CUST-001", "CUST-002", "CUST-003"],   None,           ACTION_HASH),
    # ── Cryptic column name, PII *values* (B1 catches it) ──────────────
    ("cryptic_value", "c47",        ["alice@x.com", "bob@y.com", "eve@z.com"], "EMAIL_ADDRESS", ACTION_TOKENIZE),
]

COLUMN_NAME_POLICY_FALSE_POSITIVE_CASES = [
    # Plain free-text columns must NOT be tokenised/hashed by the policy
    # layer — they continue through the row-by-row scan as ACTION_SCAN.
    ("notes_column",  "notes",      ["no issues here", "follow up scheduled", "all good"]),
    ("category",      "category",   ["Electronics", "Home", "Sports"]),
    ("description",   "description",["small product", "large widget", "blue gadget"]),
]


class TestColumnNamePIIPolicy:
    """The column-policy layer must mask whole columns based on column name
    OR aggregate value evidence — without needing per-cell NER hits."""

    @pytest.mark.parametrize(
        "case_id,column,values,expected_entity,expected_action",
        COLUMN_NAME_POLICY_CASES,
    )
    def test_column_policy_masks_known_pii(
        self, analyzer, case_id, column, values, expected_entity, expected_action,
    ):
        df = pd.DataFrame({column: values})
        policies = classify_pii_columns(df, analyzer=analyzer)

        assert column in policies, (
            f"[{case_id}] Column {column!r} was not classified at all "
            f"(policies={policies})"
        )
        policy = policies[column]
        assert policy.action == expected_action, (
            f"[{case_id}] Column {column!r} got action={policy.action!r}, "
            f"expected {expected_action!r}. Source: {policy.source}, "
            f"entity_type: {policy.entity_type}"
        )
        if expected_entity is not None:
            assert policy.entity_type == expected_entity, (
                f"[{case_id}] Column {column!r} classified as "
                f"{policy.entity_type!r}, expected {expected_entity!r}"
            )

        registry = EntityRegistry()
        pseudonymizer = _LockedTestPseudonymizer()
        masked, stats = apply_column_policies(
            df, policies, registry=registry, pseudonymizer=pseudonymizer,
        )

        # Every original value must be absent from the masked column.
        for original in values:
            assert original not in masked[column].astype(str).tolist(), (
                f"[{case_id}] Original value {original!r} leaked in masked "
                f"column: {masked[column].tolist()}"
            )

        # Equal inputs must map to equal outputs (joinability under HASH,
        # consistent tokenisation under TOKENIZE).
        if values.count(values[0]) > 1 or True:
            paired_df = pd.DataFrame({column: values + [values[0]]})
            paired_policies = classify_pii_columns(paired_df, analyzer=analyzer)
            paired_masked, _ = apply_column_policies(
                paired_df, paired_policies,
                registry=registry,
                pseudonymizer=pseudonymizer,
            )
            assert (
                paired_masked[column].iloc[0] == paired_masked[column].iloc[-1]
            ), (
                f"[{case_id}] Identical inputs produced different outputs — "
                f"not deterministic: {paired_masked[column].tolist()}"
            )

    @pytest.mark.parametrize(
        "case_id,column,values",
        COLUMN_NAME_POLICY_FALSE_POSITIVE_CASES,
    )
    def test_free_text_columns_not_force_masked(
        self, analyzer, case_id, column, values,
    ):
        df = pd.DataFrame({column: values})
        policies = classify_pii_columns(df, analyzer=analyzer)
        assert column in policies, f"[{case_id}] No policy for {column!r}"
        policy = policies[column]
        assert policy.action in (ACTION_SCAN, ACTION_BIN), (
            f"[{case_id}] Free-text column {column!r} should not be "
            f"tokenised/hashed by the column-policy layer, but got "
            f"action={policy.action!r} (entity={policy.entity_type!r}, "
            f"source={policy.source!r}, score={policy.score:.2f})"
        )

    def test_jimmy_michael_leak_closed(self, analyzer):
        """The original bug report: a `first_name` column with bare given
        names (Jimmy, Michael, Anna) — Presidio's per-cell PERSON
        recognizer misses them, but the column-policy layer must mask
        every value because the column NAME identifies the entity type
        (or presidio-structured aggregates enough weak signals)."""
        df = pd.DataFrame({"first_name": ["Jimmy", "Michael", "Anna", "Bob", "Carol"]})
        policies = classify_pii_columns(df, analyzer=analyzer)
        assert policies["first_name"].action == ACTION_TOKENIZE
        assert policies["first_name"].entity_type == "PERSON"

        registry = EntityRegistry()
        masked, _ = apply_column_policies(
            df, policies, registry=registry,
            pseudonymizer=_LockedTestPseudonymizer(),
        )
        for original in ["Jimmy", "Michael", "Anna", "Bob", "Carol"]:
            assert original not in masked["first_name"].tolist(), (
                f"Personal name {original!r} leaked through column-policy "
                f"layer: {masked['first_name'].tolist()}"
            )
        # Every value should now be a PERSON_N token.
        for value in masked["first_name"]:
            assert value.startswith("PERSON_"), (
                f"Expected PERSON_N token, got {value!r}"
            )
