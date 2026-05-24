# Learn more

Deeper documentation that doesn't belong on the main README. Read this once you have the pipeline running end-to-end and want to extend it or understand the special-case data paths.

---

## Detection layers

The pipeline is organised as a defence-in-depth stack — each layer narrows what remains to be scanned by the next.

1. **Column classification (`app/classification.py`)** — runs the existing dtype/cardinality heuristics first, then a three-tier PII column classifier:
    - **Tier A — Purview metadata.** When `PURVIEW_ACCOUNT_NAME` is set, the existing Purview infrastructure also returns per-column classifications (e.g. `MICROSOFT.PERSONAL.NAME` → `PERSON`). A small static map (`PURVIEW_TYPE_TO_ENTITY`) translates Microsoft's canonical type names — engineers don't extend it; Microsoft does.
    - **Tier B1 — `presidio-structured` value sampling.** Samples values from each column, runs the standard Presidio analyser per cell, and votes on the dominant entity per column. Catches columns whose **values** are strong PII signals even when the column name is cryptic (`c47`, `field_12`).
    - **Tier B2 — spaCy embedding similarity on the column name.** Compares each unclassified column name against per-language concept anchors (`CONCEPT_SEEDS` — 11 concepts × 4 languages, in code). Catches `first_name` / `prenom` / `Vorname` / `numm` columns whose values alone wouldn't trigger Presidio NER (bare given names like `Jimmy` / `Michael` score below threshold per cell).
    - **Tier C — fallback to row-by-row scan.** Columns no tier could place are tagged `FREE_TEXT` and go through layer 3 below.
2. **Column-policy application (`apply_column_policies`)** — every column with a Tier A/B1/B2 classification is masked WHOLE-COLUMN before per-cell scanning:
    - Identifier-like entities (IDs, IBANs, credit cards, SSN, …) → **hashed** via the Key-Vault-bound pseudonymizer (deterministic, joinable across runs).
    - Direct identifiers and Art. 9 / Art. 10 entities (PERSON, EMAIL_ADDRESS, RELIGION, …) → **tokenised** through the `EntityRegistry` (`PERSON_0`, `EMAIL_ADDRESS_1`, …).
    - DATE_OF_BIRTH / DATE_TIME → **deferred** to the existing temporal-binning layer.
    - FREE_TEXT → **deferred** to the next layer.
3. **Per-cell Presidio + spaCy scan (`anonymize_dataframe`)** — the row-by-row scanner only runs on columns still flagged as free text. It uses:
    - Presidio built-in recognizers (CREDIT_CARD, IBAN_CODE, US_SSN, EMAIL, PHONE, URL, IP_ADDRESS, …).
    - A declarative custom-recognizer registry (`RECOGNIZERS` in `app/recognizers.py`) covering 20+ EU/LU/B2B identifier shapes (LU_CCSS, LU_PASSPORT, SALARY, EU_VAT, NATIONAL_TAX_ID, SWIFT_BIC, INSURANCE_POLICY, VEHICLE_PLATE, BOOKING_REF, HEALTH_INSURANCE, POSTAL_CODE, COURT_CASE, INVOICE_NUMBER, CONTRACT_NUMBER, CUSTOMER_EMPLOYEE_ID, …) with per-entity context lists and validators (e.g. `_IPv6Recognizer` uses `ipaddress.ip_address` to filter HH:MM:SS clock false positives).
    - A semantic-concept recognizer (`_SemanticConceptRecognizer`) that covers GDPR Art. 9 / Art. 10 categories (HEALTH_CONDITION, ETHNICITY, RELIGION, SEXUAL_ORIENTATION, TRADE_UNION, CRIMINAL_RECORD) through spaCy token-level embedding similarity + rapidfuzz Levenshtein-1/2 fallback for typos. **No `.txt` keyword files** — only a tiny `SEMANTIC_CONCEPT_SEEDS` dict with ~6–10 concept anchors per category per language.
    - Per-entity score thresholds (`PRESIDIO_SCORE_THRESHOLDS`) — high-precision recognizers (CREDIT_CARD, IBAN_CODE) accept all findings; context-required recognizers stay at the conservative 0.4 default.
4. **Residual safety net (`validate_residual_pii`)** — scans every text cell of the resulting DataFrame for any remaining email, phone, IBAN, credit-card, or IPv4/v6 shape. Any hit aborts the run; the target is never written.
5. **Audit persistence (`AuditDB`)** — every run, every table, every column-policy decision, and the final entity counts are persisted to PostgreSQL.

---

## GPS trajectory data

Tables that contain GPS coordinates, a speed column, and a timestamp column are treated as **trajectory data** and follow a separate path:

1. Addresses are NLP-anonymized (names, locations stripped from free-text columns).
2. Individual rows are **aggregated** into `(grid cell × hour of day × day of week)` speed statistics — no vehicle identifiers or raw timestamps survive.
3. Cells with fewer than `K_ANONYMITY_MIN` pings are suppressed.

The resulting output contains only `avg_speed_kmh`, `p50_speed_kmh`, `p85_speed_kmh`, and `ping_count` per cell/time slot — safe for business analytics and external LLM consumption.

Non-trajectory GPS tables (no speed column) have coordinates rounded to `GPS_PRECISION` decimal places (default `2`, about 1 km for city data) and timestamps floored to day before row-level k-anonymity is applied.

---

## Extending detection coverage

The whole pipeline is designed so that **adding a new PII category does not require maintaining a keyword list**. The two extension points:

### Adding a new structured / regex-based PII category

Append one row to the `RECOGNIZERS` tuple in `app/recognizers.py`:

```python
RecognizerSpec(
    entity="MY_NEW_ENTITY",
    patterns=(
        {"name": "my_pattern", "regex": r"\b...\b", "score": 0.5},
    ),
    context=("keyword", "label", "synonym"),
    languages=("en", "fr", "de", "lb"),
    validator=my_optional_validator,  # returns True to DISCARD a match
),
```

Add `MY_NEW_ENTITY` to the appropriate group in `ENTITY_GROUPS` (`app/anonymization.py`) and an entry in `DEFAULT_ENTITY_ACTIONS` (`app/classification.py`) saying whether to hash, tokenise, bin, or scan.

### Adding a new Art. 9 / Art. 10 semantic category

Append one entry to `SEMANTIC_CONCEPT_SEEDS` in `app/recognizers.py`:

```python
"MY_SENSITIVE_CATEGORY": {
    "en": ("anchor1", "anchor2", "anchor3"),
    "fr": ("ancre1", "ancre2"),
    "de": ("anker1", "anker2"),
    "lb": ("anker1",),
},
```

That's it. 6–10 anchor words per category per language is enough for spaCy's `_lg` GloVe vectors to cluster related vocabulary, and the rapidfuzz fallback catches typos / OOV forms automatically. No lookup tables, no per-region word lists, no maintenance of synonyms / declensions.

### Adding a new column-name semantic concept

Append one entry to `CONCEPT_SEEDS` in `app/classification.py` with one seed phrase per language. The column-policy classifier (Tier B2) will pick up columns whose names are semantically close to the new seed.

---

## Test contract

The test suite in `tests/test_anonymization.py` is structured as a regression lock. The PII detection tests, the column-policy tests, and the multilingual keyword tests all assert the **masking** outcome (the original fragment must not survive in the result) as the security-critical contract.

See `CLAUDE.md` / `AGENTS.md` for the full list of locked test classes and parametrize fixtures. Briefly: any change touching `tests/test_anonymization.py` should ADD cases, never remove or weaken existing ones — and when a locked test fails, fix the recognizer, not the test.
