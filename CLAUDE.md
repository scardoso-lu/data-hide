# Repository Guidelines

## Release Status

The project has not been released. Do not add backward-compatibility shims, overloaded method signatures, fallback code paths, or migration helpers unless explicitly instructed.

## Project Structure & Module Organization

This repository contains a stateless Python pipeline for anonymizing Microsoft Fabric OneLake Delta tables. The main application lives in `app/`, with `main.py` as a thin shim entrypoint. Tests live in `tests/` and are split by behavior: orchestration, anonymization, audit database, alerts, key vault, and helpers. Runtime packaging is defined by `Dockerfile`, local orchestration by `docker-compose.yml`, and dependencies by `pyproject.toml` plus the committed `uv.lock` lock file.

## Build, Test, and Development Commands

The project uses [uv](https://docs.astral.sh/uv/) as its package manager. uv reads `.python-version` and `pyproject.toml`, manages the `.venv` directory automatically, and resolves against the committed `uv.lock`.

- `uv sync`: create the venv and install runtime + dev dependencies from `uv.lock`.
- `uv sync --no-dev`: production install — runtime deps only.
- `uv run python -m spacy download en_core_web_lg && uv run python -m spacy download fr_core_news_lg && uv run python -m spacy download de_core_news_lg`: install the required NLP models (English, French, German/Luxembourgish) for full anonymization tests and local runs.
- `uv run pytest`: run the test suite configured by `pytest.ini`.
- `uv run pytest -m "not requires_spacy and not slow"`: skip spaCy-dependent tests.
- `uv lock --upgrade`: refresh `uv.lock` to the latest versions within the ranges declared in `pyproject.toml`.
- `docker compose up --build`: build and run the pipeline with local PostgreSQL.
- `docker build -t fabric-pii-pipeline:latest .`: build the standalone container image.

Commit `uv.lock` whenever `pyproject.toml` changes — the Dockerfile uses `uv sync --frozen` and will fail the build if the lock is stale.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints where they clarify function contracts, and focused functions with explicit names. Keep constants in uppercase, environment variable names uppercase, and tests named `test_<behavior>.py`. Prefer structured data handling with pandas and existing helper functions over ad hoc parsing. Keep runtime behavior stateless; do not add container-local file writes.

## Testing Guidelines

The project uses `pytest` with `tests` as the configured test root. Mark tests that require the full spaCy model with `requires_spacy` or `slow` as appropriate. Mock external I/O in unit and orchestration tests: Delta Lake, Azure credentials, PostgreSQL, Purview, Key Vault, and alert webhooks. Run `uv run pytest` before submitting changes.

## PII Detection Test Lock (frozen contract)

The test classes and parametrize fixtures listed below in `tests/test_anonymization.py` encode regulatory-grade detection guarantees built through 25 TDD iterations covering GDPR Art. 9/10 special categories, B2B/B2C identifiers, and typo tolerance. Their **inputs and assertions are frozen**. Each input is a documented regression: it was observed leaking on a previous version of the pipeline, and the assertion is the minimum guarantee that proves the leak was closed.

**Locked test classes** (`tests/test_anonymization.py`):

`TestPIIDetection`, `TestNoPIIPassthrough`, `TestNonStringPassthrough`, `TestDataFrameAnonymization`, `TestEntityRegistry`, `TestJSONAndDictAnonymization`, `TestSSNDetection`, `TestDriverLicensePassport`, `TestStreetAddressDetection`, `TestDateOfBirthDetection`, `TestSalaryDetection`, `TestHealthConditionDetection`, `TestNRPDetection`, `TestMedicalIDDetection`, `TestTaxIDDetection`, `TestPhoneEdgeFormats`, `TestIPv6EdgeFormats`, `TestContractNumberDetection`, `TestNationalTaxIDDetection`, `TestSwiftBICDetection`, `TestInsurancePolicyDetection`, `TestVehiclePlateDetection`, `TestBookingRefDetection`, `TestCustomerEmployeeIDDetection`, `TestHealthInsuranceDetection`, `TestArt9Art10Detection`, `TestPostalCodeDetection`, `TestCourtCaseDetection`, `TestInvoiceNumberDetection`, `TestHealthConditionTypos`, `TestIdentifierLabelTypos`, `TestAddressLabelTypos`, `TestArt9TyposMasked`, `TestEmailPhoneLabelTypos`, `TestMultilingualKeywordDetection`, `TestColumnNamePIIPolicy`.

**Locked parametrize fixtures** (module-level lists in the same file):

`PII_CASES`, `NON_PII_STRINGS`, `NON_STRING_VALUES`, `SSN_CASES`, `LU_CCSS_CASES`, `DRIVER_LICENSE_CASES`, `PASSPORT_CASES`, `ADDRESS_CASES`, `DOB_CASES`, `DOB_NON_PII_CASES`, `SALARY_CASES`, `SALARY_NON_PII_CASES`, `HEALTH_CASES`, `NRP_CASES`, `MEDICAL_ID_CASES`, `TAX_ID_CASES`, `PHONE_EXTENSION_CASES`, `IPV6_CASES`, `CONTRACT_CASES`, `CONTRACT_NON_PII_CASES`, `NATIONAL_TAX_ID_CASES`, `NATIONAL_TAX_ID_NON_PII`, `SWIFT_BIC_CASES`, `INSURANCE_CASES`, `VEHICLE_PLATE_CASES`, `BOOKING_REF_CASES`, `CUSTOMER_ID_CASES`, `HEALTH_INSURANCE_CASES`, `ART9_ART10_CASES`, `POSTAL_CODE_CASES`, `POSTAL_CODE_NON_PII`, `COURT_CASE_CASES`, `INVOICE_CASES`, `HEALTH_TYPO_CASES`, `LABEL_TYPO_CASES`, `ADDRESS_TYPO_CASES`, `ART9_TYPO_CASES`, `EMAIL_PHONE_TYPO_CASES`, `MULTILINGUAL_KEYWORD_CASES`, `COLUMN_NAME_POLICY_CASES`, `COLUMN_NAME_POLICY_FALSE_POSITIVE_CASES`.

**Rules for any change touching these tests:**

- **Never weaken an assertion.** `assert fragment not in result` is the minimum bar; replacing it with `assert finding is not None` or `assert len(findings) > 0` turns a leak guard into a detection-only check and is treated as a regression.
- **Never delete or shorten an input case.** Every entry is a real-world leak that was observed and closed. Removing one re-opens that exposure.
- **Never narrow a non-PII / false-positive guard.** `NON_PII_STRINGS`, `*_NON_PII`, `*_NON_PII_CASES` defend against over-masking of legitimate values (SKUs, order codes, hex colours, generic dates, prices, postcodes-without-context, …). Removing entries causes silent data damage.
- **Adding is allowed; subtracting is not.** New entries in any locked fixture, and new test classes for new PII categories, are encouraged. Use a new fixture/class rather than modifying an existing entry.
- **If a locked test fails, fix the recognizer — not the test.** A failure here means a recognizer in `app/recognizers.py` or a setting in `app/anonymization.py` was changed in a way that re-introduced a leak (or a false positive). Restore the recognizer behaviour or add a new pattern; do not edit the input or the assertion.
- **Treat `PRESIDIO_SCORE_THRESHOLD` and `GDPR_ENTITIES` in `app/anonymization.py` as part of this contract.** Raising the threshold, or removing an entity type from the list, must be accompanied by a full `uv run pytest` pass — many context-required recognizers (NATIONAL_TAX_ID, BOOKING_REF, POSTAL_CODE, CONTRACT_NUMBER, SWIFT_BIC, DATE_OF_BIRTH, SALARY) rely on the current threshold for their score boost to clear.
- **Column-policy layer** in `app/classification.py` (`classify_pii_columns`, `apply_column_policies`, `CONCEPT_SEEDS`, `DEFAULT_ENTITY_ACTIONS`, `PURVIEW_TYPE_TO_ENTITY`) is part of this contract. The three classification tiers (Purview → presidio-structured → spaCy embedding similarity) are load-bearing — removing a tier means leaking known regressions like the `first_name` / `Jimmy` case. New env vars `ENABLE_PRESIDIO_STRUCTURED` (default on), `ENABLE_COLUMN_POLICY` (default on), and `COLUMN_SIMILARITY_THRESHOLD` (default 0.55) are operator escape hatches; do not change their defaults without re-running the full suite.
- **A truly-incorrect locked case can be amended only with explicit user approval**, in a PR whose description names the case and explains why the previous expectation was wrong. Do not silently rewrite.

## Commit & Pull Request Guidelines

Git history uses concise imperative commits, including Conventional Commit style such as `feat: identifier hashing and JSON/nested document support`. Prefer `feat:`, `fix:`, `test:`, or `docs:` when the scope is clear. Pull requests should describe behavior changes, list verification commands, mention security or data-handling impact, and link related issues. Include screenshots only for documentation or UI-adjacent changes.

## Security & Configuration Tips

Never commit `.env` or live credentials. Use `.env.example` for configuration shape. Keep Azure, Purview, PostgreSQL, and webhook secrets in environment variables only. Preserve the non-root, no-runtime-files container posture described in the README.
