"""Column-name policy used by the residual-PII safety net in `anonymization`.

These token sets are matched against the lowercased, non-alphanumeric-split
column name to either suppress or admit specific residual-PII checks.  They
live in their own module so deployments can override them (extend the set,
add new tokens for an internal naming convention) without touching the
recognizers in `anonymization` or `recognizers`.

Each set is a plain `set[str]` of single tokens (no regex, no phrases).  Add
new entries by extending the set in-place at import time, or by replacing
the binding from a configuration shim.
"""

from __future__ import annotations


# Column names that should suppress residual-PII checks for top-level
# scalars.  These are operational metadata columns whose values look like
# IBANs / phone numbers / credit cards but are intentionally structured (e.g.
# "annonces_20260523144520.csv" looks like a phone-number digit run).
RESIDUAL_METADATA_COLUMN_TOKENS: set[str] = {
    "dataset",
    "file",
    "filename",
    "key",
    "modified",
    "path",
    "record",
    "resource",
    "source",
    "update",
    "updated",
}

# Column-name tokens that explicitly *allow* a given residual-PII entity
# even when the value passes through other filters — e.g. a column named
# "phone" should always be checked for phone numbers regardless of whether
# the value looks numeric.
PHONE_COLUMN_TOKENS: set[str] = {"cell", "fax", "mobile", "phone", "tel", "telephone"}
CREDIT_CARD_COLUMN_TOKENS: set[str] = {"card", "cc", "creditcard", "pan"}
IBAN_COLUMN_TOKENS: set[str] = {"account", "bank", "iban"}
