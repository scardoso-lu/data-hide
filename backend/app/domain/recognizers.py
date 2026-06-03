"""Custom Presidio recognizers for GDPR special categories not covered by Presidio's built-ins.

This module exposes a single declarative registry (`RECOGNIZERS`) of
`RecognizerSpec` rows.  Each row carries everything Presidio needs to build
the corresponding `PatternRecognizer`: the entity name, regex patterns,
context words (including typo variants), supported languages, and an
optional validator callable that filters syntactic false positives.

`install_custom_recognizers(registry, nlp_engine)` walks the list and
installs one `PatternRecognizer` per (spec × language) combination — adding
a new entity type is now a single row.  GDPR Art. 9 / Art. 10 special
categories that used to be hand-curated deny lists in `app/keywords/*.txt`
are now driven by `SEMANTIC_CONCEPT_SEEDS` + a token-level embedding
similarity recognizer (with rapidfuzz fallback for typos).  No external
keyword files to maintain.

The validator slot generalises what used to be the bespoke `_IPv6Recognizer`
subclass: any spec can supply `validator=<callable>` and the loader builds a
subclass that delegates `invalidate_result` to it.  The callable receives the
matched text and must return True to *discard* the match (Presidio's
``invalidate_result`` semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
from typing import Any, Callable


# ─────────────────────────────────────────────────────────────────────────────
# Validators (used in the `validator` slot of RecognizerSpec).
# A validator returns True when the matched text must be DISCARDED.
# ─────────────────────────────────────────────────────────────────────────────

def _not_ipv6(pattern_text: str) -> bool:
    """Discard a colon-hex match unless `ipaddress` accepts it as IPv6."""
    try:
        return not isinstance(ipaddress.ip_address(pattern_text), ipaddress.IPv6Address)
    except ValueError:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# RecognizerSpec — the unit of declaration for a custom recognizer.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RecognizerSpec:
    entity: str
    patterns: tuple[dict, ...] = ()
    deny_list: tuple[str, ...] = ()
    context: tuple[str, ...] = ()
    languages: tuple[str, ...] = ("en", "fr", "de", "lb")
    validator: Callable[[str], bool] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Registry.  Adding a new entity type = appending one row here.
# Order is informational only.  Pattern scores are intentionally low for
# context-required recognizers (the analyzer applies a per-entity score
# threshold; the LemmaContextAwareEnhancer lifts low scores when a context
# token appears nearby).
# ─────────────────────────────────────────────────────────────────────────────

RECOGNIZERS: tuple[RecognizerSpec, ...] = (
    # ── DATE OF BIRTH ──────────────────────────────────────────────────────
    RecognizerSpec(
        entity="DATE_OF_BIRTH",
        patterns=(
            {"name": "dob_iso", "regex": r"\b(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])\b", "score": 0.1},
            {"name": "dob_dmy", "regex": r"\b(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b", "score": 0.1},
            {
                "name": "dob_written",
                "regex": r"(?i)\b(?:(?:0?[1-9]|[12]\d|3[01])\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|januar|februar|märz|maerz|april|mai|juni|juli|august|september|oktober|november|dezember)\s+(?:0?[1-9]|[12]\d|3[01])?,?\s*(?:19|20)\d{2}\b",
                "score": 0.1,
            },
            {"name": "dob_year_only", "regex": r"\b(?:19|20)\d{2}\b", "score": 0.05},
        ),
        context=(
            # Presidio's LemmaContextAwareEnhancer compares the lemma of each
            # text token against this list AS-IS.  spaCy lemmatises "Born" →
            # "bear", so the lemma form must be present here.
            "born", "bear", "birth", "birthday", "dob",
            "naissance", "naître", "naitre", "né", "nee",
            "geboren", "gebären", "gebaren", "geburt", "geburtsdatum", "geburtstag",
        ),
    ),
    # ── IPv6 ───────────────────────────────────────────────────────────────
    RecognizerSpec(
        entity="IP_ADDRESS",
        patterns=(
            {
                "name": "ipv6_general",
                "regex": r"\b[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}\b",
                "score": 0.85,
            },
        ),
        context=("ip", "ipv6", "address", "host", "source", "destination"),
        validator=_not_ipv6,
    ),
    # ── PHONE EXTENSION ────────────────────────────────────────────────────
    RecognizerSpec(
        entity="PHONE_NUMBER",
        patterns=(
            {
                "name": "phone_extension",
                "regex": r"(?i)\b(?:extn|exten|extension|extnsion|extentn|ext|poste|pste|psote|postee|x)\.?\s*\d{1,5}\b",
                "score": 0.5,
            },
        ),
        context=(
            "phone", "tel", "telephone", "call", "bureau", "office", "extension", "poste",
            "phne", "fone", "telfon", "telefone", "buro", "bureu", "cntact", "emial",
        ),
    ),
    # ── LU CCSS ────────────────────────────────────────────────────────────
    RecognizerSpec(
        entity="LU_CCSS",
        patterns=(
            {
                "name": "lu_ccss_13_digits",
                "regex": r"\b(?:18[5-9]\d|19\d{2}|20\d{2})(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{5}\b",
                "score": 0.6,
            },
        ),
        context=("ccss", "matricule", "nummer", "national", "luxembourg", "sécurité"),
    ),
    # ── LU PASSPORT / DRIVER LICENCE ───────────────────────────────────────
    RecognizerSpec(
        entity="LU_PASSPORT",
        patterns=(
            {"name": "lu_passport_letter_7digits", "regex": r"(?-i:\b[A-Z]\d{7,9}\b)", "score": 0.35},
        ),
        context=(
            "passport", "passeport", "passnummer", "reisepass",
            "driver", "drivers", "license", "licence", "permis", "führerschein",
            "passprt", "passpot", "passpotr", "passpott", "psasport",
            "drivr", "drivrs", "drvr", "drver",
            "lcense", "lisence", "licens",
        ),
    ),
    # ── SALARY ─────────────────────────────────────────────────────────────
    RecognizerSpec(
        entity="SALARY",
        patterns=(
            {
                "name": "salary_amount_with_unit",
                "regex": r"(?i)(?:€|\$|£|EUR|USD|GBP|CHF)\s?\d{1,3}(?:[ ,.]?\d{3})*(?:[.,]\d+)?(?:\s?[kKmM])?(?:\s?(?:per|/)\s?(?:year|month|annum|hour|yr|mo|hr))?(?=\W|$)",
                "score": 0.35,
            },
            {
                "name": "salary_amount_with_trailing_unit",
                "regex": r"\b\d{1,3}(?:[ ,.]?\d{3})*(?:[.,]\d+)?(?:\s?[kKmM])?\s?(?:€|\$|£|EUR|USD|GBP|CHF)(?:\s?(?:per|/)\s?(?:year|month|annum|hour|yr|mo|hr))?(?=\W|$)",
                "score": 0.35,
            },
        ),
        context=(
            "salary", "salaire", "wage", "income", "earn", "compensation",
            "remuneration", "rémunération", "gehalt", "pay",
            "annual", "annually", "yearly", "monthly", "year",
            "base",
        ),
    ),
    # ── EU VAT ─────────────────────────────────────────────────────────────
    RecognizerSpec(
        entity="EU_VAT",
        patterns=(
            {
                "name": "eu_vat_country_prefix",
                "regex": r"\b(?:LU\d{8}|DE\d{9}|FR[A-HJ-NP-Z0-9]{2}\d{9}|ES[A-Z]\d{7}[A-Z0-9]|IT\d{11}|NL\d{9}B\d{2}|BE0?\d{9,10})\b",
                "score": 0.6,
            },
        ),
        context=("vat", "tva", "ust", "ustid", "btw", "iva", "mwst"),
    ),
    # ── MEDICAL RECORD ─────────────────────────────────────────────────────
    RecognizerSpec(
        entity="MEDICAL_RECORD",
        patterns=(
            {
                "name": "mrn_labelled",
                "regex": r"(?i)\b(?:MRN|PID|chart|dossier|patient)\s*[#:.\-]?\s*[A-Z]?\d{4,}[A-Z0-9-]{0,12}\b",
                "score": 0.6,
            },
        ),
        context=("mrn", "patient", "chart", "dossier", "medical", "hospital", "admit"),
    ),
    # ── STREET ADDRESS ─────────────────────────────────────────────────────
    RecognizerSpec(
        entity="STREET_ADDRESS",
        patterns=(
            {
                "name": "street_address_with_suffix",
                "regex": r"\b\d{1,5}[A-Za-z]?\s+(?:[\wÀ-ÿ'-]+\s+){0,5}(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Lane|Ln\.?|Drive|Dr\.?|Way|Plaza|Place|Pl\.?|Square|Sq\.?|Court|Ct\.?|Rue|Route|Allée|Strasse|Straße|Str\.?|Weg|Platz|Gasse)\b",
                "score": 0.55,
            },
            {
                "name": "street_address_eu_prefix",
                "regex": r"\b(?:Rue|Route|Avenue|Boulevard|Allée|Place|Via|Piazza)\s+(?:[\wÀ-ÿ'-]+\s+){1,5}\d{1,5}[A-Za-z]?\b",
                "score": 0.55,
            },
            {
                "name": "street_address_de_suffix_number",
                "regex": r"\b[\wÀ-ÿ-]{3,}(?:strasse|straße|weg|gasse|platz)\s+\d{1,4}[A-Za-z]?\b",
                "score": 0.55,
            },
        ),
        context=(
            "address", "addresse", "adresse", "anschrift", "domicile", "resides", "lives",
            "adress", "adres", "addres", "addrss", "anschrif", "anschrff",
        ),
    ),
    # ── CONTRACT NUMBER ────────────────────────────────────────────────────
    RecognizerSpec(
        entity="CONTRACT_NUMBER",
        patterns=(
            {
                "name": "contract_ref",
                "regex": r"\b[A-Z0-9]{1,6}[-/][A-Z0-9](?:[A-Z0-9-/]{2,20})\b",
                "score": 0.1,
            },
        ),
        context=(
            "contract", "contrat", "vertrag", "agreement",
            "master", "subscription", "engagement", "convention",
            "contrct", "contractt", "kontrakt", "agreeement", "aggrement",
            "contratct", "contracct",
        ),
    ),
    # ── NATIONAL TAX ID ────────────────────────────────────────────────────
    RecognizerSpec(
        entity="NATIONAL_TAX_ID",
        patterns=(
            {"name": "siren", "regex": r"\b\d{3}\s?\d{3}\s?\d{3}\b", "score": 0.1},
            {"name": "siret", "regex": r"\b\d{14}\b", "score": 0.2},
            {"name": "insee_nir", "regex": r"\b[12]\d{12}\b", "score": 0.2},
            {"name": "steuernummer", "regex": r"\b\d{2,3}/\d{3,4}/\d{4,5}\b", "score": 0.4},
            {"name": "uk_utr", "regex": r"\b\d{10}\b", "score": 0.1},
            {"name": "us_ein", "regex": r"\b\d{2}-\d{7}\b", "score": 0.3},
        ),
        context=(
            "siren", "siret", "steuernummer", "utr", "ein", "nir", "insee",
            "tax", "fiscal", "taxpayer", "identifiant", "identifier",
            "company", "entity", "federal", "register", "registered",
        ),
    ),
    # ── SWIFT / BIC ────────────────────────────────────────────────────────
    RecognizerSpec(
        entity="SWIFT_BIC",
        patterns=(
            {"name": "swift_bic_11", "regex": r"(?-i:\b[A-Z]{6}[A-Z0-9]{5}\b)", "score": 0.1},
            {"name": "swift_bic_8",  "regex": r"(?-i:\b[A-Z]{6}[A-Z0-9]{2}\b)", "score": 0.05},
        ),
        context=(
            "bic", "swift", "bank", "banque", "iban", "wire", "sepa",
            "swft", "swiff", "swifft", "bnak", "bnk", "bic8", "bic11",
        ),
    ),
    # ── INSURANCE POLICY ───────────────────────────────────────────────────
    RecognizerSpec(
        entity="INSURANCE_POLICY",
        patterns=(
            {
                "name": "insurance_policy_labelled",
                "regex": r"(?i)\b(?:policy|policie|police|polizza|versicherungsnummer|police\s+d'assurance)\s*[#:.\-N°nn°]{0,4}\s*[A-Z0-9][A-Z0-9-/]{3,20}\b",
                "score": 0.5,
            },
            {
                "name": "insurance_policy_code",
                "regex": r"(?i)\bPOL[-/][A-Z0-9-]{4,20}\b",
                "score": 0.4,
            },
        ),
        context=(
            "policy", "policie", "police", "polizza", "versicherung",
            "insurance", "assurance", "insured", "claim", "premium",
        ),
    ),
    # ── VEHICLE PLATE ──────────────────────────────────────────────────────
    RecognizerSpec(
        entity="VEHICLE_PLATE",
        patterns=(
            {"name": "plate_lu", "regex": r"(?-i:\b[A-Z]{2}[\s-]?\d{1,5}\b)", "score": 0.1},
            {"name": "plate_fr", "regex": r"(?-i:\b[A-Z]{2}-\d{3}-[A-Z]{2}\b)", "score": 0.5},
            {"name": "plate_uk", "regex": r"(?-i:\b[A-Z]{2}\d{2}\s?[A-Z]{3}\b)", "score": 0.5},
            {"name": "plate_de", "regex": r"(?-i:\b[A-Z]{1,3}-[A-Z]{1,2}\s?\d{1,4}\b)", "score": 0.2},
            {"name": "plate_it", "regex": r"(?-i:\b[A-Z]{2}\s\d{3}\s[A-Z]{2}\b)", "score": 0.5},
        ),
        context=(
            "plate", "plaque", "kennzeichen", "targa", "vehicle", "véhicule",
            "fahrzeug", "veicolo", "license", "immatriculation", "registered",
            "car", "voiture", "auto",
        ),
    ),
    # ── BOOKING REF ────────────────────────────────────────────────────────
    RecognizerSpec(
        entity="BOOKING_REF",
        patterns=(
            {"name": "pnr_6char", "regex": r"(?-i:\b[A-Z0-9]{6}\b)", "score": 0.05},
            {
                "name": "booking_labelled",
                "regex": r"(?i)\b(?:BK|RES|RSV|BOOK|RESERVATION|RESERVE)[-/#:]\s?[A-Z0-9-]{4,16}\b",
                "score": 0.5,
            },
        ),
        context=(
            "pnr", "booking", "reservation", "flight", "passenger", "boarding",
            "réservation", "vol", "passager", "buchung", "ticket",
        ),
    ),
    # ── HEALTH INSURANCE ───────────────────────────────────────────────────
    RecognizerSpec(
        entity="HEALTH_INSURANCE",
        patterns=(
            {"name": "carte_vitale", "regex": r"\b[12]\d{14}\b", "score": 0.4},
            {"name": "kvnr_de", "regex": r"(?-i:\b[A-Z]\d{9}\b)", "score": 0.3},
            {"name": "nhs_number", "regex": r"\b\d{3}\s?\d{3}\s?\d{4}\b", "score": 0.2},
        ),
        context=(
            "carte", "vitale", "nhs", "krankenversicherung", "krankenversichertennummer",
            "kvnr", "health", "santé", "sante", "insurance", "insured",
            "assurance", "mutuelle", "cnam", "social",
        ),
    ),
    # ── POSTAL CODE ────────────────────────────────────────────────────────
    RecognizerSpec(
        entity="POSTAL_CODE",
        patterns=(
            {"name": "postcode_lu", "regex": r"(?-i:\bL[-\s]?\d{4}\b)", "score": 0.5},
            {"name": "postcode_uk", "regex": r"(?-i:\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b)", "score": 0.5},
            {"name": "postcode_nl", "regex": r"(?-i:\b\d{4}\s?[A-Z]{2}\b)", "score": 0.4},
            {"name": "postcode_5digit", "regex": r"\b\d{5}\b", "score": 0.1},
        ),
        context=(
            "postal", "postcode", "zip", "plz", "cp", "codice", "boîte", "boite",
            "address", "adresse", "anschrift", "domicile",
            "adress", "adres", "addres", "addresse", "addrss",
            "anschrif", "anschrff", "anchrift",
            "pstcode", "pstal", "postl", "psot", "pstcd", "postcde",
        ),
    ),
    # ── COURT CASE ─────────────────────────────────────────────────────────
    RecognizerSpec(
        entity="COURT_CASE",
        patterns=(
            {
                "name": "court_case_labelled",
                "regex": r"(?i)\b(?:case|affaire|sache|causa|aktenzeichen|docket|file)\s*(?:no\.?|n[°o]\.?|number|nr\.?|#)?\s*[A-Z0-9][A-Z0-9/.\-]{3,20}\b",
                "score": 0.55,
            },
            {"name": "court_case_year_court", "regex": r"(?-i:\b(?:19|20)\d{2}[-/](?:CV|CR|CIV|CRIM|FAM)[-/]\d{2,6}\b)", "score": 0.6},
            {"name": "court_case_de_format", "regex": r"(?-i:\b\d{1,3}\s[A-Z]{1,3}\s\d{2,5}/\d{2,4}\b)", "score": 0.1},
        ),
        context=(
            "case", "court", "tribunal", "affaire", "judge", "judgment", "verdict",
            "magistrate", "lawsuit", "litigation", "docket", "aktenzeichen",
        ),
    ),
    # ── INVOICE NUMBER ─────────────────────────────────────────────────────
    RecognizerSpec(
        entity="INVOICE_NUMBER",
        patterns=(
            {
                "name": "invoice_labelled",
                "regex": r"(?i)\b(?:invoice|facture|rechnung|fattura|factura)\s*(?:no\.?|n[°o]\.?|number|nr\.?|#)?\s*[A-Z0-9][A-Z0-9/.\-]{2,20}\b",
                "score": 0.5,
            },
            {
                "name": "invoice_code_prefix",
                "regex": r"(?i)\b(?:INV|FACT|RG|FA)[-/]\d{2,4}[-/]?\d{0,6}[A-Z0-9-]*\b",
                "score": 0.45,
            },
        ),
        context=(
            "invoice", "facture", "rechnung", "fattura", "factura", "billing",
            "billed", "due", "payable", "amount",
        ),
    ),
    # ── CUSTOMER / EMPLOYEE ID ─────────────────────────────────────────────
    RecognizerSpec(
        entity="CUSTOMER_EMPLOYEE_ID",
        patterns=(
            {
                "name": "labelled_internal_id",
                "regex": r"(?i)\b(?:cust|customer|client|emp|employee|empl|badge|matricule|personnel|payroll|staff)[-/#:.\s]{1,3}[A-Z]?[-/]?\d{2,5}[A-Z0-9-]{0,15}\b",
                "score": 0.5,
            },
        ),
        context=(
            "customer", "client", "employee", "employé", "badge", "matricule",
            "personnel", "payroll", "staff", "user", "account",
        ),
    ),
    # ── ART. 9 / ART. 10 SPECIAL CATEGORIES ─────────────────────────────────
    # No more `RecognizerSpec` entries here — these six categories are
    # detected by `_SemanticConceptRecognizer` (token-level spaCy embedding
    # similarity + rapidfuzz fuzzy fallback) using the small concept anchors
    # in `SEMANTIC_CONCEPT_SEEDS` below.  No `.txt` files to maintain.
)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic concept anchors for the GDPR Art. 9 / Art. 10 categories.
#
# Each category lists a handful of canonical surface forms per language
# (3-6 typical, NEVER comprehensive).  At engine-build time each anchor is
# embedded through that language's spaCy `_lg` GloVe vectors and used in
# two ways:
#
#   1. Embedding similarity — every token in a text whose vector cosines
#      ≥ SEMANTIC_SIMILARITY_THRESHOLD against an anchor is flagged.  This
#      captures semantically-related terms the engineers never enumerated
#      (e.g. `tumor`, `oncology`, `chemotherapy` all cluster near `cancer`).
#
#   2. Fuzzy fallback — Levenshtein distance ≤ 1 (or whatever the recognizer
#      is configured with) against the anchor surface forms.  This catches
#      typos and OOV variants (`diabetis`, `Catholc`) without anyone
#      having to enumerate them.
#
# Adding a new category = one entry here.  Engineering teams DO NOT have
# to extend lists of column names, typos, or surface variants.
# ─────────────────────────────────────────────────────────────────────────────


SEMANTIC_CONCEPT_SEEDS: dict[str, dict[str, tuple[str, ...]]] = {
    "HEALTH_CONDITION": {
        # Anchor set covers both broad medical vocabulary ("disease") and
        # named conditions / acronyms ("HIV", "PTSD", "AVC") so the
        # embedding catches related vocabulary AND fuzzy fallback handles
        # short OOV acronyms / typos.  "infection" / "infektion" were
        # removed because they embed too close to generic "reference" /
        # "record" in spaCy's vector space.
        "en": ("disease", "diabetes", "cancer", "syndrome",
               "tumor", "Alzheimer", "depression", "anxiety", "asthma",
               "pregnant", "HIV", "PTSD", "AIDS", "stroke"),
        "fr": ("maladie", "diabète", "cancer", "syndrome",
               "tumeur", "Alzheimer", "dépression", "anxiété", "asthme",
               "enceinte", "VIH", "SIDA", "AVC", "trouble"),
        "de": ("krankheit", "diabetes", "krebs", "syndrom",
               "tumor", "Alzheimer", "depression", "angst", "asthma",
               "schwanger", "HIV", "AIDS"),
        "lb": ("krankheet", "diabetis", "kriibs", "syndrom",
               "Alzheimer", "depression", "schwanger", "HIV", "AIDS"),
    },
    "ETHNICITY": {
        "en": ("African", "Asian", "Hispanic", "Caucasian", "Arab", "Latino",
               "Jewish", "Indigenous"),
        "fr": ("africain", "asiatique", "hispanique", "caucasien", "arabe",
               "latino", "juif"),
        "de": ("afrikaner", "asiate", "hispanisch", "kaukasier", "araber",
               "latino", "jüdisch"),
        # Luxembourgish demonyms (Lëtzebuerger, Portugis, …) are OOV in
        # the German model used for `lb`; fuzzy fallback catches them.
        "lb": ("afrikaner", "asiat", "araber", "latino", "Lëtzebuerger",
               "Portugis"),
    },
    "RELIGION": {
        "en": ("Catholic", "Muslim", "Jewish", "Buddhist", "Protestant",
               "Hindu", "Sikh", "Atheist"),
        "fr": ("catholique", "musulman", "juif", "bouddhiste", "protestant",
               "hindou", "athée"),
        "de": ("katholisch", "muslim", "jüdisch", "buddhistisch",
               "protestantisch", "atheist"),
        "lb": ("kathoulesch", "muslim", "jiddesch", "buddhist", "protestant"),
    },
    "SEXUAL_ORIENTATION": {
        "en": ("gay", "lesbian", "bisexual", "transgender", "queer",
               "homosexual", "LGBT"),
        "fr": ("gay", "lesbienne", "bisexuel", "transgenre", "homosexuel",
               "LGBT"),
        "de": ("schwul", "lesbisch", "bisexuell", "transgender",
               "homosexuell", "LGBT"),
        "lb": ("schwul", "lesbesch", "bisexuell", "transgender", "LGBT"),
    },
    "TRADE_UNION": {
        # Confederation acronyms (CGT, DGB, OGBL, …) are language-neutral
        # tokens — duplicate them across all four seed sets so the fuzzy
        # fallback catches them regardless of which spaCy model analyses
        # the cell.  Generic words ("union", "syndicat") carry the
        # embedding signal.
        "en": ("union", "syndicate", "labor", "CGT", "CFDT", "DGB", "OGBL",
               "LCGB", "ALEBA"),
        "fr": ("syndicat", "CGT", "CFDT", "FO", "OGBL", "LCGB", "ALEBA"),
        "de": ("gewerkschaft", "DGB", "Verdi", "OGBL", "LCGB", "ALEBA"),
        "lb": ("gewerkschaft", "OGBL", "LCGB", "ALEBA", "CGT", "DGB"),
    },
    "CRIMINAL_RECORD": {
        # "sentence" was removed — embeds too close to "reference" /
        # "order" in spaCy's vector space (grammatical-sentence sense).
        # The verbal forms ("convicted", "imprisonment", "arrest") are
        # unambiguously criminal-justice.
        "en": ("convicted", "felony", "imprisonment", "arrest",
               "criminal", "prison", "incarceration"),
        "fr": ("condamné", "prison", "arrestation", "infraction", "crime",
               "détention", "incarcération"),
        "de": ("verurteilt", "haft", "verbrechen", "vergehen", "gefängnis",
               "haftstrafe"),
        "lb": ("verurteelt", "prisong", "Prisongstrof", "strof",
               "verbrieche", "haft"),
    },
}


# Cosine-similarity threshold at which a token is considered to match a
# concept anchor.  Conservative default so unrelated words don't trigger
# false positives.  Overridable via env `SEMANTIC_SIMILARITY_THRESHOLD`.
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.55


# ─────────────────────────────────────────────────────────────────────────────
# Installer.
# ─────────────────────────────────────────────────────────────────────────────

def _build_validating_class(base_cls: type, validator: Callable[[str], bool]) -> type:
    """Subclass `base_cls` so its `invalidate_result` delegates to `validator`."""
    class _Validated(base_cls):  # type: ignore[misc, valid-type]
        def invalidate_result(self, pattern_text: str) -> bool:
            return validator(pattern_text)
    _Validated.__name__ = f"Validated{base_cls.__name__}"
    return _Validated


def _fuzzy_enabled() -> bool:
    """Opt-in flag: when `ENABLE_FUZZY_TYPO_MATCH=1`, install fuzzy deny-list
    recognizers in addition to the exact-match ones.  Default off so detection
    behaviour stays deterministic for the locked test suite."""
    import os
    return os.environ.get("ENABLE_FUZZY_TYPO_MATCH", "").strip() in {"1", "true", "yes", "on"}


def _build_fuzzy_recognizer_class(base_cls: type) -> type:
    """Construct a Presidio recognizer that fires when a token is within
    Levenshtein distance ≤ `max_distance` of any deny-list entry.

    The class is built lazily inside `install_custom_recognizers` because
    Presidio's `EntityRecognizer` is only importable when presidio_analyzer
    is installed (the residual safety net and tests still work without it).
    """
    import re
    from rapidfuzz.distance import Levenshtein

    class _FuzzyDenyListRecognizer(base_cls):  # type: ignore[misc, valid-type]
        # Word-boundary token regex — single token per match.  Quote/dash
        # characters are admitted so multi-word keywords like "African
        # American" still match (we additionally check 2-gram windows).
        _TOKEN_RE = re.compile(r"\b\w[\w'-]*\b", re.UNICODE)

        def __init__(self, supported_entity, deny_list, supported_language,
                     max_distance=1, fuzzy_score=0.7, min_length=4):
            super().__init__(
                supported_entities=[supported_entity],
                supported_language=supported_language,
                name=f"FuzzyDenyList<{supported_entity}>",
            )
            # Lowercase deny list for case-insensitive distance comparison.
            self._deny_lc = [d.lower() for d in deny_list]
            self._max_distance = max_distance
            self._fuzzy_score = fuzzy_score
            # Skip very short keywords (≤ 3 chars) — Levenshtein-1 produces
            # too many false positives at that length.
            self._min_length = min_length

        def load(self) -> None:
            pass

        def analyze(self, text, entities, nlp_artifacts=None):
            from presidio_analyzer import RecognizerResult
            results: list = []
            for match in self._TOKEN_RE.finditer(text):
                token = match.group()
                if len(token) < self._min_length:
                    continue
                token_lc = token.lower()
                for entry in self._deny_lc:
                    # Length pre-filter — Levenshtein ≤ k requires |len_a -
                    # len_b| ≤ k, so we can short-circuit obvious non-matches.
                    if abs(len(token_lc) - len(entry)) > self._max_distance:
                        continue
                    # Exact matches are handled by the regular deny-list
                    # recognizer; skip them here to avoid duplicate findings.
                    if token_lc == entry:
                        continue
                    if Levenshtein.distance(token_lc, entry, score_cutoff=self._max_distance) <= self._max_distance:
                        results.append(RecognizerResult(
                            entity_type=self.supported_entities[0],
                            start=match.start(),
                            end=match.end(),
                            score=self._fuzzy_score,
                            analysis_explanation=None,
                            recognition_metadata={"recognizer_name": self.name},
                        ))
                        break
            return results

    return _FuzzyDenyListRecognizer


def _build_semantic_concept_recognizer_class(base_cls: type) -> type:
    """Build a Presidio recognizer that fires on tokens whose embedding is
    close to a per-language set of concept anchors.

    Each instance carries:
      * a pre-embedded L2-normalised seed matrix (N × D, float32) so cosine
        similarity is a single matrix–vector product,
      * the surface forms of those seeds (lower-cased) so a rapidfuzz
        Levenshtein-1 pass catches typos / OOV variants the embedding can't
        see (spaCy GloVe vectors are zero for unseen words like ``Catholc``).

    Replaces the hand-curated ``app/keywords/*.txt`` deny lists.  Adding a
    new category to detection now means appending one row to
    ``SEMANTIC_CONCEPT_SEEDS``; engineering teams never have to enumerate
    surface variants again.
    """
    import numpy as np
    from rapidfuzz.distance import Levenshtein

    class _SemanticConceptRecognizer(base_cls):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            supported_entity: str,
            supported_language: str,
            seed_matrix: Any,         # numpy.ndarray, L2-normalised rows
            seed_texts_lc: tuple[str, ...],
            similarity_threshold: float,
            # Adaptive fuzzy distance applied via _max_fuzzy_for_length:
            #   3-5 char token  → distance 1   (short acronyms / OOV)
            #   ≥ 6 char token  → distance up to `fuzzy_distance` (real typos)
            # Distance 2 on long tokens catches "alzhiemer"↔"Alzheimer" and
            # "jüdischen"↔"jüdisch"; distance 1 on shorts keeps "RED"/"XL"/
            # "GARDEN" from fuzzy-matching union acronyms.
            fuzzy_distance: int = 2,
            min_token_length: int = 3,
            # Score deliberately above spaCy's default PERSON / LOCATION /
            # NRP score (0.85) so the overlap resolver picks the more
            # specific Art. 9 / Art. 10 entity over a generic spaCy NER
            # label on the same span.
            score: float = 0.9,
        ):
            super().__init__(
                supported_entities=[supported_entity],
                supported_language=supported_language,
                name=f"SemanticConcept<{supported_entity}>",
            )
            self._seed_matrix = seed_matrix
            self._seed_texts_lc = seed_texts_lc
            self._similarity_threshold = similarity_threshold
            self._fuzzy_distance = fuzzy_distance
            self._min_token_length = min_token_length
            self._score = score

        def load(self) -> None:
            pass

        def analyze(self, text, entities, nlp_artifacts=None):
            from presidio_analyzer import RecognizerResult
            if nlp_artifacts is None or not getattr(nlp_artifacts, "tokens", None):
                return []

            results: list = []
            has_seed_matrix = (
                self._seed_matrix is not None and self._seed_matrix.shape[0] > 0
            )

            for token in nlp_artifacts.tokens:
                if getattr(token, "is_punct", False) or getattr(token, "is_space", False):
                    continue
                token_text = token.text
                token_lc = token_text.lower()
                if len(token_lc) < self._min_token_length:
                    continue

                matched = False

                # Tier 1 — embedding cosine similarity against the seed matrix.
                if has_seed_matrix:
                    vector_norm = getattr(token, "vector_norm", 0.0)
                    if vector_norm and vector_norm > 0:
                        tok_unit = np.asarray(token.vector, dtype=np.float32) / float(vector_norm)
                        sims = self._seed_matrix @ tok_unit
                        if float(sims.max()) >= self._similarity_threshold:
                            matched = True

                # Tier 2 — fuzzy fallback against the seed surface forms.
                # Required because spaCy GloVe vectors are zero for OOV
                # words (e.g. "Catholc", "diabetis", LB demonyms), which
                # would slip past tier 1.  Adaptive distance keeps short
                # acronym-shaped tokens (RED / XL) from fuzzy-matching
                # 3-letter union codes (CGT / DGB).
                if not matched:
                    max_dist = self._fuzzy_distance if len(token_lc) >= 6 else 1
                    for seed_lc in self._seed_texts_lc:
                        if abs(len(token_lc) - len(seed_lc)) > max_dist:
                            continue
                        if Levenshtein.distance(
                            token_lc, seed_lc, score_cutoff=max_dist,
                        ) <= max_dist:
                            matched = True
                            break

                if matched:
                    results.append(RecognizerResult(
                        entity_type=self.supported_entities[0],
                        start=token.idx,
                        end=token.idx + len(token_text),
                        score=self._score,
                    ))
            return results

    _SemanticConceptRecognizer.__name__ = "SemanticConceptRecognizer"
    return _SemanticConceptRecognizer


def _semantic_threshold_from_env(default: float) -> float:
    import os
    raw = os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _install_semantic_recognizers(registry: Any, nlp_engine: Any) -> None:
    """Register the spaCy-driven Art. 9 / Art. 10 recognizers.

    For each (entity, language) the recognizer is given the UNION of every
    language's seeds for that entity.  langdetect routinely mis-routes
    short / ambiguous text (e.g. a 5-word English sentence misclassified as
    French); giving every per-language recognizer cross-language seed
    coverage means the detection never depends on getting language ID
    right.  Each seed is still embedded through the local language model,
    so the semantic-similarity path remains strong for that language's
    native vocabulary; the fuzzy fallback catches OOV / cross-language
    terms via surface-form Levenshtein matching.
    """
    spacy_models = getattr(nlp_engine, "nlp", None) or {}
    if not spacy_models:
        return

    try:
        import numpy as np
        from presidio_analyzer import EntityRecognizer
    except Exception:
        return

    recognizer_cls = _build_semantic_concept_recognizer_class(EntityRecognizer)
    threshold = _semantic_threshold_from_env(SEMANTIC_SIMILARITY_THRESHOLD)

    for entity, lang_seeds in SEMANTIC_CONCEPT_SEEDS.items():
        # Union of every language's seeds for this entity, de-duplicated by
        # lower-cased form while preserving the first-seen surface form.
        seen: dict[str, str] = {}
        for seeds in lang_seeds.values():
            for seed in seeds:
                seen.setdefault(seed.lower(), seed)
        union_seeds = tuple(seen.values())

        for lang in spacy_models.keys():
            nlp = spacy_models.get(lang)
            if nlp is None:
                continue

            seed_vectors: list = []
            seed_texts_lc: list[str] = []
            for seed in union_seeds:
                seed_lc = seed.lower()
                seed_texts_lc.append(seed_lc)
                doc = nlp(seed)
                norm = getattr(doc, "vector_norm", 0.0)
                if doc.has_vector and norm and norm > 0:
                    seed_vectors.append(
                        np.asarray(doc.vector, dtype=np.float32) / float(norm)
                    )

            if seed_vectors:
                seed_matrix = np.stack(seed_vectors)
            else:
                seed_matrix = np.zeros((0, 0), dtype=np.float32)

            registry.add_recognizer(recognizer_cls(
                supported_entity=entity,
                supported_language=lang,
                seed_matrix=seed_matrix,
                seed_texts_lc=tuple(seed_texts_lc),
                similarity_threshold=threshold,
            ))


def install_custom_recognizers(registry: Any, nlp_engine: Any = None) -> None:
    """Register every recognizer in `RECOGNIZERS` against the Presidio registry,
    plus the semantic concept recognizers driven by `SEMANTIC_CONCEPT_SEEDS`
    when an `nlp_engine` is provided.

    Idempotent: callers may safely invoke twice (each `add_recognizer` simply
    appends — duplicates are filtered by Presidio at analysis time).
    """
    from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer

    fuzzy_cls = _build_fuzzy_recognizer_class(EntityRecognizer) if _fuzzy_enabled() else None

    for spec in RECOGNIZERS:
        rec_cls = (
            _build_validating_class(PatternRecognizer, spec.validator)
            if spec.validator is not None
            else PatternRecognizer
        )
        for lang in spec.languages:
            kwargs: dict[str, Any] = {
                "supported_entity": spec.entity,
                "supported_language": lang,
            }
            if spec.patterns:
                kwargs["patterns"] = [
                    Pattern(name=p["name"], regex=p["regex"], score=p["score"])
                    for p in spec.patterns
                ]
            if spec.deny_list:
                kwargs["deny_list"] = list(spec.deny_list)
            if spec.context:
                kwargs["context"] = list(spec.context)
            registry.add_recognizer(rec_cls(**kwargs))

            # Optional rapidfuzz companion for any remaining deny_list specs.
            if fuzzy_cls is not None and spec.deny_list:
                registry.add_recognizer(fuzzy_cls(
                    supported_entity=spec.entity,
                    deny_list=list(spec.deny_list),
                    supported_language=lang,
                ))

    if nlp_engine is not None:
        _install_semantic_recognizers(registry, nlp_engine)
