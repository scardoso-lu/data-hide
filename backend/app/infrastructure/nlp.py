"""NLP engine infrastructure — owns Presidio analyzer lifecycle.

``NlpConfig`` captures every NLP knob in one place.  ``NlpEngine`` wraps a
single Presidio ``AnalyzerEngine`` and its resident spaCy models with an
explicit open/close lifetime, replacing the ``lru_cache(maxsize=1)``
anti-pattern in the domain layer.  ``SequentialModelSlot`` enforces the
"at most one extra spaCy model resident at a time" memory contract for the
Tier B2 embedding-similarity walk.
"""

from __future__ import annotations

import copy
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# spaCy ↔ Presidio entity mapping and ignore list.
#
# Copied here from domain/anonymization.py so that the engine builder lives
# entirely inside the infrastructure layer.  The domain module re-exports
# these names for backward compatibility with any external callers.
# ─────────────────────────────────────────────────────────────────────────────

SPACY_TO_PRESIDIO_ENTITY_MAPPING: dict[str, str] = {
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

SPACY_LABELS_TO_IGNORE: list[str] = [
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
    # German/French spaCy models emit a generic "MISC" label that has no
    # Presidio mapping; without this entry Presidio logs a warning per
    # occurrence ("Entity MISC is not mapped to a Presidio entity").
    "MISC",
]


# ─────────────────────────────────────────────────────────────────────────────
# NlpConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NlpConfig:
    """All NLP knobs in one place.

    Create via ``NlpConfig.from_env()`` for production use; construct directly
    in tests to inject mocked values without touching the environment.
    """

    spacy_models: dict[str, str]    # {"en": "en_core_web_lg", ...}
    supported_languages: list[str]
    similarity_threshold: float = 0.55          # COLUMN_SIMILARITY_THRESHOLD
    structured_enabled: bool = True             # ENABLE_PRESIDIO_STRUCTURED
    semantic_similarity_threshold: float = 0.55  # SEMANTIC_SIMILARITY_THRESHOLD
    anonymization_regions: str = "all"          # ANONYMIZATION_REGIONS
    fuzzy_enabled: bool = False                 # ENABLE_FUZZY_TYPO_MATCH

    @classmethod
    def from_env(cls) -> "NlpConfig":
        """Build an ``NlpConfig`` from the current process environment."""
        defaults: dict[str, str] = {
            "en": "en_core_web_lg",
            "fr": "fr_core_news_lg",
            "de": "de_core_news_lg",
            "lb": "de_core_news_lg",
        }
        spacy_models = {
            lang: os.environ.get(f"SPACY_MODEL_{lang.upper()}", default)
            for lang, default in defaults.items()
        }
        return cls(
            spacy_models=spacy_models,
            supported_languages=list(spacy_models.keys()),
            similarity_threshold=float(
                os.environ.get("COLUMN_SIMILARITY_THRESHOLD", "0.55")
            ),
            structured_enabled=(
                os.environ.get("ENABLE_PRESIDIO_STRUCTURED", "1") != "0"
            ),
            semantic_similarity_threshold=float(
                os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD", "0.55")
            ),
            anonymization_regions=os.environ.get("ANONYMIZATION_REGIONS", "all"),
            fuzzy_enabled=(
                os.environ.get("ENABLE_FUZZY_TYPO_MATCH", "0") == "1"
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Heap management
# ─────────────────────────────────────────────────────────────────────────────

def _trim_native_heap() -> None:
    """Ask glibc to return freed pages to the OS (Linux containers).

    ``gc.collect()`` frees the Python objects, but glibc's allocator keeps
    the pages mapped by default, so container RSS plateaus at the high-water
    mark even though the memory is reusable.  ``malloc_trim(0)`` releases
    them — making the per-stage RSS log lines reflect reality and giving
    tight-memory containers the headroom back.  No-op on non-glibc platforms.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Private builder helpers (previously in domain/anonymization.py)
# ─────────────────────────────────────────────────────────────────────────────

def _recognizer_language_code(language: object) -> str | None:
    if isinstance(language, str):
        return language
    if isinstance(language, dict):
        value = language.get("language")
        return value if isinstance(value, str) else None
    return None


def _filter_recognizer_config(config: dict, supported_languages: list[str]) -> dict:
    filtered = copy.deepcopy(config)
    filtered["supported_languages"] = supported_languages
    recognizers = []

    for recognizer in filtered.get("recognizers", []):
        languages = (
            recognizer.get("supported_languages") if isinstance(recognizer, dict) else None
        )
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


def _install_custom_recognizers(registry: Any, nlp_engine: Any = None) -> None:
    """Install GDPR-special-category recognizers (LU CCSS, salary, semantic
    Art. 9 / Art. 10 detection, …).

    Imported lazily so the build_engines unit tests that monkeypatch
    presidio_analyzer don't pull in the real PatternRecognizer class.  The
    ``nlp_engine`` is forwarded so the semantic-concept recognizers can embed
    their seed anchors through the same spaCy models the analyzer uses.
    """
    try:
        from .nlp_recognizers import install_custom_recognizers
    except Exception:
        return
    try:
        install_custom_recognizers(registry, nlp_engine)
    except Exception as exc:
        logger.warning("Failed to install custom recognizers: %s", exc)


def _build_recognizer_registry(
    nlp_engine: Any,
    configuration_loader: Any,
    list_loader: Any,
    registry_cls: Any,
    supported_languages: list[str] | None = None,
) -> Any:
    from ..domain.anonymization import SUPPORTED_LANGUAGES as _SUPPORTED_LANGUAGES
    config = _filter_recognizer_config(
        configuration_loader.get(),
        supported_languages or _SUPPORTED_LANGUAGES,
    )
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
    _install_custom_recognizers(registry, nlp_engine)
    return registry


def _build_analyzer(config: NlpConfig, languages: list[str]) -> Any:
    """Internal builder — constructs and returns a Presidio ``AnalyzerEngine``."""
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.recognizer_registry.recognizers_loader_utils import (
        RecognizerConfigurationLoader,
        RecognizerListLoader,
    )

    active_models = {
        lang: model
        for lang, model in config.spacy_models.items()
        if lang in languages
    }

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": lang, "model_name": model}
            for lang, model in active_models.items()
        ],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": SPACY_TO_PRESIDIO_ENTITY_MAPPING,
            "labels_to_ignore": SPACY_LABELS_TO_IGNORE,
        },
    })
    nlp_engine = provider.create_engine()
    registry = _build_recognizer_registry(
        nlp_engine,
        RecognizerConfigurationLoader,
        RecognizerListLoader,
        RecognizerRegistry,
        supported_languages=languages,
    )
    gc.collect()  # release any engine just evicted from a previous slot
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=languages,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SequentialModelSlot
# ─────────────────────────────────────────────────────────────────────────────

class SequentialModelSlot:
    """Single-resident slot: at most one sequentially-loaded spaCy model.

    Requesting a different model evicts (and garbage-collects) the previous
    one before loading the next, so peak RSS never holds two models at once.
    """

    def __init__(self) -> None:
        self._slot: tuple[str, Any] | None = None

    def load(self, model_name: str) -> Any | None:
        """Return the model, loading it if necessary.

        If the requested model is already resident, it is returned directly.
        If a different model is resident it is released first.  Returns
        ``None`` on load failure so the caller can gracefully skip that
        language.
        """
        if self._slot is not None and self._slot[0] == model_name:
            return self._slot[1]
        # Evict before loading so two models never coexist.
        self._slot = None
        gc.collect()
        _trim_native_heap()
        try:
            import spacy
            nlp = spacy.load(model_name)
        except Exception:
            return None
        self._slot = (model_name, nlp)
        return nlp

    def release(self) -> None:
        """Release the resident model and trim the native heap."""
        self._slot = None
        gc.collect()
        _trim_native_heap()


# ─────────────────────────────────────────────────────────────────────────────
# NlpEngine
# ─────────────────────────────────────────────────────────────────────────────

class NlpEngine:
    """Owns one Presidio ``AnalyzerEngine`` and its resident spaCy models.

    Build once per pipeline phase, call ``close()`` when done.  Replaces the
    ``lru_cache(maxsize=1)`` anti-pattern with explicit lifetime management.

    Example::

        config = NlpConfig.from_env()
        engine = NlpEngine(config, languages=("en",))
        analyzer = engine.analyzer  # builds lazily on first access
        # … use analyzer …
        engine.close()
    """

    def __init__(
        self,
        config: NlpConfig,
        languages: tuple[str, ...] | None = None,
    ) -> None:
        self._config = config
        self._languages: list[str] = (
            list(languages) if languages else list(config.supported_languages)
        )
        self._analyzer: Any = None

    def build(self) -> Any:
        """Build and cache the Presidio analyzer.  Idempotent."""
        if self._analyzer is None:
            self._analyzer = _build_analyzer(self._config, self._languages)
        return self._analyzer

    @property
    def analyzer(self) -> Any:
        """The resident ``AnalyzerEngine`` (built lazily on first access)."""
        return self.build()

    def models_from_analyzer(self) -> dict[str, Any]:
        """Extract the already-loaded spaCy pipelines from the resident analyzer.

        Presidio's ``SpacyNlpEngine`` keeps its models in ``nlp_engine.nlp``
        (``{lang_code: spacy.Language}``).  Reusing them for Tier B2 embedding
        similarity avoids loading a second copy of each model.
        """
        if self._analyzer is None:
            return {}
        try:
            nlp_map = getattr(
                getattr(self._analyzer, "nlp_engine", None), "nlp", None
            )
            if isinstance(nlp_map, dict) and nlp_map:
                return dict(nlp_map)
        except Exception:
            pass
        return {}

    def close(self) -> None:
        """Release the analyzer and trim the native heap."""
        self._analyzer = None
        gc.collect()
        _trim_native_heap()
