import sys
import types

from app.anonymization import (
    SPACY_LABELS_TO_IGNORE,
    SPACY_TO_PRESIDIO_ENTITY_MAPPING,
    SUPPORTED_LANGUAGES,
    build_engines,
)


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
