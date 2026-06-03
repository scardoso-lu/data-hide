from __future__ import annotations

import sys

import pytest

import main as main


def test_ensure_spacy_models_accepts_importable_models(mocker, monkeypatch, tmp_path):
    models_dir = str(tmp_path)
    monkeypatch.setattr(main, "_MODELS_DIR", models_dir)
    find_spec = mocker.patch("main.find_spec", return_value=object())
    download = mocker.patch("main._download_spacy_model")

    try:
        main._ensure_spacy_models()

        assert {call.args[0] for call in find_spec.call_args_list} == set(main.SPACY_MODELS.values())
        download.assert_not_called()
    finally:
        if models_dir in sys.path:
            sys.path.remove(models_dir)


def test_ensure_spacy_models_downloads_missing_model(mocker, monkeypatch, tmp_path):
    models_dir = str(tmp_path)
    monkeypatch.setattr(main, "_MODELS_DIR", models_dir)

    seen = {}

    def fake_find_spec(model: str):
        if model == "fr_core_news_lg" and not seen.get(model):
            seen[model] = True
            return None
        return object()

    mocker.patch("main.find_spec", side_effect=fake_find_spec)
    download = mocker.patch("main._download_spacy_model")

    main._ensure_spacy_models()

    download.assert_called_once_with("fr_core_news_lg", models_dir)


def test_ensure_spacy_models_adds_custom_models_dir(mocker, monkeypatch, tmp_path):
    models_dir = str(tmp_path)
    monkeypatch.setattr(main, "_MODELS_DIR", models_dir)
    mocker.patch("main.find_spec", return_value=object())

    try:
        main._ensure_spacy_models()

        assert sys.path[0] == models_dir
    finally:
        if models_dir in sys.path:
            sys.path.remove(models_dir)


def test_ensure_spacy_models_fails_if_downloaded_model_still_missing(mocker, monkeypatch, tmp_path):
    models_dir = str(tmp_path)
    monkeypatch.setattr(main, "_MODELS_DIR", models_dir)
    mocker.patch("main.find_spec", return_value=None)
    mocker.patch("main._download_spacy_model")

    with pytest.raises(RuntimeError, match="fr_core_news_lg|en_core_web_lg|de_core_news_lg"):
        main._ensure_spacy_models()
