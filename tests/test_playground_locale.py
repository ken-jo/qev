"""English UI configuration without loading model weights or a browser."""

import json
from importlib.resources import files

from qev.playground.app import english_ui_i18n


def test_owned_dictionary_is_packaged_and_matches_pinned_gradio():
    data = json.loads(files("qev.playground").joinpath("english_ui.json").read_text("utf-8"))
    assert data["source"]["version"] == "6.29.0"
    assert data["source"]["license"] == "Apache-2.0"
    assert len(data["source"]["sha256"]) == 64
    assert {"en", "ko", "ja", "zh-CN", "pt-BR"} <= set(data["locales"])
    assert len(data["messages"]) == 160
    assert all(isinstance(value, str) for value in data["messages"].values())


def test_each_browser_locale_receives_flat_english_control_labels():
    translations = english_ui_i18n().translations_dict
    english = translations["en"]
    assert english["upload_text.drop_image"] == "Drop Image Here"
    assert english["upload_text.click_to_upload"] == "Click to Upload"
    assert english["common.loading"] == "Loading"
    assert "upload_text" not in english
    assert all(messages == english for messages in translations.values())
    # Independent dictionaries prevent one locale mutation from changing another.
    translations["ko"]["common.loading"] = "Changed"
    assert english["common.loading"] == "Loading"
