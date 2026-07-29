from tofu.layers.language_models import FastTextLanguageProvider


def test_absent_fasttext_model_is_explicitly_unknown(monkeypatch, tmp_path):
    monkeypatch.delenv("TOFU_FASTTEXT_LID", raising=False)
    provider = FastTextLanguageProvider(str(tmp_path / "missing.ftz"))
    result = provider.identify(["RÉPUBLIQUE"])
    assert result.language is None
    assert result.confidence == 0.0
    assert provider.status()["ready"] is False


def test_short_text_is_never_forced_to_english(tmp_path):
    result = FastTextLanguageProvider(str(tmp_path / "missing.ftz")).identify(["A"])
    assert result.language is None
    assert result.reason == "insufficient text"
