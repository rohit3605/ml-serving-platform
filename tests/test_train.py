import pytest

from src.train import EXPERIMENTS, build_pipeline, load_data


def test_load_data_reads_uci_format_and_drops_duplicates(tmp_path):
    file = tmp_path / "SMSSpamCollection"
    file.write_text(
        "ham\tSee you soon\n"
        "spam\tWin a free prize\n"
        "ham\tSee you soon\n",  # duplicate row
        encoding="utf-8",
    )
    df = load_data(file)
    assert list(df.columns) == ["label", "text"]
    assert len(df) == 2
    assert set(df["label"]) == {"ham", "spam"}


def test_load_data_raises_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "SMSSpamCollection")


@pytest.mark.parametrize("cfg", EXPERIMENTS, ids=lambda c: c["run_name"])
def test_every_experiment_builds_and_predicts(cfg):
    model = build_pipeline(cfg)
    model.fit(
        ["free prize now", "win cash today", "lunch tomorrow?", "call me later"],
        ["spam", "spam", "ham", "ham"],
    )
    assert model.predict(["free cash prize"])[0] in {"spam", "ham"}