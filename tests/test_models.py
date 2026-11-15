"""Every model family: fit on a tiny separable dataset, beat chance, survive save/load."""
import numpy as np
import pytest

from attackmap.evaluate import ranking_metrics
from attackmap.models.ensemble import combine, normalise, search_weights


def _check(model, tiny, tmp_path, min_p1=0.9):
    X, Y, labels = tiny
    p = model.predict_proba(X)
    assert p.shape == (len(X), len(labels))
    assert np.all((p >= 0) & (p <= 1))
    assert ranking_metrics(Y, p)["P@1"] >= min_p1
    model.threshold = 0.42
    model.save(tmp_path / model.name)
    m2 = type(model).load(tmp_path / model.name)
    assert m2.labels == labels and m2.threshold == 0.42
    np.testing.assert_allclose(m2.predict_proba(X), p, atol=1e-4)
    return p


def test_tfidf(tiny, tmp_path):
    from attackmap.models.tfidf import TfidfMapper
    X, Y, labels = tiny
    m = TfidfMapper(labels).fit(X, Y)
    _check(m, tiny, tmp_path)
    terms = dict(m.explain("mimikatz dumped lsass", labels.index("T1003")))
    assert "mimikatz" in terms or "lsass" in terms


def test_bilstm(tiny, tmp_path):
    from attackmap.models.bilstm import BiLSTMMapper, tokenize
    assert tokenize("PowerShell.exe from <IP>") == ["powershell.exe", "from", "<IP>"]
    X, Y, labels = tiny
    m = BiLSTMMapper(labels, epochs=12, batch_size=16, lr=5e-3).fit(X, Y, val=(X, Y), log=lambda *_: None)
    _check(m, tiny, tmp_path, min_p1=0.8)
    att = m.attention("mimikatz dumped lsass memory")
    assert [t for t, _ in att] == ["mimikatz", "dumped", "lsass", "memory"]
    assert sum(a for _, a in att) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.slow
def test_transformer_tiny(tiny, tmp_path):
    from attackmap.models.transformer_clf import TransformerMapper
    X, Y, labels = tiny
    m = TransformerMapper(labels, "hf-internal-testing/tiny-random-bert", epochs=1, batch_size=16)
    m.fit(X, Y, val=(X, Y), log=lambda *_: None)
    _check(m, tiny, tmp_path, min_p1=0.0)  # random tiny model: test mechanics only
    assert len(m.history) == 1 and "MRR" in m.history[0]


@pytest.mark.slow
def test_sbert_zeroshot_and_finetuned(tiny, tmp_path, kb):
    from attackmap.models.sbert import SBertMapper
    X, Y, labels = tiny
    zs = SBertMapper(labels, finetune=False, kb=kb).fit(X, Y)
    p = zs.predict_proba(X)
    assert ranking_metrics(Y, p)["P@1"] > 0.5  # zero-shot from ATT&CK descriptions only
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-4)
    top = zs.retrieve("powershell -enc", ["T1059.001", "T1486", "T1566.001"], top=1)
    assert top[0][0] == "T1059.001"
    ft = SBertMapper(labels, finetune=True, epochs=1, batch_size=8, kb=kb).fit(X, Y, log=lambda *_: None)
    _check(ft, tiny, tmp_path)


def test_ensemble_weights_prefer_better_model():
    rng = np.random.default_rng(0)
    y = np.eye(5, dtype=np.float32)[rng.integers(0, 5, 200)]
    good = y * 0.8 + rng.random(y.shape) * 0.3
    bad = rng.random(y.shape)
    w, best = search_weights(y, [bad, good], step=0.25)
    assert w[1] >= 0.75 and best > 0.9
    np.testing.assert_allclose(normalise(good).sum(1), 1, atol=1e-6)
    assert combine([bad, good], [0, 1]).shape == y.shape
