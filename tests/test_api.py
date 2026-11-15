"""API + predictor contract tests against a tiny TF-IDF model trained on the fly."""
import json

import pytest
from fastapi.testclient import TestClient

from attackmap import api
from attackmap.predictor import AttackMapper, confidence_band


@pytest.fixture(scope="module")
def client(tiny, tmp_path_factory, kb):
    from attackmap.models.tfidf import TfidfMapper
    X, Y, labels = tiny
    root = tmp_path_factory.mktemp("artifacts")
    m = TfidfMapper(labels).fit(X, Y)
    m.threshold = 0.5
    m.save(root / "tfidf")
    (root / "ensemble.json").write_text(json.dumps({"weights": {"tfidf": 1.0}, "threshold": 0.4}))
    mapper = AttackMapper(root=root)
    api.mapper.cache_clear()
    api.app.dependency_overrides = {}
    orig = api.mapper
    api.mapper = lambda: mapper
    yield TestClient(api.app)
    api.mapper = orig


def test_confidence_band():
    assert [confidence_band(s) for s in (0.9, 0.4, 0.1)] == ["high", "medium", "low"]


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["models"] == {"tfidf": 1.0} and h["labels"] == 4
    assert "Reconnaissance" in h["tactics"]


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ATT&amp;CK Mapper" in r.text


def test_map_alert(client):
    r = client.post("/api/map", json={"text": "mimikatz sekurlsa dumped lsass memory from 10.0.0.5", "top_k": 3}).json()
    top = r["techniques"][0]
    assert top["id"] == "T1003" and top["name"] == "OS Credential Dumping"
    assert top["selected"] and "credential-access" in top["tactics"]
    assert top["url"].startswith("https://attack.mitre.org/")
    assert len(r["techniques"]) == 3
    assert r["iocs"]["ip"] == ["10.0.0.5"]
    assert r["tactics"][0]["shortname"] == "credential-access"
    assert abs(sum(t["score"] for t in r["tactics"]) - 1) < 1e-3
    assert r["evidence"]["terms"]


def test_map_validation(client):
    assert client.post("/api/map", json={"text": "x"}).status_code == 422
    assert client.post("/api/map", json={"text": "valid text", "top_k": 99}).status_code == 422


def test_batch_and_navigator(client):
    r = client.post("/api/map/batch", json={"texts": ["ransomware encrypted files on host", "phishing email attachment", "  "]}).json()
    assert [x["techniques"][0]["id"] for x in r["results"]] == ["T1486", "T1566"]
    layer = r["navigator_layer"]
    assert layer["domain"] == "enterprise-attack" and layer["versions"]["layer"] == "4.5"
    assert {t["techniqueID"] for t in layer["techniques"]} >= {"T1486", "T1566"}
    assert client.post("/api/map/batch", json={"texts": ["   "]}).status_code == 422


def test_report(client):
    text = ("The actor sent a phishing email with a malicious attachment to staff. "
            "Then mimikatz dumped credentials from lsass memory on the server. "
            "Finally ransomware encrypted files on every share. Short one.")
    r = client.post("/api/report", json={"text": text, "min_score": 0.3}).json()
    assert r["n_sentences"] == 3
    assert {t["id"] for t in r["techniques"]} == {"T1566", "T1003", "T1486"}
    assert client.post("/api/report", json={"text": "tiny. words. only here."}).status_code == 422


def test_technique_lookup_and_matrix(client):
    assert client.get("/api/techniques/T1059.001").json()["name"] == "PowerShell"
    assert client.get("/api/techniques/T0000").status_code == 404
    mx = client.get("/api/matrix").json()
    assert mx[0]["shortname"] == "reconnaissance"
    flags = {t["id"]: t["modelled"] for c in mx for t in c["techniques"]}
    assert flags["T1003"] and not flags["T1595"]


def test_sentence_splitter():
    s = api.split_sentences("First sentence has enough words. Second one also has words!\nok.")
    assert s == ["First sentence has enough words.", "Second one also has words!"]
