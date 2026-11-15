from attackmap.data.attack_kb import KnowledgeBase, parse_procedures, parse_tactics, parse_techniques


def _ref(eid):
    return [{"source_name": "mitre-attack", "external_id": eid, "url": f"https://attack.mitre.org/{eid}"}]


BUNDLE = {"objects": [
    {"type": "x-mitre-tactic", "id": "x-t1", "name": "Execution", "x_mitre_shortname": "execution", "external_references": _ref("TA0002")},
    {"type": "x-mitre-tactic", "id": "x-t2", "name": "Initial Access", "x_mitre_shortname": "initial-access", "external_references": _ref("TA0001")},
    {"type": "x-mitre-tactic", "id": "x-t3", "name": "Old", "x_mitre_shortname": "old", "x_mitre_deprecated": True, "external_references": _ref("TA9999")},
    {"type": "x-mitre-matrix", "id": "m", "tactic_refs": ["x-t2", "x-t1", "x-t3"]},
    {"type": "attack-pattern", "id": "ap1", "name": "Command and Scripting Interpreter", "description": "Adversaries abuse interpreters.(Citation: x)",
     "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "execution"}], "external_references": _ref("T1059")},
    {"type": "attack-pattern", "id": "ap2", "name": "PowerShell", "x_mitre_is_subtechnique": True, "description": "PS.",
     "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "execution"}], "external_references": _ref("T1059.001")},
    {"type": "attack-pattern", "id": "ap3", "name": "Phishing", "description": "Phish.",
     "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "initial-access"},
                           {"kill_chain_name": "mitre-attack", "phase_name": "execution"}], "external_references": _ref("T1566")},
    {"type": "attack-pattern", "id": "ap4", "name": "Revoked", "revoked": True, "external_references": _ref("T1000")},
    {"type": "intrusion-set", "id": "g1", "name": "APT-X"},
    {"type": "relationship", "id": "r1", "relationship_type": "uses", "source_ref": "g1", "target_ref": "ap2",
     "description": "[APT-X](https://x) used PowerShell to run payloads."},
    {"type": "relationship", "id": "r2", "relationship_type": "uses", "source_ref": "g1", "target_ref": "ap4", "description": "ignored"},
    {"type": "relationship", "id": "r3", "relationship_type": "mitigates", "source_ref": "g1", "target_ref": "ap1", "description": "ignored"},
]}


def test_tactics_follow_matrix_order_and_skip_deprecated():
    tac = parse_tactics(BUNDLE)
    assert [t.shortname for t in tac] == ["initial-access", "execution"]
    assert tac[0].id == "TA0001"


def test_parse_techniques():
    t = parse_techniques(BUNDLE)
    assert set(t) == {"T1059", "T1059.001", "T1566"}  # revoked dropped
    assert t["T1566"].tactics == ["initial-access", "execution"]  # matrix order
    assert "Citation" not in t["T1059"].description
    assert t["T1059.001"].is_subtechnique and t["T1059.001"].parent == "T1059"


def test_parse_procedures():
    rows = parse_procedures(BUNDLE)
    assert len(rows) == 1
    assert rows[0]["labels"] == ["T1059.001"] and rows[0]["origin"] == "intrusion-set:APT-X"


def test_kb_roundtrip(tmp_path):
    kb = KnowledgeBase.from_bundle(BUNDLE)
    p = kb.save(tmp_path / "kb.json")
    kb2 = KnowledgeBase.load(p)
    assert len(kb2) == 3 and "t1059" in kb2
    assert kb2.tactics_of("T1059.001") == ["execution"]
    assert kb2.tactics_of("T1059.999") == ["execution"]  # unknown sub-technique falls back to parent
    assert kb2.name_of("T9999") == "T9999"
    assert set(kb2.parents) == {"T1059", "T1566"}


def test_real_kb(kb):
    assert len(kb) > 600
    assert kb.get("T1059").name == "Command and Scripting Interpreter"
    assert len(kb.tactics) >= 14
    assert kb.tactics[0].shortname == "reconnaissance"
