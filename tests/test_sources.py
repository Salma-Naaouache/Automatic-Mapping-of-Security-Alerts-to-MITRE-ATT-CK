from attackmap.data.sources import sigma_labels, sigma_rule_to_alert

RULE = {
    "title": "Suspicious Encoded PowerShell",
    "description": "Detects base64 encoded PowerShell",
    "logsource": {"product": "windows", "category": "process_creation"},
    "detection": {
        "selection": {"Image|endswith": "\\powershell.exe", "CommandLine|contains": [" -enc ", " -EncodedCommand "]},
        "filter": {"ParentImage": "C:\\Windows\\explorer.exe"},
        "condition": "selection and not filter",
    },
    "tags": ["attack.execution", "attack.t1059.001", "attack.T1027", "attack.g0016", "cve.2021-1234"],
}


def test_sigma_labels_only_techniques():
    assert sigma_labels(RULE) == ["T1027", "T1059.001"]
    assert sigma_labels({"tags": None}) == []


def test_sigma_rule_to_alert():
    a = sigma_rule_to_alert(RULE)
    assert a.startswith("Alert: Suspicious Encoded PowerShell.")
    assert "Detects base64 encoded PowerShell." in a
    assert "Log source: windows/process_creation." in a
    assert "Image endswith \\powershell.exe" in a
    assert "CommandLine contains  -enc " in a
    assert "ParentImage is C:\\Windows\\explorer.exe" in a
    assert "condition" not in a


def test_sigma_rule_without_title():
    assert sigma_rule_to_alert({"description": "x"}) is None
