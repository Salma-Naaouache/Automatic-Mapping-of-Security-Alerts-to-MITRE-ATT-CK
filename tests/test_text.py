import pytest

from attackmap.text import clean, extract_iocs, mask_iocs, parent_technique, strip_markup


def test_strip_citations_and_links():
    s = "[APT29](https://attack.mitre.org/groups/G0016) used PowerShell.(Citation: Mandiant 2020)"
    assert strip_markup(s) == "APT29 used PowerShell."


@pytest.mark.parametrize("raw,token", [
    ("beacon to 185.220.101.4:443", "<IP>"),
    ("defanged 10[.]0[.]0[.]1", "<IP>"),
    ("hash " + "a" * 64, "<SHA256>"),
    ("hash " + "b" * 40, "<SHA1>"),
    ("hash " + "c" * 32, "<MD5>"),
    ("get https://evil.example/x.php?a=1", "<URL>"),
    ("mail bob@corp.example", "<EMAIL>"),
    ("guid 3f2504e0-4f89-11d3-9a0c-0305e82c3301", "<GUID>"),
])
def test_mask_iocs(raw, token):
    assert token in mask_iocs(raw)


def test_clean_is_idempotent_and_collapses_ws():
    s = "  PowerShell   -enc   from 1.2.3.4 \n\n (Citation: X) "
    once = clean(s)
    assert once == "PowerShell -enc from <IP>"
    assert clean(once) == once
    assert clean("") == ""


@pytest.mark.parametrize("tid,parent", [("T1059.001", "T1059"), ("t1003", "T1003"), (" T1566.002 ", "T1566")])
def test_parent_technique(tid, parent):
    assert parent_technique(tid) == parent


@pytest.mark.parametrize("bad", ["TA0001", "1059", "T105", "T1059.1", "S0002"])
def test_parent_technique_rejects(bad):
    with pytest.raises(ValueError):
        parent_technique(bad)


def test_extract_iocs():
    s = ("C2 185.220.101.4 via hxxp://bad.example/a dropped run.ps1 sha256 " + "e" * 64 +
         " exploiting CVE-2021-44228; persistence HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\x")
    io = extract_iocs(s)
    assert io["ip"] == ["185.220.101.4"]
    assert io["cve"] == ["CVE-2021-44228"]
    assert io["file"] == ["run.ps1"]
    assert io["sha256"] == ["e" * 64]
    assert "md5" not in io and "sha1" not in io  # substrings of the sha256 are not double-reported
    assert io["registry"][0].startswith("HKCU\\Software")
