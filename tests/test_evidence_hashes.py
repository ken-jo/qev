from veyra.release_gate import evidence_hashes, evidence_matches


def test_published_newlines_preserve_report_evidence_identity():
    windows = b'{\r\n  "accuracy": 0.89\r\n}\r\n'
    published = windows.replace(b"\r\n", b"\n")
    reference = evidence_hashes(windows)
    assert reference["sha256"] != reference["lf_sha256"]
    assert evidence_matches(windows, reference)
    assert evidence_matches(published, reference)


def test_content_changes_cannot_pass_as_newline_normalization():
    original = b'{\r\n  "accuracy": 0.89\r\n}\r\n'
    reference = evidence_hashes(original)
    changed = original.replace(b"0.89", b"0.99").replace(b"\r\n", b"\n")
    assert not evidence_matches(changed, reference)
    assert not evidence_matches(original + b" ", reference)


def test_legacy_evidence_without_lf_hash_still_requires_exact_bytes():
    original = b"original\r\n"
    reference = {"sha256": evidence_hashes(original)["sha256"]}
    assert evidence_matches(original, reference)
    assert not evidence_matches(b"original\n", reference)
