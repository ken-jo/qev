from veyra.augment_data import variants
from veyra.build_data import text_records
from veyra.data import audit_records


def test_context_variants_preserve_targets_and_groups():
    original = text_records(1, 51)[0]
    expanded = variants(original, 1)
    assert len(expanded) == 3
    assert len(expanded[1].request.questions) == 1
    assert {x.group_id for x in expanded} == {original.group_id}
    assert {x.split for x in expanded} == {original.split}
    for record in expanded:
        for name, target in record.targets.items():
            assert target == original.targets[name]
    audit_records(expanded)
