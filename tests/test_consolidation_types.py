from memory_inference.consolidation.consolidation_types import UpdateType


def test_update_type_has_four_variants() -> None:
    assert set(UpdateType) == {
        UpdateType.NEW,
        UpdateType.REINFORCEMENT,
        UpdateType.SUPERSESSION,
        UpdateType.CONFLICT,
    }
