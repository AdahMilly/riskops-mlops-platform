import pytest

from riskops.ml.lineage import get_dvc_lock_hash


def test_get_dvc_lock_hash(tmp_path) -> None:
    lock_file = tmp_path / "dvc.lock"

    lock_file.write_text(
        "test dataset version",
        encoding="utf-8",
    )

    result = get_dvc_lock_hash(lock_file)

    assert len(result) == 64
    assert all(character in "0123456789abcdef" for character in result)


def test_get_dvc_lock_hash_requires_file(tmp_path) -> None:
    missing_file = tmp_path / "missing.lock"

    with pytest.raises(FileNotFoundError):
        get_dvc_lock_hash(missing_file)
