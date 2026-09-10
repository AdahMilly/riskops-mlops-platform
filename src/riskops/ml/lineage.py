import hashlib
import subprocess
from pathlib import Path


def get_git_revision() -> str:
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def get_dvc_lock_hash(
    path: Path = Path("dvc.lock"),
) -> str:
    if not path.exists():
        raise FileNotFoundError(f"DVC lock file not found: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(8192),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_dataset_lineage() -> dict[str, str]:
    return {
        "git_revision": get_git_revision(),
        "dvc_lock_sha256": get_dvc_lock_hash(),
    }
