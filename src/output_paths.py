from __future__ import annotations

from pathlib import Path


OUTPUT_ROOT = Path("/mnt/binary_nn")
ARTIFACTS_DIRNAME = "artifacts"
CHECKPOINTS_DIRNAME = "checkpoints"


def output_root() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT


def output_dir(name: str) -> Path:
    path = output_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_root() -> Path:
    return output_dir(CHECKPOINTS_DIRNAME)


def artifacts_root() -> Path:
    return output_dir(ARTIFACTS_DIRNAME)


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_path(
    requested_path: Path | None,
    *,
    default_subdir: str,
    default_name: str,
) -> Path:
    if requested_path is None:
        return ensure_parent_dir(output_dir(default_subdir) / default_name)

    if requested_path.is_absolute():
        try:
            requested_path.relative_to(output_root())
            return ensure_parent_dir(requested_path)
        except ValueError:
            return ensure_parent_dir(output_dir(default_subdir) / requested_path.name)

    return ensure_parent_dir(output_dir(default_subdir) / requested_path)