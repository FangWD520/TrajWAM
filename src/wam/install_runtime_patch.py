"""Install or verify the version-locked DeepSight/WAM Transformers patch.

The repository's vendored Transformers tree intentionally omits generated
``models/auto/modeling_auto.py``, so it must not shadow the complete conda
package through PYTHONPATH. Instead, this tool copies only the project-specific
Qwen2.5-VL and DINOv3 runtime files into the Transformers 4.56.0 environment.
Existing files receive a one-time recoverable backup.

Run inside the dedicated evaluation environment.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import transformers


# The reported wam-stage1-v6 checkpoint and its training logs serialize 4.56.0.
# The vendored patch sources identify their development snapshot as dev0, but
# only the model files are copied into the complete installed 4.56.0 package.
EXPECTED_TRANSFORMERS_VERSION = "4.56.0"
GRAPH_VERSION = "native_eval_v1"
PATCH_FILES = (
    "models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "models/dinov3_vit/__init__.py",
    "models/dinov3_vit/configuration_dinov3_vit.py",
    "models/dinov3_vit/image_processing_dinov3_vit_fast.py",
    "models/dinov3_vit/modeling_dinov3_vit.py",
)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def roots() -> tuple[Path, Path, Path]:
    repo_src = Path(__file__).resolve().parents[1]
    patch_root = repo_src / "transformers" / "src" / "transformers"
    runtime_root = Path(transformers.__file__).resolve().parent
    backup_root = runtime_root.parent / f"{runtime_root.name}.wam_backup_{GRAPH_VERSION}"
    if patch_root == runtime_root:
        raise RuntimeError(
            "The incomplete vendored Transformers tree is shadowing the conda package. "
            "Remove DeepSight/src/transformers/src from PYTHONPATH and run again."
        )
    return patch_root, runtime_root, backup_root


def verify_version() -> None:
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Runtime patch refused: expected Transformers "
            f"{EXPECTED_TRANSFORMERS_VERSION}, found {transformers.__version__}. "
            "Create a version-matched environment instead of forcing the patch."
        )


def mismatches(patch_root: Path, runtime_root: Path) -> list[str]:
    bad = []
    for relative in PATCH_FILES:
        source = patch_root / relative
        target = runtime_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing repository patch file: {source}")
        if not target.is_file() or file_digest(source) != file_digest(target):
            bad.append(relative)
    return bad


def install(patch_root: Path, runtime_root: Path, backup_root: Path) -> None:
    for relative in PATCH_FILES:
        source = patch_root / relative
        target = runtime_root / relative
        backup = backup_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
        print(f"installed {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--install", action="store_true")
    args = parser.parse_args()

    verify_version()
    patch_root, runtime_root, backup_root = roots()
    if args.install:
        install(patch_root, runtime_root, backup_root)

    bad = mismatches(patch_root, runtime_root)
    if bad:
        joined = "\n  - ".join(bad)
        raise RuntimeError(
            "active Transformers does not contain the required WAM patch:\n  - "
            f"{joined}\nRun this tool once with --install."
        )

    modeling_path = runtime_root / PATCH_FILES[0]
    if f'WAM_GRAPH_VERSION = "{GRAPH_VERSION}"' not in modeling_path.read_text():
        raise RuntimeError(f"graph marker missing from {modeling_path}")
    print(f"PASS: runtime patch {GRAPH_VERSION} is installed in {runtime_root}")
    if backup_root.exists():
        print(f"Original files are recoverable from {backup_root}")


if __name__ == "__main__":
    main()
