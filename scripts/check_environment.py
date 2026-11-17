"""Check simulator dependencies and checkpoint layout before evaluation."""
import argparse
import importlib
from pathlib import Path
import os
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--files-only", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()
    from wam.native_loading import checkpoint_files
    _, tensors = checkpoint_files(args.checkpoint)
    routes = ET.parse(ROOT / "bench2drive/leaderboard/data/bench2drive220.xml").getroot().findall("route")
    assert len(routes) == len({r.get("id") for r in routes}) == 220
    assert (ROOT / "assets/hisblack.jpg").is_file()
    print(f"Checkpoint: {len(tensors)} tensor entries; route XML: 220 unique routes")
    if args.files_only:
        return
    root = Path(os.environ["CARLA_ROOT"]).resolve()
    if not (root / "CarlaUE4.sh").is_file():
        raise FileNotFoundError(root / "CarlaUE4.sh")
    if os.geteuid() == 0:
        import pwd
        user = os.environ.get("CARLA_USER")
        if not user or pwd.getpwnam(user).pw_uid == 0:
            raise RuntimeError("Set CARLA_USER to a non-root account with access to CARLA_ROOT")
    sys.path[:0] = [str(ROOT / "bench2drive"), str(ROOT / "bench2drive/leaderboard"),
                   str(ROOT / "bench2drive/scenario_runner"), str(root / "PythonAPI"),
                   str(root / "PythonAPI/carla")]
    for name in ["carla", "torch", "torchvision", "transformers", "qwen_vl_utils", "cv2",
                 "py_trees", "dictor", "ephem", "shapely", "agents.navigation.global_route_planner",
                 "leaderboard.leaderboard_evaluator", "team_code.qwen_b2d_agent"]:
        importlib.import_module(name)
    from wam.install_runtime_patch import verify_version, roots, mismatches
    verify_version()
    patch, runtime, _ = roots()
    if mismatches(patch, runtime):
        raise RuntimeError("Install the supplied runtime patch with scripts/setup.sh")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for closed-loop evaluation")
    print("Imports and runtime patch: OK; GPU:", torch.cuda.get_device_name(0))
    if args.load_model:
        from wam.native_loading import load_native_model
        _, model, _ = load_native_model(args.checkpoint)
        print("Complete checkpoint load: OK; parameters:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
