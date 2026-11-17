"""Evaluate disjoint route shards, with one planner and CARLA per GPU."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="physical GPU indices, e.g. 0,1,2,3; one GPU is also supported")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--tm-port", type=int, default=50100)
    parser.add_argument("--tm-seed", type=int, default=0)
    parser.add_argument("--startup-delay", type=float, default=15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if not all(g.isdigit() for g in gpus) or len(gpus) != len(set(gpus)):
        parser.error("GPU indices must be distinct non-negative integers")
    route_file = ROOT / "bench2drive/leaderboard/data/bench2drive220.xml"
    tree = ET.parse(route_file).getroot()
    routes = tree.findall("route")
    assert len(routes) == len({r.get("id") for r in routes}) == 220
    if len(gpus) > len(routes):
        parser.error("Too many GPUs")
    # Reserve RPC, streaming and secondary ports for each CARLA process.
    ports = [args.port + i * 10 + offset for i in range(len(gpus)) for offset in [0, 1, 2]]
    ports += [args.tm_port + i * 10 for i in range(len(gpus))]
    if len(ports) != len(set(ports)) or not all(1024 <= p <= 65535 for p in ports):
        parser.error("Invalid or overlapping simulator/traffic-manager ports")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    identity = {"checkpoint": checkpoint, "gpus": gpus, "port": args.port,
                "tm_port": args.tm_port, "tm_seed": args.tm_seed,
                "routes_sha256": hashlib.sha256(route_file.read_bytes()).hexdigest()}
    identity_path = output / "evaluation.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            parser.error("This output directory belongs to a different evaluation")
        if list(output.glob("result_shard*.json")) and not args.resume:
            parser.error("Results already exist; use --resume or a new output directory")
    elif any(output.iterdir()):
        parser.error("Use an empty output directory")
    if not args.dry_run:
        if not os.environ.get("CARLA_ROOT"):
            parser.error("Set CARLA_ROOT")
        subprocess.run([os.sys.executable, str(ROOT / "scripts/check_environment.py"),
                        "--checkpoint", checkpoint, "--files-only"], check=True)
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    jobs, offset = [], 0
    for i, gpu in enumerate(gpus):
        count = len(routes) // len(gpus) + (i < len(routes) % len(gpus))
        shard = ET.Element(tree.tag, tree.attrib)
        for child in tree:
            if child.tag != "route":
                shard.append(copy.deepcopy(child))
        selected = routes[offset:offset+count]
        offset += count
        shard.extend(copy.deepcopy(selected))
        xml_path = output / f"routes_{i}.xml"
        ET.ElementTree(shard).write(xml_path, encoding="utf-8", xml_declaration=True)
        jobs.append({"gpu": gpu, "route_ids": [r.get("id") for r in selected],
                     "port": args.port+i*10, "tm_port": args.tm_port+i*10})
    (output / "shards.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print("Routes per worker:", [len(j["route_ids"]) for j in jobs], flush=True)
    if args.dry_run:
        return
    processes = []
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        for i, job in enumerate(jobs):
            if i:
                time.sleep(args.startup_delay)
            env = dict(os.environ, GPU_RANK=job["gpu"], TEAM_CONFIG=checkpoint,
                       ROUTES=str(output / f"routes_{i}.xml"), SAVE_PATH=str(output / f"frames_{i}"),
                       CHECKPOINT_ENDPOINT=str(output / f"result_shard{i}.json"),
                       PORT=str(job["port"]), TM_PORT=str(job["tm_port"]),
                       TM_SEED=str(args.tm_seed), RESUME=str(args.resume))
            with (output / f"log_shard{i}.txt").open("a" if args.resume else "w") as log:
                processes.append(subprocess.Popen(["bash", str(ROOT / "scripts/evaluate.sh")],
                                                  env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("A worker failed; inspect log_shard*.txt before resuming")
            time.sleep(2)
        if any(p.returncode for p in processes):
            raise RuntimeError("A worker failed; inspect log_shard*.txt")
        subprocess.run([os.sys.executable, str(ROOT / "scripts/aggregate_results.py"),
                        "--output", str(output)], check=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
