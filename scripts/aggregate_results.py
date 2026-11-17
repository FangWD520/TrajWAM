"""Merge complete, unique route records and compute the benchmark DS and strict SR."""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def aggregate(output):
    output = Path(output)
    expected = {r.get("id") for r in ET.parse(ROOT / "bench2drive/leaderboard/data/bench2drive220.xml").getroot().findall("route")}
    records = {}
    files = sorted(output.glob("result_shard*.json"))
    if not files:
        raise ValueError("No result_shard*.json files")
    for path in files:
        data = json.loads(path.read_text())
        if data.get("entry_status") in {"Crashed", "Rejected", "Started", "Invalid"}:
            raise ValueError(f"Unfinished evaluator shard: {path.name}")
        checkpoint = data["_checkpoint"]
        progress = checkpoint.get("progress")
        if progress and progress[0] != progress[1]:
            raise ValueError(f"Incomplete route progress: {path.name}")
        for record in checkpoint["records"]:
            match = re.fullmatch(r"RouteScenario_(\d+)_rep0", record["route_id"])
            if not match:
                raise ValueError(f"Unexpected route identifier: {record['route_id']}")
            route_id = match.group(1)
            if route_id not in expected or route_id in records:
                raise ValueError(f"Unexpected or duplicate route: {route_id}")
            if record["status"] not in {"Completed", "Perfect", "Failed"} and not record["status"].startswith("Failed - "):
                raise ValueError(f"Unfinished or infrastructure-failed route: {record['route_id']}")
            if not math.isfinite(record["scores"]["score_composed"]):
                raise ValueError("Non-finite driving score")
            records[route_id] = copy.deepcopy(record)
    missing = expected - set(records)
    if missing:
        raise ValueError(f"Missing {len(missing)} of 220 routes; resume the evaluation")
    ordered = [records[r] for r in sorted(records, key=int)]
    successes = sum(r["status"] in {"Completed", "Perfect"}
                    and all(not v for k, v in r["infractions"].items() if k != "min_speed_infractions")
                    for r in ordered)
    ds = sum(r["scores"]["score_composed"] for r in ordered) / 220
    # Preserve the official merged.json SR scale (0..1); print/report percentages separately.
    merged = {"_checkpoint": {"records": ordered}, "driving score": ds,
              "success rate": successes / 220, "eval num": 220}
    (output / "merged.json").write_text(json.dumps(merged, indent=2) + "\n")
    summary = {"routes": 220, "DS": ds, "SR_percent": 100 * successes / 220}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="evaluation directory containing result_shard*.json")
    aggregate(parser.parse_args().output)
