"""Arrange generated per-route control metrics for the benchmark scorer."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    records = json.loads((output / "merged.json").read_text())["_checkpoint"]["records"]
    candidates = list(output.glob("frames_*/Scenarios/*/metric_info.json"))
    pairs = []
    for record in records:
        name = record["save_name"]
        if Path(name).name != name:
            raise ValueError("Invalid route save name")
        matches = [p for p in candidates if p.parent.name.endswith("_" + name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one metric file for {name}, found {len(matches)}")
        pairs.append((name, matches[0]))
    for name, source in pairs:
        target = output / "route_metrics" / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / "metric_info.json")
    print(f"Prepared {len(pairs)} route metric files")


if __name__ == "__main__":
    main()
