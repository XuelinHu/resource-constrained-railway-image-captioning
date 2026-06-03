from __future__ import annotations

import argparse
import json
from pathlib import Path

from railcap.config import load_config
from railcap.data.dataset import load_manifest
from railcap.knowledge.ontology import RailwayOntology
from railcap.metrics.domain_metrics import compute_domain_metrics


def load_predictions(path: str | Path) -> list[str]:
    predictions: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "prediction" not in item:
                raise ValueError(f"{path}:{line_no}: missing prediction field")
            predictions.append(str(item["prediction"]))
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate railway caption predictions.")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--manifest", type=Path, help="Override test manifest path.")
    parser.add_argument("--predictions", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw = cfg.raw

    manifest_path = args.manifest or Path(raw["data"]["test_manifest"])
    samples = load_manifest(manifest_path)
    predictions = load_predictions(args.predictions)
    ontology = RailwayOntology.from_yaml(raw["knowledge"]["ontology_path"])

    metrics = compute_domain_metrics(samples, predictions, ontology)
    print(json.dumps(metrics.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
