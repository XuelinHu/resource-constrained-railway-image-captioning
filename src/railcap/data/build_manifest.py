from __future__ import annotations

import argparse
import json
from pathlib import Path

from railcap.data.schema import CaptionSample


def validate_manifest(input_path: Path, output_path: Path | None = None) -> int:
    valid_rows: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            CaptionSample.from_json(item)
            valid_rows.append(item)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for item in valid_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(valid_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and normalize railway caption JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = validate_manifest(args.input, args.output)
    print(f"validated {count} samples")


if __name__ == "__main__":
    main()
