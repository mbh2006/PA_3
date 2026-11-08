"""Fetch small reference assets that are needed for Task 3 scoring.

Only metadata is fetched (no images, no model weights):
  * MathVista `testmini.json`  -- question metadata (answer_type, choices, precision, unit)
  * `first_plot_questions.yaml` -- Task 1 qualitative prompts (8 questions x 3 formats)

Every download is hashed and recorded in `data/reference/ASSETS.json` so the exact
bytes used for scoring can be verified later. `data/` is gitignored; re-run this
script on a fresh checkout to reproduce the environment.

Run:  .venv/Scripts/python.exe scripts/fetch_reference_assets.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "reference"

ASSETS = {
    "mathvista_testmini.json": {
        "url": "https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json",
        "note": "MathVista testmini question metadata (answer_type, choices, precision, unit).",
    },
    "first_plot_questions.yaml": {
        "url": "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml",
        "note": "Task 1 qualitative prompts: 8 questions in plain/JSON/template formats (24 entries).",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "ASSETS.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for name, spec in ASSETS.items():
        dest = OUT / name
        if dest.exists():
            print(f"[skip] {name} already present")
        else:
            print(f"[get ] {name} <- {spec['url']}")
            with urllib.request.urlopen(spec["url"], timeout=60) as resp:
                dest.write_bytes(resp.read())
        manifest[name] = {
            "url": spec["url"],
            "note": spec["note"],
            "bytes": dest.stat().st_size,
            "sha256": sha256(dest),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"       {dest.stat().st_size:,} bytes  sha256={manifest[name]['sha256'][:16]}...")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
