"""Print the variant-test report as a compact table."""

import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "outputs" / "environment" / "variant_tests.json"
data = json.loads(path.read_text())

print("verdict:")
print(json.dumps(data["verdict"], indent=2))
print()
print(f"{'test':34s} {'ok':5s} {'peak GB':>8s} {'sec':>6s}  note")
for name, entry in data.items():
    if not isinstance(entry, dict) or "test" not in entry:
        continue
    note = entry.get("error", "")
    if entry.get("split_micro_batches") is not None:
        note = f"micro_batches={entry['split_micro_batches']} {note}"
    print(f"{name:34s} {str(entry.get('ok')):5s} {entry.get('peak_mem_gb', ''):>8} "
          f"{entry.get('seconds', ''):>6}  {note[:70]}")
