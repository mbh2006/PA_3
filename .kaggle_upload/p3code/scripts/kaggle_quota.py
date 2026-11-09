"""Record the Kaggle GPU/CPU/TPU quota as evidence (authenticated, secrets never printed)."""

import base64
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "environment" / "kaggle_quota.json"

cfg = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
token = base64.b64encode(f"{cfg['username']}:{cfg['key']}".encode()).decode()

req = urllib.request.Request("https://www.kaggle.com/api/v1/kernels/quota")
req.add_header("Authorization", f"Basic {token}")
with urllib.request.urlopen(req, timeout=20) as resp:
    payload = json.loads(resp.read())

payload["_fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
payload["_account"] = cfg["username"]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2) + "\n")

for key in ("gpuQuota", "tpuQuota"):
    q = payload.get(key)
    if not q:
        continue
    used = q["timeUsed"]["seconds"] + q["timeUsed"]["nanos"] / 1e9
    total = q["totalTimeAllowed"]["seconds"] / 3600
    print(f"{key}: used {used / 3600:.3f} h of {total:.1f} h  "
          f"({used:.0f} s)  refresh {payload.get('quotaRefreshTime')}")
print("wrote", OUT)
