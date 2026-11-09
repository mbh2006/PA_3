"""Probe the Kaggle API for quota information (read-only, no secrets printed)."""

import json
import os
import urllib.error
import urllib.request

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

cfg = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
user, key = cfg["username"], cfg["key"]

urls = [
    "https://www.kaggle.com/api/v1/kernels/quota",
    "https://www.kaggle.com/api/v1/account",
    "https://www.kaggle.com/api/v1/users/self",
    "https://www.kaggle.com/api/v1/kernels/list?user=mbh2006&pageSize=1",
]

for url in urls:
    for auth in ("basic", "none"):
        try:
            req = urllib.request.Request(url)
            if auth == "basic":
                import base64

                token = base64.b64encode(f"{user}:{key}".encode()).decode()
                req.add_header("Authorization", f"Basic {token}")
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read()[:300]
                print(f"{url} [{auth}] -> {resp.status} {body!r}")
                break
        except Exception as exc:
            print(f"{url} [{auth}] -> {type(exc).__name__}: {str(exc)[:100]}")
