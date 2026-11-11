"""Download the missing COCO JPEGs needed by the Task 2 splits."""

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("/root/pa3/task2_data")
TRAIN_URL = "http://images.cocodataset.org/train2014/COCO_train2014_{:012d}.jpg"
VAL_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"


def main() -> None:
    needed = {int(i) for i in json.loads((DATA / "needed_images.json").read_text())}
    splits = json.loads((DATA / "split_images.json").read_text())
    split_of = {}
    for name in ("train", "dev", "test"):
        for i in splits[name]:
            split_of[int(i)] = name
    img_dir = DATA / "images"
    img_dir.mkdir(exist_ok=True)

    todo = [i for i in sorted(needed) if not (img_dir / f"{i:012d}.jpg").exists()]
    print(f"{len(needed)} needed, {len(todo)} missing", flush=True)

    def fetch(i):
        split = split_of.get(i, "train")
        url = (TRAIN_URL if split == "train" else VAL_URL).format(i)
        dest = img_dir / f"{i:012d}.jpg"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
            return 1
        except Exception as exc:
            print(f"  {i} failed: {exc}", flush=True)
            return 0

    with ThreadPoolExecutor(max_workers=16) as pool:
        done = sum(pool.map(fetch, todo))
    print(f"downloaded {done}/{len(todo)}", flush=True)

    index = {str(i): f"images/{i:012d}.jpg" for i in sorted(needed)}
    (img_dir / "index.json").write_text(json.dumps(index))

    still = [i for i in needed if not (img_dir / f"{i:012d}.jpg").exists()]
    if still:
        raise SystemExit(f"{len(still)} images still missing after download")
    print("ALL_IMAGES_PRESENT", flush=True)


if __name__ == "__main__":
    main()
