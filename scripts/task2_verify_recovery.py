"""Verify Task 2 evidence downloaded from the private Kaggle job."""

import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def main(root, bundle):
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    config = json.loads((bundle / "recovery_config.json").read_text(encoding="utf-8-sig"))
    provenance = read("provenance.json")
    assert provenance["checkpoint_sha256"] == config["checkpoint_sha256"]
    for name, expected in config["checkpoint_sha256"].items():
        with (bundle / name).open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == expected
    assert provenance["batch_size"] == 8 and provenance["max_new_tokens"] == 16
    conditions = {"correct", "complementary", "mismatched", "no_image"}
    smoke = read("gpu_smoke.json")
    assert smoke.keys() == conditions
    assert all(row["batch_size"] == row["matching_predictions"] == 8 for row in smoke.values())
    splits = {key: set(ids) for key, ids in read("split_images.json").items()}
    assert not (splits["train"] & splits["dev"] or splits["train"] & splits["test"]
                or splits["dev"] & splits["test"])
    metrics = read("metrics.json")
    final_ids = set()
    for suite, count, refs in (("grounding", 128, 1), ("natural", 192, 10)):
        rows = read(f"{suite}_predictions.json")
        assert len(rows) == len({(r["pair_id"], r["side"]) for r in rows}) == count
        assert len({r["pair_id"] for r in rows}) == count // 2
        for row in rows:
            images = row["images"]
            assert images.keys() == row["conditions"].keys() == conditions
            assert len(set(images.values())) == 4 and images["no_image"] is None
            assert len(row["references"]) == refs
            assert all(1 <= len(result["generated_ids"]) <= 16 and
                       math.isfinite(result["reference_nll"]) and 0 <= result["score"] <= 1
                       for result in row["conditions"].values())
            final_ids.add(images["correct"])
        for cond in conditions:
            saved = metrics[suite]["overall"]["conditions"][cond]
            mean_score = sum(row["conditions"][cond]["score"] for row in rows) / count
            assert saved["n_answers"] == count and math.isclose(saved["accuracy"]["mean"], mean_score)
    assert final_ids == splits["test"]
    assert read("evaluation_resources.json")["predictions"] == 1280
    captions = read("caption_dev_endpoints.json")
    assert {row["update"] for row in captions} == {3750, 4500}
    for row in captions:
        losses = row["image_id_and_loss"]
        assert row["n_examples"] == len(losses) == len({img for img, _ in losses}) == 96
        assert {img for img, _ in losses} <= splits["dev"]
        # The saved torch mean can be rounded to FP16 before JSON conversion.
        assert math.isclose(row["mean_caption_loss"], sum(loss for _, loss in losses) / 96, rel_tol=5e-4)
    with (root / "condition_table.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 28
    print(f"Verified 1,280 predictions, {len(table)} condition rows and 2 caption dev endpoints.")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
