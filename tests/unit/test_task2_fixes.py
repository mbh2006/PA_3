"""Task 2 correctness tests: sampler staging, caption cycling, prompt hygiene, scoring."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task2.data import normalize_answer
from task2.grounding import vqa_soft_consensus
from task2.loss_sampler import MixtureSampler
from task2.train import STAGES, CaptionCycler

import os

DATA = Path(os.environ.get("PA3_TASK2_DATA", "/root/pa3/task2_data"))


def _has_data() -> bool:
    try:
        return (DATA / "data.json").exists()
    except OSError:
        return False


# ------------------------------------------------------------------ samplers ------

def _caption_pool(n=200):
    return [{"question": "Describe the image.", "answer": f"caption {i}",
             "features": None, "pair_id": ("cap", i)} for i in range(n)]


def test_caption_cycler_makes_two_complete_passes():
    pool = _caption_pool(200)
    cyc = CaptionCycler(pool)
    seen = []
    for _ in range(200):          # 200 draws = 400 examples = two passes over 200
        _, ex = cyc.draw()
        assert len(ex) == 2
        seen.extend(ex)
    ids = [e["pair_id"][1] for e in seen]
    assert len(ids) == 400
    from collections import Counter
    assert Counter(ids) == {i: 2 for i in range(200)}


def test_caption_cycler_stage1_is_caption_only():
    stage1 = STAGES[0]
    assert stage1.mix == "captions_only"
    assert stage1.updates == 2250          # 2 x 18,000 / effective batch 16
    assert stage1.lora_lr is None          # connector only
    assert STAGES[1].lora_lr is not None and STAGES[2].lora_lr is not None


def test_sampler_mixtures_match_the_assignment():
    assert STAGES[1].mix == "stage2"
    assert STAGES[2].mix == "stage3"


def test_caption_budget_matches_two_passes():
    # 2,250 updates x 16 examples = 36,000 = 2 x 18,000
    assert STAGES[0].updates * 16 == 36000


# ------------------------------------------------------------------ data ----------

@pytest.mark.skipif(not _has_data(), reason="data not built")
def test_splits_are_image_disjoint():
    splits = json.loads((DATA / "split_images.json").read_text())
    train, dev, test = set(splits["train"]), set(splits["dev"]), set(splits["test"])
    assert not train & dev, f"train/dev overlap {len(train & dev)}"
    assert not train & test, f"train/test overlap {len(train & test)}"
    assert not dev & test, f"dev/test overlap {len(dev & test)}"


@pytest.mark.skipif(not _has_data(), reason="data not built")
def test_grounding_train_present_and_from_train_split():
    data = json.loads((DATA / "data.json").read_text())
    splits = json.loads((DATA / "split_images.json").read_text())
    train_imgs = set(splits["train"])
    assert len(data["grounding_train"]) >= 24
    kinds = {p["kind"] for p in data["grounding_train"]}
    assert kinds == {"presence", "counting"}
    for p in data["grounding_train"]:
        assert p["image_a"] in train_imgs and p["image_b"] in train_imgs


@pytest.mark.skipif(not _has_data(), reason="data not built")
def test_record_schemas():
    data = json.loads((DATA / "data.json").read_text())
    for p in data["grounding_test"]:
        assert "answer_a" in p and "answer_b" in p
        assert "consensus_a" not in p
    for p in data["vqa_test"]:
        assert len(p["answers_a"]) == 10 and len(p["answers_b"]) == 10


# ------------------------------------------------------------------ generation -----

def test_generation_prompt_contains_no_reference():
    """A prompt example carries no answer field, so nothing but the prompt is fed."""
    from task2.evaluate import prompt_example

    ex = prompt_example("How many cats?", None, 123)
    assert "answer" not in ex
    assert ex["question"] == "How many cats?"


def test_nll_example_does_carry_the_reference():
    from task2.evaluate import nll_example

    ex = nll_example("q", "42", None, 1)
    assert ex["answer"] == "42"


# ------------------------------------------------------------------ scoring --------

@pytest.mark.parametrize("pred,refs,expected", [
    ("cat", ["cat"] * 10, 1.0),
    ("cat", ["cat"] * 3 + ["dog"] * 7, 0.9),
    ("cat", ["cat"] + ["dog"] * 9, 0.3),
    (None, ["cat"] * 10, 0.0),
])
def test_soft_consensus_uses_all_ten_references(pred, refs, expected):
    assert abs(vqa_soft_consensus(pred, refs) - expected) < 1e-9


def test_soft_consensus_rejects_wrong_reference_count():
    with pytest.raises(ValueError):
        vqa_soft_consensus("cat", ["cat"] * 9)


# ------------------------------------------------------------------ checkpoint -----

def test_checkpoint_round_trips_adapter_optimizer_scaler_and_sampler(tmp_path):
    from task2.train import CurriculumTrainer

    torch.manual_seed(0)
    import torch.nn as nn

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def get_input_embeddings(self):
            return None

    class TinyConnector(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

    lm, conn = TinyLM(), TinyConnector()
    for p in conn.parameters():
        p.requires_grad_(True)
    pools = {"caption": _caption_pool(8), "presence": [], "counting": [], "natural": []}
    tr = CurriculumTrainer(lm, conn, tokenizer=None, pools=pools, out_dir=tmp_path)
    tr.sampler = CaptionCycler(pools["caption"])
    tr.sampler.draw()
    tr.sampler.draw()

    stage = STAGES[0]
    tr.optimizer = torch.optim.AdamW(conn.parameters(), lr=1e-3)
    tr.scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tr.save_checkpoint(stage, 250)
    payload = torch.load(path, map_location="cpu")
    for key in ("connector", "adapter", "optimizer", "scaler", "sampler", "update"):
        assert key in payload, key
    assert payload["update"] == 250
    assert payload["sampler"]["pos"] == 4      # two draws advanced the cursor by 4
