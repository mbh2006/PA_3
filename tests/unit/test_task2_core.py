"""Task 2: visual insertion, example-weighted loss, sampler, grounding scoring."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from _task1_fakes import FakeTokenizer
from task2.connector import Connector, N_VISUAL_TOKENS, calibrate_magnitude
from task2.grounding import (
    build_counting_pairs,
    build_presence_pairs,
    counting_question,
    pair_accuracy,
    presence_question,
    vqa_soft_consensus,
)
from task2.inputs import build_inputs
from task2.loss_sampler import (
    STAGE2_MIX,
    STAGE3_MIX,
    MixtureSampler,
    example_weighted_loss,
    realized_fractions,
    reference_answer_nll,
)

IM_END, IM_START = 3, 4


# ---------------------------------------------------------------- connector -------

def test_connector_output_shape_and_dtype():
    conn = Connector()
    x = torch.randn(196, 768)
    y = conn(x)
    assert y.shape == (196, 1024)
    assert y.dtype == torch.float32


def test_calibration_matches_rms():
    torch.manual_seed(0)
    conn = Connector()
    feats = torch.randn(16, 196, 768)
    text = torch.randn(500, 1024) * 0.3
    cal = calibrate_magnitude(conn, feats, text)
    assert cal.scale > 0
    assert "training data only" in cal.rule
    out = conn(feats[:1]).reshape(-1) * cal.scale
    assert abs(float(out.pow(2).mean().sqrt()) - cal.text_rms) < 1e-3


# ---------------------------------------------------------------- inputs ----------

def _embedder(vocab=256, dim=16):
    torch.manual_seed(0)
    return nn.Embedding(vocab, dim)


def test_visual_tokens_inserted_before_question_and_masked():
    tok = FakeTokenizer()
    emb = _embedder()
    vis = torch.randn(N_VISUAL_TOKENS, 16)
    out = build_inputs(emb, tok, question="What is shown?", answer="a cat",
                       visual_embeds=vis, im_start_id=IM_START, im_end_id=IM_END)

    assert out.n_visual == N_VISUAL_TOKENS
    assert out.labels.shape == (1, out.seq_len)
    assert out.position_ids.tolist() == [list(range(out.seq_len))]

    # every supervised position is an answer token or the EOT
    supervised = (out.labels[0] != -100)
    assert int(supervised.sum()) == out.n_answer
    assert int(out.labels[0, -1]) == IM_END
    # the visual span is never supervised
    assert not bool(supervised[:N_VISUAL_TOKENS].any())


def test_no_image_prefix_removes_exactly_the_visual_tokens():
    tok = FakeTokenizer()
    emb = _embedder()
    vis = torch.randn(N_VISUAL_TOKENS, 16)
    with_img = build_inputs(emb, tok, question="q?", answer="a",
                            visual_embeds=vis, im_start_id=IM_START, im_end_id=IM_END)
    no_img = build_inputs(emb, tok, question="q?", answer="a",
                          visual_embeds=None, im_start_id=IM_START, im_end_id=IM_END)
    assert no_img.seq_len == with_img.seq_len - N_VISUAL_TOKENS
    assert no_img.n_visual == 0
    assert no_img.position_ids.tolist() == [list(range(no_img.seq_len))]
    assert int(no_img.labels[0, -1]) == IM_END


def test_answer_nll_mode_excludes_termination():
    tok = FakeTokenizer()
    emb = _embedder()
    out = build_inputs(emb, tok, question="q?", answer="abc",
                       visual_embeds=torch.randn(N_VISUAL_TOKENS, 16),
                       im_start_id=IM_START, im_end_id=IM_END, eot_supervised=False)
    supervised = out.labels[0] != -100
    assert int(supervised.sum()) == len("abc")
    assert int(out.labels[0, -1]) == -100


def test_wrong_visual_count_rejected():
    tok = FakeTokenizer()
    with pytest.raises(ValueError):
        build_inputs(_embedder(), tok, question="q", answer="a",
                     visual_embeds=torch.randn(10, 16), im_start_id=IM_START, im_end_id=IM_END)


# ---------------------------------------------------------------- loss ------------

def test_example_weighted_loss_matches_hand_computation():
    torch.manual_seed(0)
    vocab = 20
    logits = torch.randn(2, 6, vocab)
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[0, 2:5] = torch.tensor([3, 4, 5])
    labels[1, 2:6] = torch.tensor([6, 7, 8, 9])

    got = example_weighted_loss(logits, labels)

    expected_terms = []
    for i, n in ((0, 3), (1, 4)):
        ce = torch.nn.functional.cross_entropy(
            logits[i, :-1], labels[i, 1:], ignore_index=-100, reduction="sum")
        expected_terms.append(ce / n)
    expected = torch.stack(expected_terms).mean()
    assert torch.allclose(got, expected, atol=1e-6)


def test_example_weighted_differs_from_token_weighted():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 10)
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[0, 2:4] = torch.tensor([3, 4])          # 2 tokens
    labels[1, 2:6] = torch.tensor([5, 6, 7, 8])    # 4 tokens
    example_w = example_weighted_loss(logits, labels)
    token_w = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 10), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert not torch.allclose(example_w, token_w, atol=1e-4)


def test_reference_nll_is_per_example_mean():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 12)
    labels = torch.full((1, 5), -100, dtype=torch.long)
    labels[0, 1:4] = torch.tensor([2, 3, 4])
    nll = reference_answer_nll(logits, labels)
    manual = torch.nn.functional.cross_entropy(
        logits[0, 0:3], labels[0, 1:4], reduction="sum") / 3
    assert torch.allclose(nll[0], manual, atol=1e-6)


def test_empty_example_is_excluded_from_mean():
    logits = torch.randn(2, 4, 8)
    labels = torch.full((2, 4), -100, dtype=torch.long)
    labels[1, 1:3] = torch.tensor([2, 3])
    only_second = example_weighted_loss(logits, labels)
    single = example_weighted_loss(logits[1:2], labels[1:2])
    assert torch.allclose(only_second, single, atol=1e-6)


# ---------------------------------------------------------------- sampler ----------

def _pools(n=40):
    pairs = [[{"id": f"p{i}", "image": i + 1, "answer": "yes"},
              {"id": f"p{i}", "image": -(i + 1), "answer": "no"}] for i in range(n)]
    captions = [{"id": f"c{i}", "caption": f"caption {i}"} for i in range(n)]
    return {"presence": pairs, "counting": pairs, "natural": pairs, "caption": captions}


def test_every_draw_returns_exactly_two_examples():
    s = MixtureSampler("stage2", _pools(), seed=6304)
    for _ in range(200):
        kind, ex = s.draw()
        assert len(ex) == 2, kind


def test_question_pairs_are_preserved():
    s = MixtureSampler("stage2", _pools(), seed=6304)
    for _ in range(200):
        kind, ex = s.draw()
        if kind != "caption":
            assert ex[0]["id"] == ex[1]["id"]
            assert ex[0]["image"] != ex[1]["image"]


def test_sampler_is_deterministic_and_resumable():
    a = MixtureSampler("stage2", _pools(), seed=6304)
    b = MixtureSampler("stage2", _pools(), seed=6304)
    seq_a = [a.draw()[0] for _ in range(50)]
    seq_b = [b.draw()[0] for _ in range(50)]
    assert seq_a == seq_b

    c = MixtureSampler("stage2", _pools(), seed=6304)
    for _ in range(20):
        c.draw()
    state = c.save_state()
    rest_c = [c.draw()[0] for _ in range(10)]

    d = MixtureSampler("stage2", _pools(), seed=6304)
    for _ in range(20):
        d.draw()
    d.load_state(state)
    rest_d = [d.draw()[0] for _ in range(10)]
    assert rest_c == rest_d


def test_mixture_is_not_enforced_as_quotas_but_matches_weights():
    s = MixtureSampler("stage3", _pools(), seed=6304)
    for _ in range(4000):
        s.draw()
    frac = realized_fractions(s.state.counts)
    for k, target in STAGE3_MIX.items():
        assert abs(frac[k] - target) < 0.05, (k, frac[k], target)


def test_caption_draws_use_two_independent_captions():
    s = MixtureSampler("stage2", _pools(), seed=1)
    kinds = []
    while "caption" not in kinds:
        kind, ex = s.draw()
        if kind == "caption":
            assert len(ex) == 2
            kinds.append(kind)


# ---------------------------------------------------------------- grounding -------

def test_question_templates():
    assert presence_question("dog") == "Does this image contain dog?"
    assert counting_question("dog") == "How many dogs can you see?"
    assert counting_question("person") == "How many persons can you see?"


def test_presence_pairs_require_absent_category():
    annotations = {1: ["dog"], 2: ["cat"], 3: ["dog", "cat"]}
    pairs = build_presence_pairs(annotations, ["dog"], "train")
    assert len(pairs) == 1
    p = pairs[0]
    assert {p.image_a, p.image_b} == {1, 2} or {p.image_a, p.image_b} == {3, 2}
    assert (p.answer_a, p.answer_b) == ("yes", "no")
    assert p.image_a != p.image_b


def test_counting_pairs_need_distinct_counts_in_range():
    counts = {1: {"dog": 2}, 2: {"dog": 3}, 3: {"dog": 3}, 4: {"dog": 7}}
    pairs = build_counting_pairs(counts, ["dog"], "train")
    assert pairs, "expected at least one pair"
    for p in pairs:
        assert p.answer_a != p.answer_b
        assert 1 <= int(p.answer_a) <= 4 and 1 <= int(p.answer_b) <= 4


def test_counting_pairs_ignore_out_of_range_counts():
    counts = {1: {"dog": 5}, 2: {"dog": 6}}
    assert build_counting_pairs(counts, ["dog"], "train") == []


@pytest.mark.parametrize("pred,refs,expected", [
    ("cat", ["cat"] * 10, 1.0),
    ("cat", ["dog"] * 10, 0.0),
    # official VQA quirk: leave-one-out means even a minority prediction can score highly
    ("cat", ["cat"] * 1 + ["dog"] * 9, 0.3),
    ("cat", ["cat"] * 2 + ["dog"] * 8, 0.6),
    ("cat", ["cat"] * 3 + ["dog"] * 7, 0.9),
    ("cat", ["cat"] * 4 + ["dog"] * 6, 1.0),
    (None, ["cat"] * 10, 0.0),
    ("", ["cat"] * 10, 0.0),
])
def test_vqa_soft_consensus(pred, refs, expected):
    assert abs(vqa_soft_consensus(pred, refs) - expected) < 1e-9


def test_vqa_soft_consensus_leave_one_out_is_not_min_count_over_three():
    # a naive min(count/3, 1) would give 1.0 for three matches; the official
    # leave-one-out formula gives 0.9 because the matching annotator is excluded
    refs = ["cat"] * 3 + ["dog"] * 7
    assert abs(vqa_soft_consensus("cat", refs) - 0.9) < 1e-9
    assert vqa_soft_consensus("cat", refs) != min(3 / 3, 1.0)


def test_vqa_soft_consensus_caps_at_one_per_annotator():
    # 8 of the other 9 annotators agree with the prediction -> each term saturates at 1
    assert vqa_soft_consensus("cat", ["cat"] * 9 + ["dog"]) == 1.0


def test_vqa_requires_ten_references():
    with pytest.raises(ValueError):
        vqa_soft_consensus("cat", ["cat"] * 9)


def test_normalisation_is_case_and_space_insensitive():
    assert vqa_soft_consensus("  Cat ", ["cat"] * 10) == 1.0
    assert pair_accuracy("  Cat ", "cat", "Cat", "cat") == 1.0
    assert pair_accuracy("cat", "dog", "cat", "cat") == 0.0
