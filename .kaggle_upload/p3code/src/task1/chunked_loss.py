"""Memory-bounded language-model loss.

The 151,936-token vocabulary makes the output projection dominate GPU memory: fp16
logits plus the fp32 cross-entropy workspace cost ~2.6 MB per token, so a single
1,942-token example needs ~5 GB on top of the model and optimiser state and OOMs a
16 GB T4.

This module computes exactly the same quantity — ``sum_CE / total_valid`` — while
materialising the logits for at most ``chunk_size`` positions at a time:

1. run the backbone once and keep its graph (hidden states only),
2. detach those hidden states and accumulate ``lm_head`` + CE **per sequence chunk**
   into ``h.grad`` (each chunk's graph is freed immediately),
3. propagate ``h.grad`` through the backbone once.

The gradient is identical to the standard ``model(labels=...)`` path; the cost is one
extra backward through the backbone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _hidden_states(model, input_ids, attention_mask):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
    )
    if getattr(out, "hidden_states", None) is None:
        raise ValueError("model did not return hidden states")
    return out.hidden_states[-1]


def backward_chunked_lm_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    total_valid: int,
    loss_scale: float = 1.0,
    chunk_size: int = 256,
) -> tuple[float, int]:
    """Backpropagate ``loss_scale * sum_CE / total_valid`` with bounded memory.

    Returns ``(sum_ce, n_valid)`` for logging; gradients are already accumulated.
    """
    if total_valid <= 0:
        raise ValueError("effective batch has no supervised tokens")

    h = _hidden_states(model, input_ids, attention_mask)
    h_det = h.detach().requires_grad_(True)
    head = model.get_output_embeddings()
    if head is None:
        raise ValueError("model has no output embeddings")

    seq_len = h_det.size(1)
    sum_ce = torch.zeros((), device=h_det.device, dtype=torch.float32)
    n_valid = 0

    for start in range(0, max(seq_len - 1, 1), chunk_size):
        end = min(start + chunk_size, seq_len - 1)
        if end <= start:
            break
        logits = head(h_det[:, start:end])
        target = labels[:, start + 1:end + 1]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_valid += int((target != -100).sum())
        sum_ce = sum_ce + ce.detach().float()
        (ce * (loss_scale / total_valid)).backward()

    if h_det.grad is not None:
        torch.autograd.backward(h, grad_tensors=h_det.grad)
    return float(sum_ce), n_valid
