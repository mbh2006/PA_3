# ATML PA 3 — Task 1 technical record

**Task:** Large language model fine-tuning — Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16 (PDF pp. 2–6)
**Status:** implementation complete and unit-tested; training runs and the seven evaluations **not yet executed** (awaiting GPU).
**Owner of interpretation:** the student.

---

## 1. Implementation map

| Requirement | Module | Notes |
|---|---|---|
| First 40,000 eligible MetaMathQA rows in source order | `src/task1/data.py::select_examples` | exclusion counters per reason; no shuffle; never truncates |
| Empty question/response exclusion | same | `excluded_empty_query`, `excluded_empty_response` |
| GSM8K test-question exclusion after normalisation | `normalize_question` | NFKC, lower-case, `$`→space, whitespace collapse, trailing punctuation stripped; matches on `query` **or** `original_question` |
| >2,048-token exclusion | same | length measured on the fully formatted chat sequence |
| Chat formatting, thinking disabled | `format_example` | `apply_chat_template(..., enable_thinking=False)` |
| Supervise assistant tokens + end-of-turn | `format_example` | prompt masked; everything after the last `<|im_end|>` masked (the separator newline is not part of the turn) |
| Token-weighted loss over the effective batch | `src/task1/train.py::update_once` | per-micro-batch mean × (n_valid_mb / n_valid_effective_batch); summing gives ΣCE/Σtokens |
| FP32 master parameters, FP16 compute | `model.py`, `train.py` | model loaded in FP32; `torch.autocast(float16)`; `GradScaler` |
| AdamW, wd 0.01 excluding biases and <2-D | `optim.py::build_param_groups` | two groups; no-decay for 1-D and biases |
| Cosine, 38 warm-up updates | `optim.py::lr_at` | linear warm-up then cosine to 0; one scheduler step per optimizer update |
| Global-norm clip 1.0, unscale first | `train.py` | unscale → L2-SP grad → clip → step |
| Skipped updates | `train.py` | detected via scaler scale drop; logged; the same effective batch is retried so 1,250 real updates happen |
| Recovery checkpoints | `train.py::save_checkpoint` | model/adapter, optimizer, scheduler, scaler, RNG, update, examples seen |
| L2-SP Ω | `l2sp.py` | `0.5·Σ(θ−θ0)²` over all unique params (tied counted once); reference fixed in FP32 |
| L2-SP gradient scaling | `l2sp.py::add_gradient` | `λ(θ−θ0)` added **once per update**, never multiplied by accumulation |
| LoRA r/α, dropout, targets | `lora.py` | r=4/α=8, r=16/α=32; dropout 0.05; random A, zero B; no trainable biases; 7 projections × every layer |
| OPLoRA k=128 | `lora.py` | FP32 SVD bases computed once; `B_eff`, `A_eff`; no dense projection; projected factors exported by `adapter_state` |
| Merge check | `lora.py::merge_adapters` | merges `scaling·B_eff·A_eff` into the base weight |
| Diagnostics | `diagnostics.py` | ρ_k, η, R(z); zero update → `rho_k=None` |
| Displacement, Ω logging | `train.py`, `l2sp.py` | ‖θ−θ0‖₂ and λΩ recorded |

## 2. Verification (CPU)

`tests/unit/test_task1_data.py`, `test_task1_loss.py`, `test_task1_l2sp_optim.py`,
`test_task1_lora.py` — 40 tests, all passing.

Hand-checked properties:

* prompt masked, response + EOT supervised, nothing after the turn supervised;
* the token-weighted accumulation reproduces a single-batch `ΣCE/Σtokens` reference to
  `1e-6` with deliberately unequal example lengths;
* the Trainer's own `update_once` gradient equals that analytic reference;
* Ω = 0 at start, equals `0.5·Σd²` after a known perturbation, includes biases/norms,
  counts tied parameters once;
* `λ(θ−θ0)` gradient equals splitting λ across 16 micro-batches (i.e. not multiplied);
* LoRA forward equals the manual expression; OPLoRA residuals `U_k^T B_eff` and
  `A_eff V_k` vanish; `k = min(d_out,d_in)` gives a zero update;
* LoRA-16 and OPLoRA-16 get identical initial A factors from the same seed;
* LR: `peak/38` at step 0, peak at 37–38, 0 at 1,250.

## 3. Measured GPU behaviour (Kaggle T4, 15.6 GB)

Real runs of the prescribed recipe, not estimates. Source:
`outputs/environment/kaggle_gpu_test_v3.json` (kernel `mbh2006/atml-pa3-gpu-test`, v3).
Both variants completed one full optimizer update (16 micro-batches of 2) on real
MetaMathQA rows.

| Quantity | Full FT | LoRA-16 |
|---|---|---|
| Trainable parameters | 596,049,920 | 10,092,544 (backbone frozen) |
| Parameter dtypes | FP32 | FP16 backbone + FP32 adapters |
| Gradient dtype | FP32 | FP32 |
| Optimizer-state dtype | FP32 | FP32 |
| Peak allocated / reserved | **13.18 / 14.02 GB** | **8.49 / 10.29 GB** |
| Seconds per optimizer update | 6.2–8.2 s | 4.5 s |
| Throughput | 850–1,128 tokens/s | 1,548 tokens/s |
| Estimated 1,250-update run | **2.2–2.9 h** | **1.6 h** |
| SFT loss after one update | 1.028 | 1.029 |
| Grad norm | 33.3 (clipping active) | 4.16 |
| Skipped update | no | no |
| Environment | torch 2.11.0+cu130, CUDA 13.0, transformers 4.57.1, T4 capability 7.5 | same |

**Conclusion:** the prescribed configuration fits on a free Kaggle T4 with ~1.5 GB of
headroom for full fine-tuning and ~5 GB for adapters. Memory is *not* the limitation;
weekly GPU quota is. The per-update time varied between the two kernels (6.2 s vs 8.2 s),
so the run estimate is reported as a range.

The LoRA-16 variant failed on the first attempt with
`Expected a 'cuda' device type for generator but found 'cpu'`: the adapter `A` factor was
allocated on the GPU while the init generator lived on the CPU. Fixed by initialising `A`
on the CPU and moving it to the parameter device; the unit suite still passes and the
re-run above succeeded.

## 4. Not yet measured

Everything that needs full runs: the five training runs, the seven evaluation conditions,
the four benchmarks, the activation probes, and the 168 qualitative generations.
No Task 1 benchmark number may be reported until those exist.

## 5. Deviations to disclose

1. The trailing newline after `<|im_end|>` is masked (it is turn-separator formatting,
   not response content).
2. If a scaler-skipped update occurs, the same effective batch is retried so that the
   prescribed 1,250 updates and 40,000 examples are still consumed; this is logged.
3. LoRA/OPLoRA are implemented directly rather than via `peft`, to keep the backbone in
   FP16 and the adapters in FP32 as the assignment specifies.
