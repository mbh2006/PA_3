"""LoRA / OPLoRA adapters: initialisation, projection residuals, merge equivalence."""

import torch
import torch.nn as nn

from task1.diagnostics import eta, rho_k
from task1.lora import (
    LoRALinear,
    adapter_parameter_count,
    adapter_state,
    build_oploRA_bases,
    inject_adapters,
    load_adapter_state,
    merge_adapters,
    target_module_names,
)


class Block(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, 2 * d, bias=False)
        self.up_proj = nn.Linear(d, 2 * d, bias=False)
        self.down_proj = nn.Linear(2 * d, d, bias=False)


class TinyQwen(nn.Module):
    def __init__(self, d=8, n=2):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])


def test_targets_are_the_seven_projections():
    model = TinyQwen()
    names = target_module_names(model)
    assert len(names) == 7 * 2
    assert all(n.endswith(("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")) for n in names)


def test_lora_forward_matches_manual_expression():
    torch.manual_seed(0)
    base = nn.Linear(6, 5, bias=False)
    lin = LoRALinear(base, r=2, alpha=4, dropout=0.0)
    torch.nn.init.normal_(lin.A)
    torch.nn.init.normal_(lin.B)
    x = torch.randn(3, 6)
    expected = base(x) + ((x @ lin.A.T) @ lin.B.T) * (4 / 2)
    assert torch.allclose(lin(x), expected, atol=1e-6)


def test_dropout_is_active_in_train_mode_only():
    base = nn.Linear(6, 5, bias=False)
    lin = LoRALinear(base, r=2, alpha=4, dropout=0.5)
    torch.nn.init.normal_(lin.B)
    x = torch.randn(4, 6)
    lin.eval()
    a = lin(x)
    lin.train()
    b = lin(x)
    assert not torch.allclose(a, b)


def test_oploRA_projection_residuals_vanish():
    torch.manual_seed(0)
    base = nn.Linear(10, 8, bias=False)
    ul = torch.randn(8, 8)
    u, _ = torch.linalg.qr(ul)
    uk = u[:, :3]
    vh = torch.randn(3, 10)
    vk = torch.linalg.qr(vh.T)[0]
    lin = LoRALinear(base, r=4, alpha=8, dropout=0.0, uk=uk, vk=vk)
    torch.nn.init.normal_(lin.A)
    torch.nn.init.normal_(lin.B)

    b_eff = lin.effective_B()
    a_eff = lin.effective_A()
    assert torch.allclose(uk.T @ b_eff, torch.zeros_like(uk.T @ b_eff), atol=1e-5)
    assert torch.allclose(a_eff @ vk, torch.zeros_like(a_eff @ vk), atol=1e-5)


def test_oploRA_with_full_protection_rank_gives_zero_update():
    base = nn.Linear(6, 6, bias=False)
    u, _ = torch.linalg.qr(torch.randn(6, 6))
    vk = torch.linalg.qr(torch.randn(6, 6))[0]
    lin = LoRALinear(base, r=4, alpha=8, dropout=0.0, uk=u, vk=vk)
    torch.nn.init.normal_(lin.A)
    torch.nn.init.normal_(lin.B)
    assert torch.allclose(lin.delta_weight(), torch.zeros(6, 6), atol=1e-5)


def test_build_bases_shapes_for_non_square_weights():
    model = TinyQwen()
    bases = build_oploRA_bases(model, k=4)
    assert bases["layers.0.down_proj"][0].shape == (8, 4)     # U: (d_out, k)
    assert bases["layers.0.down_proj"][1].shape == (16, 4)    # V: (d_in, k)


def test_same_seed_gives_identical_initial_factors():
    torch.manual_seed(0)
    a_model = TinyQwen()
    b_model = TinyQwen()
    b_model.load_state_dict(a_model.state_dict())
    bases = build_oploRA_bases(a_model, k=4)

    inject_adapters(a_model, r=16, alpha=32, dropout=0.05, seed=6304)
    inject_adapters(b_model, r=16, alpha=32, dropout=0.05, seed=6304, oplora_bases=bases)

    sa, sb = adapter_state(a_model), adapter_state(b_model)
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    # zero B at init -> zero correction
    for name, module in a_model.named_modules():
        if isinstance(module, LoRALinear):
            assert torch.count_nonzero(module.effective_B()) == 0


def test_adapter_state_roundtrip():
    model = TinyQwen()
    inject_adapters(model, r=4, alpha=8, dropout=0.05, seed=1)
    state = adapter_state(model)
    for t in state.values():
        t.add_(1.0)
    load_adapter_state(model, state)
    after = adapter_state(model)
    for k in state:
        assert torch.equal(state[k], after[k])


def test_merge_equals_forward():
    torch.manual_seed(0)
    base = nn.Linear(6, 5, bias=False)
    lin = LoRALinear(base, r=2, alpha=4, dropout=0.0)
    torch.nn.init.normal_(lin.A)
    torch.nn.init.normal_(lin.B)
    x = torch.randn(4, 6)
    before = lin(x).clone()
    merge_adapters(type("M", (), {"named_modules": lambda self: [("lin", lin)]})())
    after = lin.base(x)
    assert torch.allclose(before, after, atol=1e-5)


def test_param_count_and_dtypes():
    model = TinyQwen()
    inject_adapters(model, r=4, alpha=8, dropout=0.05, seed=1)
    n = adapter_parameter_count(model)
    # per layer: q,k,v,o (8x8) + gate,up (16x8) + down (8x16); r=4
    per_layer = 4 * (4 * 8 + 8 * 4) + 3 * (4 * 8 + 16 * 4)
    assert n == per_layer * 2
    assert all(p.dtype == torch.float32 for p in model.parameters() if ".A" in "" or p.requires_grad)


def test_diagnostics_zero_update_is_none_and_eta_positive():
    delta = torch.zeros(4, 4)
    u = torch.eye(4)[:, :2]
    assert rho_k(delta, u) is None
    w0 = torch.eye(4)
    assert abs(eta(torch.eye(4) * 0.1, w0) - 0.1) < 1e-6
