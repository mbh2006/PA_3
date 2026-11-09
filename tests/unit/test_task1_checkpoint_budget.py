"""Checkpoint retention under a disk budget."""

import time
from pathlib import Path

from task1.run import _enforce_checkpoint_budget


def _fake_checkpoint(path: Path, mb: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * int(mb * 1_000_000))
    return path


def test_budget_keeps_everything_when_it_fits(tmp_path):
    ck = tmp_path / "checkpoints"
    for i in (312, 625):
        _fake_checkpoint(ck / f"checkpoint_{i:04d}.pt", 0.5)
        time.sleep(0.01)
    log: list[dict] = []
    _enforce_checkpoint_budget(ck, max_gb=10.0, log=log)
    assert len(list(ck.glob("checkpoint_*.pt"))) == 2
    assert log == []


def test_budget_removes_oldest_first_and_logs(tmp_path):
    ck = tmp_path / "checkpoints"
    for i in (312, 625, 937):
        _fake_checkpoint(ck / f"checkpoint_{i:04d}.pt", 0.5)
        time.sleep(0.01)
    log: list[dict] = []
    _enforce_checkpoint_budget(ck, max_gb=0.001, log=log)
    remaining = sorted(p.name for p in ck.glob("checkpoint_*.pt"))
    assert remaining == ["checkpoint_0625.pt", "checkpoint_0937.pt"]
    assert len(log) == 1
    assert log[0]["removed"] == "checkpoint_0312.pt"
    assert "disk budget" in log[0]["reason"]


def test_budget_never_deletes_the_last_checkpoint(tmp_path):
    ck = tmp_path / "checkpoints"
    _fake_checkpoint(ck / "checkpoint_1250.pt", 2.0)
    log: list[dict] = []
    _enforce_checkpoint_budget(ck, max_gb=0.001, log=log)
    assert len(list(ck.glob("checkpoint_*.pt"))) == 1
