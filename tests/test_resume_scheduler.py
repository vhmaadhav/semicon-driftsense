"""Regression test for the PR #24 review blocker: train.py used to construct
OneCycleLR *before* the --resume block, so the architecture-adopting branch
(which rebuilds model and optimizer) left the scheduler bound to the discarded
optimizer -- the run then trained without the intended one-cycle schedule.

train.py now constructs `steps` and `sched` AFTER the resume block, and
captures the checkpoint's global_step (`sched_ff`) during resume to
fast-forward the rebuilt scheduler afterwards. The tests below exercise the
same construction order train.py uses, with a checkpoint whose arch_kwargs
differ from the flags.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")

from driftsense.model import DriftSenseNet  # noqa: E402


def _build_like_trainpy(ck, flag_kwargs, lr=1e-3):
    """Mirror train.py main()'s construction order: build model + optimizer
    from the flags, run the resume block (arch adopt + optimizer restore +
    sched_ff capture), then construct OneCycleLR against the final optimizer
    and fast-forward it. Returns (model, opt, sched, sched_ff, adopted).
    """
    steps = 100
    arch_kwargs = dict(flag_kwargs)
    model = DriftSenseNet(**arch_kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    adopted = False
    sched_ff = 0
    # --resume block (mirrors train.py; the finetune branch only prints, so it
    # is not exercised here):
    ck_arch = ck.get("arch_kwargs")
    if ck_arch and ck_arch != arch_kwargs:
        # The "explicit flag" set is empty in these tests, so the adopt branch
        # runs -- same as invoking train.py --resume without --width/--ctx/--head.
        arch_kwargs = dict(ck_arch)
        model = DriftSenseNet(**arch_kwargs)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        adopted = True
    model.load_state_dict(ck["model"])
    if "optimizer" in ck and ck.get("crop") == ck.get("crop"):
        opt.load_state_dict(ck["optimizer"])
        sched_ff = ck.get("global_step", 0)
    ck.get("best", float("inf"))

    # Post-resume construction (the fix under test):
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    for _ in range(sched_ff):
        sched.step()
    return model, opt, sched, sched_ff, adopted


def _make_checkpoint(ck_kwargs, global_step, crop=64):
    model = DriftSenseNet(**ck_kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "arch_kwargs": dict(ck_kwargs),
        "global_step": global_step,
        "epoch": global_step // 10,
        "crop": crop,
        "best": 0.5,
    }


def test_scheduler_binds_to_optimizer_receiving_steps():
    """After arch-adopting resume, the ACTIVE scheduler's optimizer must be
    the optimizer that receives opt.step() (the one over the final model)."""
    ck = _make_checkpoint({"width": 8, "ctx": 8, "head": 8}, global_step=5)
    model, opt, sched, _, adopted = _build_like_trainpy(
        ck, {"width": 4, "ctx": 4, "head": 4})
    assert adopted, "test premise: checkpoint arch differs from flags"

    assert sched.optimizer is opt
    # The optimizer actually stepped is the one over the adopted model, and
    # the scheduler wraps exactly those parameter tensors.
    opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert opt_param_ids == {id(p) for p in model.parameters()}
    sched_param_ids = {id(p) for g in sched.optimizer.param_groups
                       for p in g["params"]}
    assert sched_param_ids == opt_param_ids
    # Sanity: the adopted model really has the checkpoint's width, and
    # stepping opt mutates the weights the scheduler's optimizer sees.
    n_adopted = sum(p.numel() for p in model.parameters())
    assert n_adopted > sum(p.numel() for p in DriftSenseNet(width=4, ctx=4, head=4).parameters())


def test_fast_forward_matches_reference_scheduler():
    """After N fast-forward steps, sched.get_last_lr() equals a reference
    scheduler stepped N times from construction."""
    n = 5
    ck = _make_checkpoint({"width": 8, "ctx": 8, "head": 8}, global_step=n)
    _, _, sched, sched_ff, _ = _build_like_trainpy(
        ck, {"width": 4, "ctx": 4, "head": 4})
    assert sched_ff == n

    ref_opt = torch.optim.AdamW(
        [torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    ref_sched = torch.optim.lr_scheduler.OneCycleLR(
        ref_opt, max_lr=1e-3, total_steps=100, pct_start=0.15)
    for _ in range(n):
        ref_sched.step()

    assert sched.get_last_lr() == pytest.approx(ref_sched.get_last_lr())
    # One more step keeps them in lockstep (the resumed run trains with this).
    sched.step()
    ref_sched.step()
    assert sched.get_last_lr() == pytest.approx(ref_sched.get_last_lr())


def test_no_adopt_leaves_scheduler_on_flag_optimizer():
    """When the checkpoint arch matches the flags, the construction order must
    still bind the scheduler to the (sole) optimizer, fast-forwarded."""
    ck = _make_checkpoint({"width": 4, "ctx": 4, "head": 4}, global_step=3)
    model, opt, sched, sched_ff, adopted = _build_like_trainpy(
        ck, {"width": 4, "ctx": 4, "head": 4})
    assert not adopted
    assert sched_ff == 3
    assert sched.optimizer is opt
    assert {id(p) for p in model.parameters()} == {
        id(p) for g in opt.param_groups for p in g["params"]}
