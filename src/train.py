from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn

from src.core.config.config_train import load_train_cfg
from src.utils.seed_device import seed_everything, get_device
from src.training.optim_sched_factory import build_optimizer, build_baseline_scheduler
from src.training.run_context import RunContext
from src.training.run_io import bootstrap_io, close_writers
from src.training.checkpoints import CheckpointIO
from src.training.logging_hooks import (
    make_train_metrics_hook, make_val_metrics_hook, # make_tb_hook_optional
)
from src.core.hooks.hook_composition import HookList
from src.core.hooks.common_hooks import LambdaHook
from src.training.engine import Trainer, _grad_l2_norm
from src.data_.cifar10 import get_dataloaders
from src.models.resnet_cifar10 import resnet20

def parse_args():
    p = argparse.ArgumentParser("Baseline trainer")
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "auto"], help="override device")
    p.add_argument("--dry-run", action="store_true")
    # p.add_argument("--override", action="append", default=[])  # TODO optional future
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_train_cfg(args.config) # or load_train_cfg(args.config, overrides=parsed_overrides)

    # run context & IO
    run_ctx = RunContext([args.config], cfg) # sets run_dir, start_time, git etc.
    io_ctx = bootstrap_io(run_ctx) # writers & paths
    run_ctx.write_config()
    run_ctx.write_meta()

    # seed & device
    seed_everything(cfg.seed)
    device = get_device(args.device or cfg.device)
    print("DEVICE:", device)

    # data & model
    train_loader, val_loader = get_dataloaders(
        root=cfg.data.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        # fall back to True if your DataCfg doesn’t define `augment`
        augment=getattr(cfg.data, "augment", True),
    )
    model = resnet20().to(device)

    # optim & sched
    optimizer = build_optimizer(model, cfg.optim)
    sched_handle = build_baseline_scheduler(
        optimizer, cfg.sched, cfg.optim,
        steps_per_epoch=len(train_loader), epochs=cfg.epochs
    )

    # loss & metrics
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Hooks: logging and checkpoints
    hooks = []
    hooks.append(make_train_metrics_hook(io_ctx, cfg.log.log_interval))
    hooks.append(make_val_metrics_hook(io_ctx))

    ckpt_io = CheckpointIO(run_ctx.run_dir / "checkpoints")
    # hook that calls ckpt_io on eval end
    hooks.append(ckpt_io.make_checkpoint_hook())

    def _after_backward(state):
        # gradient clipping
        max_norm = cfg.grad_clip_norm
        if max_norm is None:
            pre = _grad_l2_norm(state["model"])
            state["grad_norm_pre_clip"] = pre
            state["grad_norm_post_clip"] = pre
        else:
            # returns pre-clip norm and clips in-place
            pre = float(torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm))
            state["grad_norm_pre_clip"] = pre
            state["grad_norm_post_clip"] = min(pre, max_norm)
        # if max_norm is None:
        #     state["clip"] = False
        #     return
        # total_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=float(max_norm))
        # try:
        #     state["clip"] = float(total_norm) > float(max_norm)
        # except Exception:
        #     state["clip"] = bool(total_norm > max_norm)

    hooks.append(LambdaHook(on_after_backward=_after_backward))

    # tb_hook = make_tb_hook_optional(cfg.log.dir_tb, run_ctx) # TODO if we want tensorboard
    # if tb_hook:
    #     hooks.append(tb_hook)

    hook_list = HookList(hooks)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=sched_handle.scheduler,
        scheduler_step_when=sched_handle.step_when,
        hooks=hook_list,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        mixed_precision=False,
        metric_fn=None,
        cfg=cfg.to_dict(),
    )
    status = "unknown"
    try:
        trainer.fit()
        status = "success"
    except Exception as e:
        status = "error"
        run_ctx.log_exception(e)
        raise
    finally:
        hook_list.close()
        close_writers(io_ctx.writers)
        run_ctx.finalize(status=status)

    # console summary
    best = getattr(trainer, "best_metric", None)
    print(f"\nRun dir: {run_ctx.run_dir}")
    print(f"Best metric: {best}")
    print(f"Train logs: {io_ctx.paths.get('train_parquet')}")
    print(f"Val logs:   {io_ctx.paths.get('val_parquet')}")
    print(f"Checkpoints: {run_ctx.run_dir / 'checkpoints'}")

if __name__ == "__main__":
    main()
