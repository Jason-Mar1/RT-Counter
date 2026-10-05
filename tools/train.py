"""Trains RT-Counter."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter import RTCounter, RTCounterConfig, RTCounterLoss  # noqa: E402
from rtcounter.engine import evaluate, load_model, param_groups, save_checkpoint  # noqa: E402


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="fsc147", choices=["fsc147", "carpk", "rec8k"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="runs/rtcounter")
    p.add_argument("--yoloe_weights", default="weights/yoloe-11s-seg.pt")
    p.add_argument("--text_weights", default="weights/mobileclip_blt.pt")
    p.add_argument("--resume", default=None, help="last.pt of an interrupted run")
    # Optimisation.
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lr_drop", type=int, default=100, help="StepLR step size in epochs")
    p.add_argument("--val_freq", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    # Loss (lambda_1, lambda_2, sigma).
    p.add_argument("--lambda_reg", type=float, default=0.25)
    p.add_argument("--lambda_cls", type=float, default=5.0)
    p.add_argument("--sigma", type=float, default=0.5)
    # Model.
    p.add_argument("--num_prototypes", type=int, default=16)
    p.add_argument("--enhancer_depth", type=int, default=2)
    p.add_argument(
        "--enhancer_type",
        choices=["reference_fim", "v2_fim", "historical_fim", "weaformer"],
        default="reference_fim",
        help="feature enhancer; reference_fim is the default",
    )
    p.add_argument(
        "--vpt_type",
        choices=["reference", "v2", "legacy"],
        default="reference",
        help="prototype context path; reference is the default",
    )
    p.add_argument("--fim_downsample_ratio", type=int, default=None,
                   help="historical FIM downsampling for each spatial side (default: 2)")
    p.add_argument("--downsample_rate", type=int, default=None,
                   help="Weaformer-only visual-token downsampling rate")
    p.add_argument("--global_ratio", type=float, default=None,
                   help="Weaformer-only global-channel ratio")
    p.add_argument("--no_vpt", action="store_true", help="ablation: the enhancer attends to E_t only")
    p.add_argument("--freeze_backbone", action="store_true")
    args = p.parse_args(argv)
    if args.enhancer_type == "historical_fim":
        if args.downsample_rate is not None or args.global_ratio is not None:
            p.error(
                "--downsample_rate and --global_ratio are only valid with "
                "--enhancer_type weaformer; historical_fim fixes the global/local split at 1:1."
            )
        if args.fim_downsample_ratio is None:
            args.fim_downsample_ratio = 2
        elif args.fim_downsample_ratio < 2:
            p.error("--fim_downsample_ratio must be an integer >= 2 for --enhancer_type historical_fim.")
    else:
        if args.fim_downsample_ratio is not None:
            p.error("--fim_downsample_ratio is only valid with --enhancer_type historical_fim.")
        if args.enhancer_type in {"reference_fim", "v2_fim"} and (
            args.downsample_rate is not None or args.global_ratio is not None
        ):
            p.error("--downsample_rate and --global_ratio are only valid with --enhancer_type weaformer.")
    required_vpt_type = {
        "reference_fim": "reference",
        "v2_fim": "v2",
    }.get(args.enhancer_type)
    if not args.no_vpt and required_vpt_type is not None and args.vpt_type != required_vpt_type:
        p.error(
            f"--enhancer_type {args.enhancer_type} requires --vpt_type {required_vpt_type} when VPT is enabled."
        )
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = get_args()
    from rtcounter.data import build_dataset, collate_fn

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_set = build_dataset(args.dataset, args.data_root, "train")
    val_set = build_dataset(args.dataset, args.data_root, "test" if args.dataset == "carpk" else "val")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, 1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    if args.resume:
        model, ckpt = load_model(args.resume, device, text_weights=args.text_weights)
    else:
        cfg_kwargs = dict(
            yoloe_weights=args.yoloe_weights, text_weights=args.text_weights,
            num_prototypes=args.num_prototypes, enhancer_depth=args.enhancer_depth,
            enhancer_type=args.enhancer_type, vpt_type=args.vpt_type,
            use_vpt=not args.no_vpt, freeze_backbone=args.freeze_backbone,
        )
        if args.enhancer_type == "historical_fim":
            cfg_kwargs["fim_downsample_ratio"] = args.fim_downsample_ratio
        elif args.enhancer_type == "weaformer":
            if args.downsample_rate is not None:
                cfg_kwargs["downsample_rate"] = args.downsample_rate
            if args.global_ratio is not None:
                cfg_kwargs["global_ratio"] = args.global_ratio
        cfg = RTCounterConfig(**cfg_kwargs)
        model, ckpt = RTCounter(cfg).to(device), None
    criterion = RTCounterLoss(args.lambda_reg, args.lambda_cls, sigma=args.sigma)
    optimizer = torch.optim.AdamW(param_groups(model, args.lr, args.backbone_lr), betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.33)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    start_epoch, best_mae = 0, float("inf")
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_mae = ckpt["epoch"] + 1, ckpt["best_mae"]

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"RT-Counter: {n_params:.2f}M parameters (text encoder excluded), training on {device}")

    log = open(out_dir / "log.txt", "a")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), {}
        for it, (images, points, prompts, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(images, prompts)
            losses = criterion(outputs, points)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v)
            if it % 50 == 0:
                print(f"epoch {epoch} iter {it}/{len(train_loader)} "
                      + " ".join(f"{k} {float(v):.4f}" for k, v in losses.items()), flush=True)
        scheduler.step()
        stats = {k: v / len(train_loader) for k, v in running.items()}
        line = {"epoch": epoch, **stats, "time": round(time.time() - t0, 1)}

        if (epoch + 1) % args.val_freq == 0 or epoch + 1 == args.epochs:
            mae, rmse, _ = evaluate(model, val_loader, device, amp=amp)
            line.update(val_mae=mae, val_rmse=rmse)
            if mae < best_mae:
                best_mae = mae
                save_checkpoint(out_dir / "best.pt", model, epoch=epoch, val_mae=mae, val_rmse=rmse)
            print(f"epoch {epoch}: val MAE {mae:.2f} RMSE {rmse:.2f} (best MAE {best_mae:.2f})", flush=True)
        log.write(json.dumps(line) + "\n")
        log.flush()
        save_checkpoint(out_dir / "last.pt", model, epoch=epoch, best_mae=best_mae,
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        scaler=scaler.state_dict())
    log.close()


if __name__ == "__main__":
    main()
