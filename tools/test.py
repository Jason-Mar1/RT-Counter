"""Evaluates an RT-Counter checkpoint (MAE / RMSE)."""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter.data import build_dataset, collate_fn  # noqa: E402
from rtcounter.engine import evaluate, load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="fsc147", choices=["fsc147", "carpk", "rec8k"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--text_weights", default="weights/mobileclip_blt.pt")
    p.add_argument("--threshold", type=float, default=None, help="phi; defaults to the checkpoint value (0.5)")
    p.add_argument("--save_csv", default=None, help="optional per-image predictions")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.ckpt, device, text_weights=args.text_weights)
    rows = []
    for split in args.splits:
        loader = DataLoader(build_dataset(args.dataset, args.data_root, split), 1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
        mae, rmse, records = evaluate(model, loader, device, threshold=args.threshold)
        print(f"{args.dataset} {split}: MAE {mae:.2f}  RMSE {rmse:.2f}  ({len(records)} images)")
        rows += [{"split": split, **r} for r in records]
    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "id", "prompt", "pred", "gt"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
