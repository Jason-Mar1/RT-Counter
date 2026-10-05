"""Measures Params, GFLOPs and FPS on 384 x 384 inputs (Table 4 protocol).

As in the paper, text embeddings are pre-computed, so the text encoder is not
part of the timed forward pass or of the parameter count.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter import RTCounter, RTCounterConfig  # noqa: E402
from rtcounter.engine import load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=None, help="optional; a randomly initialised model is timed otherwise")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--half", action="store_true", help="time in FP16")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)[0] if args.ckpt else RTCounter(RTCounterConfig()).to(device).eval()
    images = torch.rand(args.batch_size, 3, args.size, args.size, device=device)
    text = torch.nn.functional.normalize(torch.randn(args.batch_size, 1, model.cfg.dim, device=device), dim=-1)
    if args.half:
        model, images, text = model.half(), images.half(), text.half()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params:.2f}M")
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(images[:1], None, text[:1]), verbose=False)
        print(f"GFLOPs: {2 * macs / 1e9:.2f}  (GMACs {macs / 1e9:.2f})")
    except ImportError:
        print("GFLOPs: install `thop` (ultralytics-thop) to measure")

    with torch.no_grad():
        for _ in range(args.warmup):
            model(images, text_embeddings=text)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            model(images, text_embeddings=text)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"FPS: {args.iters * args.batch_size / elapsed:.2f}  ({device}, batch {args.batch_size}, "
          f"{'fp16' if args.half else 'fp32'})")


if __name__ == "__main__":
    main()
