"""Counts the objects described by a text prompt in one image and draws the predicted points."""
import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter.engine import load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True, help="e.g. 'the apples'")
    p.add_argument("--text_weights", default="weights/mobileclip_blt.pt")
    p.add_argument("--max_size", type=int, default=1024, help="longest side after resizing")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--output", default="demo_result.jpg")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.ckpt, device, text_weights=args.text_weights)

    image = Image.open(args.image).convert("RGB")
    W, H = image.size
    scale = min(1.0, args.max_size / max(W, H))
    w, h = max(32, int(W * scale) // 32 * 32), max(32, int(H * scale) // 32 * 32)
    resized = image.resize((w, h), Image.BILINEAR)
    counts, points = model.count(TF.to_tensor(resized)[None].to(device), [args.prompt], threshold=args.threshold)
    count = int(counts[0])
    print(f"{args.prompt}: {count}")

    draw = ImageDraw.Draw(image)
    r = max(2, int(max(W, H) / 200))
    for x, y in (points[0].cpu() * torch.tensor([W / w, H / h])).tolist():
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0), outline=(255, 255, 255))
    draw.text((10, 10), f"{args.prompt}: {count}", fill=(255, 255, 0))
    image.save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
