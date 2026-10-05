"""Pre-computes MobileCLIP text embeddings (E_t') for a list of prompts and saves them as .pt.

At inference, ``RTCounter.text_encoder.load_cache(path)`` makes the model skip text
encoding for these prompts (supplementary material, "Model Details").
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter.backbone import TextEncoder  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", nargs="*", default=[], help="prompts, e.g. 'the apples'")
    p.add_argument("--prompt_file", default=None, help="text file with one prompt per line")
    p.add_argument("--text_weights", default="weights/mobileclip_blt.pt")
    p.add_argument("--output", default="text_embeddings.pt")
    args = p.parse_args()

    prompts = list(args.prompts)
    if args.prompt_file:
        prompts += [line.strip() for line in open(args.prompt_file, encoding="utf-8") if line.strip()]
    if not prompts:
        p.error("no prompts given")
    encoder = TextEncoder(weights=args.text_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder(prompts, device)
    encoder.save_cache(args.output)
    print(f"saved {len(encoder.cache)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
