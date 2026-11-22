"""Shrink every ROCO-mini image into a Stable Diffusion latent once, so all runs reuse the same numbers.

Saves data/rocov2/latents/<res>/<ID>.npy: the VAE's mean and spread (8 x res/8 x res/8, float16).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from common_sd import LATENTS, SPLIT, load_vae, threads  # noqa: E402
import rocolib as rl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--roles", default="gen_train")
ap.add_argument("--max-tier", type=float, default=1e9)
args = ap.parse_args()
threads()

df = pd.read_csv(SPLIT)
df = df[df.role.isin(args.roles.split(",")) & ((df.role != "gen_train") | (df.tier <= args.max_tier))]
out = LATENTS / str(args.res)
out.mkdir(parents=True, exist_ok=True)
todo = [i for i in df.ID if not (out / f"{i}.npy").exists()]
print(f"{len(df)} images, {len(todo)} still to encode at {args.res} px", flush=True)
vae = load_vae()
t = time.time()
for k in range(0, len(todo), 8):
    ids = todo[k:k + 8]
    x = np.stack([np.asarray(rl.to_square(rl.open_image(i), args.res), np.float32) / 127.5 - 1 for i in ids])
    with torch.no_grad():
        d = vae.encode(torch.from_numpy(x).permute(0, 3, 1, 2)).latent_dist
    for i, m, s in zip(ids, d.mean, d.std):
        np.save(out / f"{i}.npy", torch.cat([m, s]).numpy().astype(np.float16))
    if k % 200 == 0:
        print(f"  {k + len(ids)}/{len(todo)}  {time.time() - t:.0f} s", flush=True)
print(f"done in {time.time() - t:.0f} s", flush=True)
