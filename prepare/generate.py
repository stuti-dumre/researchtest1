"""Generate one image per test item for a trained model (or for the base model with --run base).

Each test item uses the same random seed in every model, so differences come from the model, not luck.
Writes models/<run>/gen/<ID>.png and models/<run>/prompts.csv.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from common_sd import RUNS, SD, SPLIT, load_unet, load_vae, lora_config, threads  # noqa: E402
from texts import make_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="a folder in models/, or 'base' for Stable Diffusion without fine-tuning")
ap.add_argument("--arm", default=None, help="caption format for prompts (default: the run's; base uses minim_paper)")
ap.add_argument("--n-per-mod", type=int, default=50)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--cfg", type=float, default=4.0, help="MINIM uses guidance 4.0")
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--ids", default="", help="CSV with an ID column: draw only these test items (instead of --n-per-mod)")
ap.add_argument("--shard", default="0/1", help="i/n: draw only the i-th of n equal parts of the items")
ap.add_argument("--seed-base", type=int, default=1000, help="item seed = this + item number")
ap.add_argument("--negative", default="", help="negative text when drawing (default: empty)")
ap.add_argument("--out", default="", help="folder in models/ for the images (default: the --run folder)")
args = ap.parse_args()
threads()

run = RUNS / args.run
run.mkdir(parents=True, exist_ok=True)
arm = args.arm or (json.loads((run / "config.json").read_text())["arm"] if args.run != "base" else "minim_paper")
out_run = RUNS / (args.out or args.run)
out = out_run / "gen"
out.mkdir(parents=True, exist_ok=True)

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
unet = load_unet()
if args.run != "base":
    from peft.utils import set_peft_model_state_dict
    cfg = json.loads((run / "config.json").read_text())
    unet.add_adapter(lora_config(cfg["rank"]))
    set_peft_model_state_dict(unet, torch.load(run / "lora.pt"))
pipe = StableDiffusionPipeline.from_pretrained(SD, unet=unet, vae=load_vae(), safety_checker=None,
                                               requires_safety_checker=False)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

df = pd.read_csv(SPLIT)
test = df[df.role == "test"]
if args.ids:
    test = test[test.ID.isin(set(pd.read_csv(args.ids).ID))]
else:
    test = test.groupby("modality", group_keys=False).head(args.n_per_mod)
part, parts = map(int, args.shard.split("/"))
test = test.reset_index(drop=True)
test = test.iloc[part * len(test) // parts:(part + 1) * len(test) // parts].reset_index(drop=True)
prompts = []
t0 = time.time()
for k, row in test.iterrows():
    path = out / f"{row.ID}.png"
    text = make_text(row.to_dict(), arm, for_eval=True)
    prompts.append({"ID": row.ID, "modality": row.modality, "region": row.region, "prompt": text})
    if path.exists():
        continue
    seed = args.seed_base + int(row.ID.split("_")[-1])
    img = pipe(text, negative_prompt=args.negative or None, num_inference_steps=args.steps, guidance_scale=args.cfg,
               height=args.res, width=args.res, generator=torch.Generator().manual_seed(seed)).images[0]
    img.save(path)
    if k % 10 == 0:
        print(f"{k + 1}/{len(test)}  {time.time() - t0:.0f} s", flush=True)
pd.DataFrame(prompts).to_csv(out_run / ("prompts.csv" if parts == 1 else f"prompts_{part}.csv"), index=False)
print(f"done: {len(test)} images in {time.time() - t0:.0f} s -> {out}", flush=True)
