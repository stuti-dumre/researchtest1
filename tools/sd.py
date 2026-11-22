"""Stable Diffusion 1.5 + LoRA helpers for test 4. Same model pieces (common_sd.py) and training loop as
prepare/train_lora.py, with two additions: training from an item list (any image IDs or generated PNGs, with their own
text) and starting from an existing LoRA.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common_sd import SCALE, SD, encode_text, load_text, load_unet, load_vae, lora_config, threads
import rocolib as rl


def encode_latents(images, out_paths, vae=None):
    """PIL images (any size) -> VAE mean and spread (8 x 32 x 32, float16), saved like prep_latents.py."""
    vae = vae or load_vae()
    for k in range(0, len(images), 8):
        x = np.stack([np.asarray(rl.to_square(im, 256), np.float32) / 127.5 - 1 for im in images[k:k + 8]])
        with torch.no_grad():
            d = vae.encode(torch.from_numpy(x).permute(0, 3, 1, 2)).latent_dist
        for p, m, s in zip(out_paths[k:k + 8], d.mean, d.std):
            np.save(p, torch.cat([m, s]).numpy().astype(np.float16))
    return vae


def train(items, lat_paths, steps, out, init_lora=None, seed=0, batch=4, lr=1e-4, rank=16, drop=0.1, save_every=100,
          log=print):
    """items: DataFrame with a 'text' column; lat_paths: one latent file per row. Same loop as train_lora.py; saves
    state.pt every save_every steps and resumes from it exactly."""
    threads()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "lora.pt").exists():
        log(f"  {out.name}: already trained")
        return
    torch.manual_seed(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    texts_all = list(items.text)
    lat = np.stack([np.load(p) for p in lat_paths]).astype(np.float32)
    from diffusers import DDPMScheduler
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    noise_sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler")
    tok, enc = load_text()
    unet = load_unet().requires_grad_(False)
    unet.add_adapter(lora_config(rank))
    if init_lora:
        set_peft_model_state_dict(unet, torch.load(init_lora))
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    unet.train()
    order, pos, hist, first, t0 = np_rng.permutation(len(texts_all)), 0, [], 1, time.time()
    state = out / "state.pt"
    if state.exists():
        sv = torch.load(state, weights_only=False)
        set_peft_model_state_dict(unet, sv["lora"])
        opt.load_state_dict(sv["opt"])
        rng.setstate(sv["rng"])
        np_rng.bit_generator.state = sv["np_rng"]
        torch.set_rng_state(sv["torch_rng"])
        order, pos, hist, first = sv["order"], sv["pos"], sv["log"], sv["step"] + 1
        log(f"  {out.name}: resumed at step {first}")
    for step in range(first, steps + 1):
        if pos + batch > len(order):
            order, pos = np_rng.permutation(len(texts_all)), 0
        idx = order[pos:pos + batch]
        pos += batch
        texts = [("" if rng.random() < drop else texts_all[i]) for i in idx]
        cond = encode_text(tok, enc, texts)
        m, s = torch.from_numpy(lat[idx, :4]), torch.from_numpy(lat[idx, 4:])
        x0 = (m + s * torch.randn_like(s)) * SCALE
        noise = torch.randn_like(x0)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (len(idx),))
        pred = unet(noise_sched.add_noise(x0, noise, t), t, cond).sample
        loss = F.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append({"step": step, "loss": float(loss.detach()), "sec": round(time.time() - t0, 1)})
        if step % 50 == 0 or step == 1:
            log(f"  {out.name}: step {step}/{steps}  loss {np.mean([h['loss'] for h in hist[-50:]]):.4f}  "
                f"{time.time() - t0:.0f} s")
        if step % save_every == 0 and step < steps:
            torch.save({"lora": get_peft_model_state_dict(unet), "opt": opt.state_dict(), "rng": rng.getstate(),
                        "np_rng": np_rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "order": order,
                        "pos": pos, "log": hist, "step": step}, state)
    torch.save(get_peft_model_state_dict(unet), out / "lora.pt")
    state.unlink(missing_ok=True)
    pd.DataFrame(hist).to_csv(out / "loss.csv", index=False)
    (out / "config.json").write_text(json.dumps({"steps": steps, "batch": batch, "lr": lr, "rank": rank, "drop": drop,
                                                 "seed": seed, "init_lora": str(init_lora or ""), "images": len(items),
                                                 "seconds": round(time.time() - t0)}, indent=2))
    del unet, opt
    log(f"  {out.name}: saved in {time.time() - t0:.0f} s")


def pipeline(lora_path, rank=16):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from peft.utils import set_peft_model_state_dict
    threads()
    unet = load_unet()
    unet.add_adapter(lora_config(rank))
    set_peft_model_state_dict(unet, torch.load(lora_path))
    pipe = StableDiffusionPipeline.from_pretrained(SD, unet=unet, vae=load_vae(), safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def draw(pipe, jobs, steps=15, cfg=4.0, res=256, log=print, label=""):
    """jobs: list of (text, seed, out_path). Skips images that already exist (resumable)."""
    t0, todo = time.time(), [j for j in jobs if not Path(j[2]).exists()]
    for k, (text, seed, path) in enumerate(todo):
        img = pipe(text, num_inference_steps=steps, guidance_scale=cfg, height=res, width=res,
                   generator=torch.Generator().manual_seed(int(seed))).images[0]
        img.save(path)
        if k % 25 == 0 or k == len(todo) - 1:
            log(f"  {label}: {k + 1}/{len(todo)} drawn ({len(jobs) - len(todo)} already there)  {time.time() - t0:.0f} s")
