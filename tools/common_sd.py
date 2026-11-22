"""Stable Diffusion 1.5 + LoRA pieces shared by the training and drawing scripts (CPU)."""
import os

import torch

from paths import DATA, MODELS, ROCO

SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"      # U-Net, text encoder, tokenizer, scheduler
VAE = "stabilityai/sd-vae-ft-mse"                       # SD's improved decoder; same latent space
SPLIT = DATA / "roco_mini_v1.csv"
RUNS = MODELS
LATENTS = ROCO / "latents"
SCALE = 0.18215                                         # SD 1.x latent scaling factor
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]     # every attention projection in the U-Net


def threads():
    torch.set_num_threads(min(8, os.cpu_count() or 8))


def lora_config(rank):
    from peft import LoraConfig
    return LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=LORA_TARGETS)


def load_unet():
    from diffusers import UNet2DConditionModel
    return UNet2DConditionModel.from_pretrained(SD, subfolder="unet")


def load_text():
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(SD, subfolder="tokenizer")
    enc = CLIPTextModel.from_pretrained(SD, subfolder="text_encoder").eval().requires_grad_(False)
    return tok, enc


def load_vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(VAE).eval().requires_grad_(False)


@torch.no_grad()
def encode_text(tok, enc, texts):
    ids = tok(texts, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
    return enc(ids)[0]
