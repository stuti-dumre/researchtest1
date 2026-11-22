"""T06: time Stable Diffusion 1.x sized parts on this CPU, at 512 and 256 px.

The U-Net, VAE and CLIP text encoder are built from their configs with random weights, since speed depends only on
the shapes. One generation step runs the U-Net on a batch of 2 (with and without text).
"""
import gc
import time
import warnings

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel

from common import save

warnings.filterwarnings("ignore")
torch.manual_seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))
res = {"torch_threads": torch.get_num_threads()}

unet_cfg = dict(sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
                block_out_channels=(320, 640, 1280, 1280),
                down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
                up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
                cross_attention_dim=768, attention_head_dim=8)
vae_cfg = dict(in_channels=3, out_channels=3, latent_channels=4, layers_per_block=2, sample_size=512,
               block_out_channels=(128, 256, 512, 512),
               down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4)
text_cfg = CLIPTextConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                          num_attention_heads=12, max_position_embeddings=77, hidden_act="quick_gelu")

unet = UNet2DConditionModel(**unet_cfg)
vae = AutoencoderKL(**vae_cfg).eval()
text = CLIPTextModel(text_cfg).eval()
res["unet_params_millions"] = round(sum(p.numel() for p in unet.parameters()) / 1e6, 1)
res["vae_params_millions"] = round(sum(p.numel() for p in vae.parameters()) / 1e6, 1)
res["text_params_millions"] = round(sum(p.numel() for p in text.parameters()) / 1e6, 1)
print(res)


def timed(fn, reps):
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


ids = torch.randint(0, 49408, (2, 77))
with torch.no_grad():
    res["text_encode_s"] = round(timed(lambda: text(ids), 3), 3)
    cond = text(ids)[0]

for px in (512, 256):
    lat = px // 8
    r = {}
    with torch.no_grad():
        img = torch.randn(1, 3, px, px)
        r["vae_encode_s"] = round(timed(lambda: vae.encode(img).latent_dist.sample(), 1), 2)
        z = torch.randn(1, 4, lat, lat)
        r["vae_decode_s"] = round(timed(lambda: vae.decode(z).sample, 1), 2)
        unet.eval()
        r["generate_step_s"] = round(timed(lambda: unet(torch.randn(2, 4, lat, lat), 500, cond).sample, 2), 2)
    # one training step for one image, with gradient checkpointing as in train.sh
    unet.train()
    unet.enable_gradient_checkpointing()

    def train_step():
        unet.zero_grad(set_to_none=True)
        out = unet(torch.randn(1, 4, lat, lat), torch.tensor([500]), cond[:1]).sample
        out.float().pow(2).mean().backward()

    r["train_step_1_image_s"] = round(timed(train_step, 2), 2)
    unet.disable_gradient_checkpointing()
    unet.zero_grad(set_to_none=True)
    gc.collect()
    r["one_image_100_steps_min"] = round((r["generate_step_s"] * 100 + r["vae_decode_s"]) / 60, 1)
    per_image_train = r["train_step_1_image_s"] + r["vae_encode_s"]
    r["MINIM_training_20k_steps_x_128_images_years"] = round(per_image_train * 128 * 20000 / 3600 / 24 / 365, 1)
    r["one_epoch_60_examples_min"] = round(per_image_train * 60 / 60, 1)
    res[f"{px}px"] = r
    print(px, r)

save("t06_laptop_speed", res)
