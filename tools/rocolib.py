"""ROCOv2 helpers: images read straight from the official zip files, padding to a square, image features and
distances. A small cache folder (data/rocov2/images/) keeps images after the first read.
"""
import io
import zipfile
from functools import lru_cache

import numpy as np
from PIL import Image

from paths import ROCO

ZIPS = ROCO / "zips"
IMG_CACHE = ROCO / "images"
PARTS = ("train", "valid", "test")


def part_of(image_id):
    """'ROCOv2_2023_valid_000123' -> 'valid'"""
    return image_id.split("_")[2]


@lru_cache(maxsize=None)
def _zip(part):
    return zipfile.ZipFile(ZIPS / f"{part}_images.zip")


def read_bytes(image_id):
    cached = IMG_CACHE / f"{image_id}.jpg"
    if cached.exists():
        return cached.read_bytes()
    part = part_of(image_id)
    return _zip(part).read(f"{part}/{image_id}.jpg")


def open_image(image_id):
    return Image.open(io.BytesIO(read_bytes(image_id)))


def cache_images(ids):
    """Copy the given images out of the zips into data/rocov2/images/."""
    IMG_CACHE.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in ids:
        p = IMG_CACHE / f"{i}.jpg"
        if not p.exists():
            p.write_bytes(read_bytes(i))
            n += 1
    return n


def to_square(img, size=256, fill=0):
    """RGB, keep the whole image (no cropping): shrink the long side to `size`, pad the short side with black."""
    img = img.convert("RGB")
    w, h = img.size
    s = size / max(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    out = Image.new("RGB", (size, size), (fill, fill, fill))
    out.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return out


# ------------------------------------------------------------------ features and distances
_NETS = {}


def dino():
    if "dino" not in _NETS:
        import torch
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True, verbose=False)
        _NETS["dino"] = net.eval().requires_grad_(False)
    return _NETS["dino"]


def swav():
    """SwAV ResNet-50 (self-supervised on ImageNet): the backbone of judge 2, independent of DINOv2."""
    if "swav" not in _NETS:
        import torch
        net = torch.hub.load("facebookresearch/swav:main", "resnet50", trust_repo=True, verbose=False)
        net.fc = torch.nn.Identity()
        _NETS["swav"] = net.eval().requires_grad_(False)
    return _NETS["swav"]


def swav_features(images, batch=32):
    """List of PIL images -> (n, 2048) SwAV ResNet-50 pooled features (same preprocessing as dino_features)."""
    return dino_features(images, batch, net_fn=swav)


def dino_features(images, batch=32, net_fn=None):
    """List of PIL images -> (n, 768) DINOv2 ViT-B/14 embeddings (ImageNet normalisation, 224 px)."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    net = (net_fn or dino)()
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = np.stack([np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC), np.float32) / 255
                          for im in images[i:i + batch]])
            x = (torch.from_numpy(x).permute(0, 3, 1, 2) - mean) / std
            out.append(net(x).double().numpy())
    return np.concatenate(out)


def cosine_nn(a, b, exclude_self=False):
    """For each row of a: highest cosine similarity to any row of b (and its index)."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    sims = a @ b.T
    if exclude_self:
        np.fill_diagonal(sims, -1)
    j = sims.argmax(1)
    return sims[np.arange(len(a)), j], j


def frechet_distance(f1, f2):
    from scipy import linalg
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        eps = np.eye(len(s1)) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean.real))


def kid(f1, f2, n_subsets=100, subset_size=1000, seed=0):
    """Unbiased KID (MMD^2, cubic polynomial kernel), mean and std over random subsets. Works with small sets."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(f1), len(f2))
    d = f1.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))
