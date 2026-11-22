"""T03: compare MINIM's metrics.py (FID, Inception Score, SSIM) with the standard formulas.

Their functions are imported unchanged. The image sets are random crops of the 60 example images: real, a second
real set, and blurred noisy copies that stand in for generated images. Features come from the standard FID
Inception network (pytorch-fid); metrics.py itself loads torchvision's ImageNet Inception.
Takes about 15 minutes on a CPU.
"""
import importlib.util
import random
import warnings

import numpy as np
import torch
from PIL import Image, ImageFilter
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import fid_inception_v3

from common import BASE, EXAMPLE, read_example_csv, save

warnings.filterwarnings("ignore")
torch.manual_seed(0)
spec = importlib.util.spec_from_file_location("minim_metrics", BASE / "metrics.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)
res = {}

# main() calls read_images(folder) without a transform
try:
    mm.read_images(str(EXAMPLE / "oct"))
    res["read_images_default"] = "ok"
except Exception as e:
    res["read_images_default"] = f"CRASH: {type(e).__name__}: {e}"
print("read_images:", res["read_images_default"])

# the paper computes FID "for each synthetic-real image pair": try it with one image per set
one = np.random.rand(1, 8).astype(np.float32)
fake_net = lambda x: torch.tensor(x)  # their functions call net(x); this passes precomputed logits through
with np.errstate(all="ignore"):
    try:
        cov = np.cov(one.T)
        res["fid_one_pair"] = f"covariance of one image is all NaN: {bool(np.isnan(cov).all())}; FID = {mm.calculate_FID(fake_net, one, one + 0.1)}"
    except Exception as e:
        res["fid_one_pair"] = f"CRASH: {type(e).__name__}: {e}"
print("FID on one pair:", res["fid_one_pair"])

net = fid_inception_v3().eval()
pool = {}
net.avgpool.register_forward_hook(lambda m, i, o: pool.__setitem__("x", o.flatten(1)))

_, rows = read_example_csv()
imgs = []
for r in rows:
    im = Image.open(EXAMPLE / r["path"]).convert("RGB")
    im.thumbnail((1024, 1024))  # speed only
    imgs.append(im)
print("images loaded", flush=True)
rng = random.Random(0)


def crops(images, n_each, degrade=False):
    out = []
    for im in images:
        base = im.filter(ImageFilter.GaussianBlur(3)) if degrade else im
        w, h = base.size
        for _ in range(n_each):
            s = rng.uniform(0.5, 1.0)
            cw, ch = int(w * s), int(h * s)
            x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
            c = base.crop((x, y, x + cw, y + ch)).resize((299, 299), Image.BILINEAR)
            a = np.asarray(c, dtype=np.float32) / 255.0
            if degrade:
                a = np.clip(a + np.random.default_rng(len(out)).normal(0, 0.03, a.shape), 0, 1)
            out.append(a.astype(np.float32))
    return np.stack(out)


def feats(arr, bs=50):
    P, L = [], []
    with torch.no_grad():
        for i in range(0, len(arr), bs):
            x = torch.from_numpy(arr[i:i + bs]).permute(0, 3, 1, 2) * 2 - 1
            L.append(net(x).numpy())
            P.append(pool["x"].numpy())
    return np.concatenate(P), np.concatenate(L)


N_EACH = 42  # 60 x 42 = 2,520 crops per set, more than the 2,048 features
sets = {"real": crops(imgs, N_EACH), "real2": crops(imgs, N_EACH), "fake": crops(imgs, N_EACH, degrade=True)}
print("crops built", {k: v.shape for k, v in sets.items()}, flush=True)
F = {}
for k, v in sets.items():
    F[k] = feats(v)
    print("features done:", k, flush=True)
res["n_per_set"] = len(sets["real"])


def frechet(a, b):
    return float(calculate_frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False)))


def minim_fid(la, lb):
    with np.errstate(all="ignore"):
        return float(mm.calculate_FID(fake_net, la, lb))


for a, b in [("real", "real2"), ("real", "fake")]:
    key = f"{a}_vs_{b}"
    res[key] = {
        "standard_FID_pool3_squared": round(frechet(F[a][0], F[b][0]), 2),
        "MINIM_formula_logits_sqrt": round(minim_fid(F[a][1], F[b][1]), 2),
        "logits_squared": round(frechet(F[a][1], F[b][1]), 2),
        "pool3_sqrt": round(frechet(F[a][0], F[b][0]) ** 0.5, 2),
    }
    print(key, res[key])


def is_standard(logits, splits=10):
    p = torch.softmax(torch.tensor(logits), 1).numpy()
    scores = []
    for part in np.array_split(p, splits):
        py = part.mean(0, keepdims=True)
        kl = (part * (np.log(part + 1e-12) - np.log(py + 1e-12))).sum(1).mean()
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


for k in ["real", "fake"]:
    with np.errstate(all="ignore"):
        theirs = float(mm.calculate_IS(fake_net, F[k][1]))
    std_mean, std_sd = is_standard(F[k][1])
    res[f"IS_{k}"] = {"MINIM_formula": round(theirs, 4), "standard_mean": round(std_mean, 3),
                      "standard_sd": round(std_sd, 3)}
    print("IS", k, res[f"IS_{k}"])
# 10 confident, all-different predictions: the true IS is 10
toy = np.full((10, 10), -20.0, dtype=np.float32)
np.fill_diagonal(toy, 20.0)
with np.errstate(all="ignore"):
    res["IS_toy_true_value_10"] = {"MINIM_formula": float(mm.calculate_IS(fake_net, toy)),
                                   "standard": round(is_standard(toy, splits=1)[0], 3)}
print("IS toy:", res["IS_toy_true_value_10"])

# SSIM between real and fake, in matched and shuffled file order
small = lambda a: torch.from_numpy(a[:60]).permute(0, 3, 1, 2)
r_t, f_t = small(sets["real"][::N_EACH]), small(sets["fake"][::N_EACH])
matched = float(mm.calculate_SSIM(r_t, f_t))
perm = torch.randperm(len(f_t))
shuffled = float(mm.calculate_SSIM(r_t, f_t[perm]))
res["SSIM_same_images_matched_order"] = round(matched, 4)
res["SSIM_same_images_shuffled_order"] = round(shuffled, 4)
print("SSIM matched vs shuffled:", matched, shuffled)
save("t03_metrics", res)
