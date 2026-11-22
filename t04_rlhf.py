"""T04: test the doctor-feedback code in MINIM_RLHF, using its own classes.

Checks: selector, stage-1 policy loss vs the paper's update rule, expert weighting, CT/MRI loading, quality gate,
generator wrapper (tiny test model), policy image encoder.
"""
import copy
import shutil
import types
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import yaml
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from scipy.stats import spearmanr

from common import EXAMPLE, RESULTS, TINY_SD, import_rlhf, read_example_csv, rlhf_config, save

warnings.filterwarnings("ignore")
import_rlhf()
from src.raters.rating_system import RatingSystem
from src.rl.policy import RLPolicy, VisionTransformer
from src.selector.receive_selector import Selector
from src.utils.medical_processor import MedicalImageProcessor

cfg = rlhf_config()
work = RESULTS / "t04_work"
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
res = {}
_, rows = read_example_csv()
proc = MedicalImageProcessor(cfg)

# selector
torch.manual_seed(0)
sel = Selector(cfg).eval()
picks = []
for m in ["OCT", "Fundus", "CXR", "ChestCT", "BrainMRI", "BreastMRI"]:
    picks += [r for r in rows if r["modal"] == m][:2]
batch = torch.stack([torch.tensor(proc.load_medical_image(str(EXAMPLE / r["path"]))).unsqueeze(0).float()
                     for r in picks])
extra = torch.stack([torch.zeros(1, 512, 512), torch.rand(1, 512, 512)])  # black image, noise
allimg = torch.cat([batch, extra])
names = [r["path"] for r in picks] + ["all-black", "random-noise"]
with torch.no_grad():
    one_by_one = [sel(x.unsqueeze(0)).item() for x in allimg]
    as_batch = sel(allimg).squeeze(1).tolist()
# same weights, with the transformer told that dim 0 is the batch
fixed = copy.deepcopy(sel)
for layer in fixed.transformer.layers:
    layer.self_attn.batch_first = True
    layer.batch_first = True
with torch.no_grad():
    fixed_scores = [fixed(x.unsqueeze(0)).item() for x in allimg]


def grad_norm(model, x):
    x = x.clone().unsqueeze(0).requires_grad_()
    model(x).sum().backward()
    return float(x.grad.abs().sum())


res["selector"] = {
    "gradient_of_score_wrt_image": {"released": grad_norm(sel, allimg[0]), "with_fix": grad_norm(fixed, allimg[0])},
    "images": names,
    "scores_one_by_one": [round(s, 6) for s in one_by_one],
    "scores_as_batch": [round(s, 6) for s in as_batch],
    "spread_one_by_one": float(np.ptp(one_by_one)),
    "spread_as_batch": float(np.ptp(as_batch)),
    "spread_with_batch_first_fix": float(np.ptp(fixed_scores)),
    "scores_with_fix": [round(s, 6) for s in fixed_scores],
}
print("selector spread (one by one, batch, fixed):", res["selector"]["spread_one_by_one"],
      res["selector"]["spread_as_batch"], res["selector"]["spread_with_batch_first_fix"])

# policy loss: their stage-1 loss against the paper's rule, whose gradient is (R - pi) * grad(pi)
stub = types.SimpleNamespace(config=cfg)
stage1 = lambda s, r: RLPolicy.compute_stage1_loss(stub, s, r)
paper = lambda s, r: (0.5 * ((r - 1) / 2 - s.squeeze()) ** 2).mean()


def fit(loss_fn, setting, steps=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = 300
    ratings = torch.randint(1, 4, (n,), generator=g).float()
    if setting == "free":  # one free score per image
        z = torch.randn(n, 1, generator=g).requires_grad_()
        params, score = [z], lambda: torch.sigmoid(z)
    else:  # a small network that sees a feature tied to the rating
        x = ratings.unsqueeze(1) + 0.3 * torch.randn(n, 1, generator=g)
        net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        torch.manual_seed(seed)
        params, score = list(net.parameters()), lambda: net(x)
    opt = torch.optim.Adam(params, lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(score(), ratings).backward()
        opt.step()
    s = score().detach().squeeze()
    return {"mean_score_by_rating": {int(k): round(float(s[ratings == k].mean()), 3) for k in (1, 2, 3)},
            "target_by_rating": {1: 0.0, 2: 0.5, 3: 1.0},
            "share_scores_above_0.95": round(float((s > 0.95).float().mean()), 3),
            "spearman_with_rating": round(float(spearmanr(s.numpy(), ratings.numpy())[0]), 3)}


res["policy_loss"] = {}
for setting in ["free", "network"]:
    res["policy_loss"][setting] = {"MINIM_code_stage1": fit(stage1, setting), "paper_equation": fit(paper, setting)}
    print("policy", setting, res["policy_loss"][setting])

# expert weighting: every rating at every expert level
grid = pd.DataFrame([{"image_path": f"img_{r}_{e}.png", "rating": r, "modality": "OCT", "expert_level": e,
                      "prompt": "OCT: x"} for e in (1, 2, 3) for r in (1, 2, 3)])
grid.to_csv(work / "ratings_grid.csv", index=False)
rs = RatingSystem(cfg)
df = rs.load_ratings(str(work / "ratings_grid.csv"))
kept_paths, kept_ratings, _ = rs.process_ratings(df)
kept = dict(zip(kept_paths, kept_ratings))
res["expert_weighting"] = [{"doctor_rating": int(r.rating_orig), "expert_level": int(r.expert_level),
                            "after_weighting": round(float(r.rating), 3), "kept_for_training": r.image_path in kept}
                           for r in df.assign(rating_orig=grid.rating).itertuples()]
print("expert weighting kept", sum(x["kept_for_training"] for x in res["expert_weighting"]), "of 9")

# CT phantom in Hounsfield units, saved as DICOM the way scanners store it (stored = HU + 1024)
yy, xx = np.mgrid[:512, :512]
hu = np.full((512, 512), -1000.0)
body = ((xx - 256) / 220) ** 2 + ((yy - 256) / 170) ** 2 < 1
hu[body] = 40
hu[body & ~(((xx - 256) / 200) ** 2 + ((yy - 256) / 150) ** 2 < 1)] = 700
for cx in (170, 342):
    hu[((xx - cx) / 70) ** 2 + ((yy - 250) / 110) ** 2 < 1] = -820
hu[(xx - 150) ** 2 + (yy - 230) ** 2 < 14 ** 2] = 20
stored = (hu + 1024).astype(np.uint16)

meta = FileMetaDataset()
meta.TransferSyntaxUID = ExplicitVRLittleEndian
meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
meta.MediaStorageSOPInstanceUID = generate_uid()
ds = Dataset()
ds.file_meta = meta
ds.Modality = "CT"
ds.Rows, ds.Columns = 512, 512
ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
ds.PixelData = stored.tobytes()
ct_path = work / "phantom_ct.dcm"
ds.save_as(str(ct_path), enforce_file_format=True)
theirs = proc.load_medical_image(str(ct_path))


def window(h, c, w):
    lo, hi = c - w / 2, c + w / 2
    return np.clip((h - lo) / (hi - lo), 0, 1)


res["ct_dicom"] = {
    "their_output_unique_values": int(len(np.unique(theirs))),
    "their_output_nonzero_fraction": round(float((theirs > 0).mean()), 4),
    "correct_soft_tissue_window_unique_values": int(len(np.unique((window(hu, 40, 400) * 255).astype(np.uint8)))),
    "rescale_intercept_used_by_their_code": False,
}
np.save(work / "ct_hu.npy", hu)
np.save(work / "ct_theirs.npy", theirs)
print("CT:", res["ct_dicom"])

# MRI volume: a smooth ramp from 0 to 3000
mri = np.zeros((128, 128, 9), dtype=np.float32)
mri[16:112, 16:112, :] = np.linspace(0, 3000, 96)[None, :, None]
nib.save(nib.Nifti1Image(mri, np.eye(4)), str(work / "ramp.nii.gz"))
out = proc.load_medical_image(str(work / "ramp.nii.gz"))
row = out[256, 64:448]
res["mri_nifti"] = {"input_is_smooth_ramp_0_to_3000": True,
                    "output_direction_changes_along_ramp": int((np.diff(np.sign(np.diff(row))) != 0).sum()),
                    "slices_used": "middle slice only (of 9)"}
np.save(work / "mri_theirs.npy", out)
print("NIfTI:", res["mri_nifti"])

# quality gate on the real example images
q = cfg["selector"]["quality_metrics"]
gate = {}
for r in rows:
    m = proc.compute_quality_metrics(proc.load_medical_image(str(EXAMPLE / r["path"])))
    passed = m["snr"] >= q["snr_threshold"] and m["contrast"] >= q["contrast_threshold"]
    gate.setdefault(r["modal"], []).append((passed, float(m["snr"]), float(m["contrast"])))
res["quality_gate_on_real_images"] = {k: {"passed": int(sum(p for p, _, _ in v)), "of": len(v),
                                          "median_snr": round(float(np.median([s for _, s, _ in v])), 2),
                                          "min_contrast": round(min(c for _, _, c in v), 3)} for k, v in gate.items()}
print("gate:", res["quality_gate_on_real_images"])

# generator wrapper with a tiny model
tiny_st = work / "tiny_safetensors"
StableDiffusionPipeline.from_pretrained(snapshot_download(TINY_SD), safety_checker=None,
                                        requires_safety_checker=False).save_pretrained(str(tiny_st))
from src.synthetic.synthetic_system import SyntheticSystem

c2 = copy.deepcopy(cfg)
c2["model_paths"]["stable_diffusion"] = str(tiny_st)
c2["synthetic_system"]["num_inference_steps"] = 2
res["config_has_selector_threshold"] = "threshold" in cfg["selector"]
try:
    ss = SyntheticSystem(c2)
    res["synthetic_vae_after_init"] = repr(ss.model.vae)
    dummy_sel = types.SimpleNamespace(evaluate_image=lambda im: 1.0)
    dummy_pol = types.SimpleNamespace(evaluate_image=lambda im, p: 1.0)
    ss.generate_images(["OCT: healthy retinal layers"], dummy_sel, dummy_pol)
    res["synthetic_generate"] = "ok"
except Exception as e:
    res["synthetic_generate"] = f"CRASH: {type(e).__name__}: {str(e)[:200]}"
print("SyntheticSystem.generate_images:", res["synthetic_generate"])
c3 = copy.deepcopy(c2)
c3["synthetic_system"]["enable_training"] = True
c3["device"]["precision"] = "float32"
try:
    ss3 = SyntheticSystem(c3)
    ss3.update(torch.rand(1, 1, 64, 64), ["OCT: x"])
    res["synthetic_update"] = "ok"
except Exception as e:
    res["synthetic_update"] = f"CRASH: {type(e).__name__}: {str(e)[:200]}"
print("SyntheticSystem.update:", res["synthetic_update"])
res["yaml_reads_1e-5_as"] = repr(yaml.safe_load("lr: 1e-5")["lr"])
c4 = copy.deepcopy(c3)
c4["synthetic_system"]["training"]["learning_rate"] = 1e-5  # only fix the string-vs-number problem
try:
    ss4 = SyntheticSystem(c4)
    ss4.update(torch.rand(1, 1, 64, 64), ["OCT: x"])
    res["synthetic_update_after_lr_fix"] = "ok"
except Exception as e:
    res["synthetic_update_after_lr_fix"] = f"CRASH: {type(e).__name__}: {str(e)[:200]}"
print("SyntheticSystem.update after lr fix:", res["synthetic_update_after_lr_fix"])

# policy image encoder on one image
x = RLPolicy.preprocess_image(stub, str(EXAMPLE / rows[0]["path"]))
res["policy_preprocess_shape"] = list(x.shape)
try:
    VisionTransformer(cfg)(x)
    res["policy_image_encoder_on_one_image"] = "ok"
except Exception as e:
    res["policy_image_encoder_on_one_image"] = f"CRASH: {type(e).__name__}: {str(e)[:160]}"
res["policy_image_encoder_pretrained"] = ("config says pretrained: true, but VisionTransformer is built from "
                                          "random weights (no loading code)")
print("policy evaluate path:", res["policy_image_encoder_on_one_image"])
save("t04_rlhf", res)
