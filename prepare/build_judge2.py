"""Build judge 2, the second scan-type and X-ray region judge, before it sees any generated image.

SwAV ResNet-50 features (self-supervised on ImageNet; judge 1 uses DINOv2), standardised, logistic regression (C 0.5,
class-balanced, as judge 1). Trained on ROCOv2 official-train-split images from articles outside the ROCO-mini test set
and not used anywhere in ROCO-mini: 800 per scan type (seed 7) plus up to 1,500 more X-ray images with a region label.
Writes models/judge2.pkl and reports its accuracy on the real ROCO-mini test images (and its agreement with judge 1).
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from common_sd import SPLIT  # noqa: E402
from paths import DERIVED, MODELS  # noqa: E402
import rocolib as rl  # noqa: E402

MODS = ["CT", "X-ray", "MRI", "Ultrasound"]
ck1 = pickle.load(open(MODELS / "checker.pkl", "rb"))
REG = ck1["regions"]
mini = pd.read_csv(SPLIT)
df = pd.read_pickle(DERIVED / "rocov2_master.pkl")
test = mini[mini.role == "test"].reset_index(drop=True)
cand = df[(df.split_official == "train") & df.modality.isin(MODS) & ~df.PMCID.isin(set(test.PMCID))
          & ~df.ID.isin(set(mini.ID))]
pick = cand.groupby("modality", group_keys=False).apply(lambda g: g.sample(min(len(g), 800), random_state=7))
extra = cand[(cand.modality == "X-ray") & cand.region.isin(REG) & ~cand.ID.isin(set(pick.ID))]
train = pd.concat([pick, extra.sample(min(len(extra), 1500), random_state=7)]).reset_index(drop=True)
print(f"judge 2 training images: {len(train)} ({train.modality.value_counts().to_dict()}); "
      f"X-ray with a region label: {int((train.modality.eq('X-ray') & train.region.isin(REG)).sum())}")

t0 = time.time()
F = rl.swav_features([rl.to_square(rl.open_image(i), 256) for i in train.ID])
print(f"features in {time.time() - t0:.0f} s")
clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
mod = clf().fit(F, train.modality)
xr = (train.modality == "X-ray").values & train.region.isin(REG).values
reg = clf().fit(F[xr], train.region[xr])

Ft = rl.swav_features([rl.to_square(rl.open_image(i), 256) for i in test.ID])
Fd = rl.dino_features([rl.to_square(rl.open_image(i), 256) for i in test.ID])
x97 = (test.modality == "X-ray").values & test.region.isin(REG).values
acc = {"scan type (400 real test images)": float((mod.predict(Ft) == test.modality).mean()),
       "X-ray region (97 real test images)": float((reg.predict(Ft[x97]) == test.region[x97]).mean()),
       "agreement with judge 1, scan type": float((mod.predict(Ft) == ck1["modality"].predict(Fd)).mean()),
       "agreement with judge 1, region": float((reg.predict(Ft[x97]) == ck1["region"].predict(Fd[x97])).mean())}
for k, v in acc.items():
    print(f"{k}: {v:.1%}")
pickle.dump({"modality": mod, "region": reg, "regions": REG, "mods": MODS, "acc_real": acc,
             "train_ids": train.ID.tolist()}, open(MODELS / "judge2.pkl", "wb"))
print("saved", MODELS / "judge2.pkl")
