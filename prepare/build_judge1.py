"""Build judge 1: DINOv2 features and two logistic regressions, one for the scan type and one for the X-ray body
region.

Training images come from the valid and test zips, from articles outside the ROCO-mini test set, and are not ROCO-mini
generator training or test images: up to 900 per scan type (seed 5) plus up to 1,200 more X-ray images with a region
label. Writes models/checker.pkl and the features of its training and test images (data/rocov2/derived/).
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rocolib as rl  # noqa: E402
from paths import DATA, DERIVED, MODELS  # noqa: E402

MODS = ["CT", "X-ray", "MRI", "Ultrasound"]
REGIONS = ["Chest", "Head", "Abdomen", "Pelvis", "Spine", "Upper limb", "Lower limb"]
MODELS.mkdir(parents=True, exist_ok=True)
df = pd.read_pickle(DERIVED / "rocov2_master.pkl")
mini = pd.read_csv(DATA / "roco_mini_v1.csv")
test_arts = set(mini[mini.role == "test"].PMCID)
used = set(mini[mini.role.isin(["gen_train", "test"])].ID)
cand = df[df.split_official.isin(["valid", "test"]) & df.modality.isin(MODS) & ~df.PMCID.isin(test_arts)
          & ~df.ID.isin(used)]
pick = cand.groupby("modality", group_keys=False).apply(lambda g: g.sample(min(len(g), 900), random_state=5))
xr_extra = cand[(cand.modality == "X-ray") & cand.region.isin(REGIONS) & ~cand.ID.isin(pick.ID)]
train = pd.concat([pick, xr_extra.sample(min(len(xr_extra), 1200), random_state=5)])
test = mini[mini.role == "test"]
print(f"training images: {len(train)} {train.modality.value_counts().reindex(MODS).to_dict()}")


def feats_for(ids):
    return rl.dino_features([rl.to_square(rl.to_square(rl.open_image(i), 256), 224) for i in ids])


t = time.time()
F_train, F_test = feats_for(train.ID), feats_for(test.ID)
print(f"features: {len(F_train)} training + {len(F_test)} test images in {time.time() - t:.0f} s")
np.save(DERIVED / "checker_train_feats.npy", F_train)
np.save(DERIVED / "checker_test_feats.npy", F_test)

mod_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
mod_clf.fit(F_train, train.modality)
acc_mod = accuracy_score(test.modality, mod_clf.predict(F_test))
xr_tr = train.modality.eq("X-ray").values & train.region.isin(REGIONS).values
reg_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
reg_clf.fit(F_train[xr_tr], train.region[xr_tr])
xr_te = test.modality.eq("X-ray").values & test.region.isin(REGIONS).values
acc_reg = accuracy_score(test.region[xr_te], reg_clf.predict(F_test[xr_te]))
print(f"scan type on 400 real test images: {100 * acc_mod:.1f}%; region on {xr_te.sum()} test X-rays: "
      f"{100 * acc_reg:.1f}%")
pickle.dump({"modality": mod_clf, "region": reg_clf, "regions": REGIONS, "mods": MODS,
             "acc_real": {"modality": acc_mod, "region": acc_reg}}, open(MODELS / "checker.pkl", "wb"))
print("saved", MODELS / "checker.pkl")
