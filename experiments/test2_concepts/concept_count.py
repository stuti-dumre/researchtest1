"""Test 2: which ROCOv2 caption concepts can an image checker recognise?

Checker A: DINOv2 features, trained on the reward side (judge 1's images), scored on the judge side (judge 2's images).
Checker B: SwAV features, trained on the judge side, scored on the reward side. Same classifier recipe as the judges.
Features are cached in data/features/.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import splits as sp  # noqa: E402
import rocolib as rl  # noqa: E402
from paths import FEATURES as FEAT, ROCO  # noqa: E402

OUT = HERE / "runs"
OUT.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)
MIN_POS, AUC_MIN, BOOT = 20, 0.80, 200

df = sp.master().set_index("ID")
r_ids, j_ids = sp.judge1_ids(), sp.judge2_ids()
test = sp.test_items()


def cached(name, ids, fn):
    f = FEAT / f"{name}.npy"
    if f.exists():
        a = np.load(f)
        if len(a) == len(ids):
            return a
    t0, parts = time.time(), []
    for k in range(0, len(ids), 400):
        parts.append(fn(ids[k:k + 400]))
        print(f"  {name}: {min(k + 400, len(ids))}/{len(ids)}  {time.time() - t0:.0f} s", flush=True)
    a = np.concatenate(parts)
    np.save(f, a)
    pd.Series(ids, name="ID").to_csv(FEAT / f"{name}_ids.csv", index=False)
    return a


dino_prep = lambda ids: rl.dino_features([rl.to_square(rl.to_square(rl.open_image(i), 256), 224) for i in ids])
swav_prep = lambda ids: rl.swav_features([rl.to_square(rl.open_image(i), 256) for i in ids])
t_feat = time.time()
R_dino = np.load(sp.DER / "checker_train_feats.npy")                 # judge 1's stored features, same order
J_dino = cached("judge_side_dinov2", j_ids, dino_prep)
J_swav = cached("judge_side_swav", j_ids, swav_prep)
R_swav = cached("reward_side_swav", r_ids, swav_prep)
feat_sec = time.time() - t_feat
print(f"features ready in {feat_sec:.0f} s", flush=True)

names = pd.read_csv(ROCO / "tables" / "cui_mapping.csv")
NAME = dict(zip(names.iloc[:, 0], names.iloc[:, 1]))
manual = Counter(c for s in df.cuis_manual.dropna() for c in s.split(";") if c)


def codes(i):
    row = df.loc[i]
    return {c for col in ("cuis_auto", "cuis_manual") if isinstance(row[col], str) for c in row[col].split(";") if c}


R_codes, J_codes = [codes(i) for i in r_ids], [codes(i) for i in j_ids]
cr, cj = Counter(c for s in R_codes for c in s), Counter(c for s in J_codes for c in s)
eligible = sorted(c for c in set(cr) & set(cj) if cr[c] >= MIN_POS and cj[c] >= MIN_POS)
print(f"concepts: {len(set(cr) | set(cj))} seen on either side; eligible (>= {MIN_POS} positives on each side): "
      f"{len(eligible)}", flush=True)


def fit_score(Xtr, ytr, Xte, yte, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    rng, boots = np.random.default_rng(seed), []
    for _ in range(BOOT):
        k = rng.integers(0, len(yte), len(yte))
        if 0 < yte[k].sum() < len(k):
            boots.append(roc_auc_score(yte[k], p[k]))
    return float(auc), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def one(c):
    yr = np.array([c in s for s in R_codes], int)
    yj = np.array([c in s for s in J_codes], int)
    a = fit_score(R_dino, yr, J_dino, yj, 1)
    b = fit_score(J_swav, yj, R_swav, yr, 2)
    return {"CUI": c, "name": NAME.get(c, "?"), "category": "scan type, view, region" if c in manual else "other",
            "pos_reward_side": int(yr.sum()), "pos_judge_side": int(yj.sum()),
            "auc_A_dinov2": a[0], "ciA_low": a[1], "ciA_high": a[2], "auc_B_swav": b[0], "ciB_low": b[1],
            "ciB_high": b[2], "usable": bool(a[0] >= AUC_MIN and b[0] >= AUC_MIN)}


from joblib import Parallel, delayed  # noqa: E402
t0 = time.time()
rows = Parallel(n_jobs=8, verbose=0)(delayed(one)(c) for c in eligible)
fit_sec = time.time() - t0
res = pd.DataFrame(rows).sort_values(["usable", "auc_A_dinov2"], ascending=False)
res.to_csv(OUT / "concepts.csv", index=False)

usable_other = set(res[res.usable & res.category.eq("other")].CUI)
tc = [codes(i) & usable_other for i in test.ID]
n_other = len(usable_other)
decision = ("20 or more: concepts can be used as facts" if n_other >= 20 else
            "5 to 19: limited; concept results secondary" if n_other >= 5 else
            "fewer than 5: scan type and region only; plan must change")
summary = {"eligible": len(eligible), "eligible_by_category": res.category.value_counts().to_dict(),
           "usable": int(res.usable.sum()), "usable_by_category": res[res.usable].category.value_counts().to_dict(),
           "usable_other": n_other, "decision_rule": decision,
           "test_coverage": {"items_with_at_least_one_usable_other": float(np.mean([len(s) > 0 for s in tc])),
                             "mean_usable_other_per_item": float(np.mean([len(s) for s in tc]))},
           "seconds": {"features": round(feat_sec), "classifiers": round(fit_sec)}}
json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(summary, indent=1))
show = res[res.category.eq("other")].head(40)
print(show[["name", "pos_reward_side", "pos_judge_side", "auc_A_dinov2", "auc_B_swav", "usable"]].round(3).to_string())
print("\nscan type, view and region concepts:")
print(res[res.category.ne("other")][["name", "pos_reward_side", "pos_judge_side", "auc_A_dinov2", "auc_B_swav",
                                      "usable"]].round(3).to_string())
