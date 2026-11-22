"""Test 3: larger checker sides (split v2) and negation-aware labels.

Writes data/split_v2.csv and data/pool_v2.csv. Features are cached in data/features/ (test 2's caches are reused for
the original checker images).
"""
import hashlib
import json
import re
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
from paths import DATA, FEATURES as FEAT, ROCO  # noqa: E402

OUT = HERE / "runs"
OUT.mkdir(parents=True, exist_ok=True)
MIN_POS, AUC_MIN, BOOT, ADD = 20, 0.80, 200, 10000
T0 = time.time()

# ------------------------------------------------------------------ change 1: split v2 by article
df = sp.master().set_index("ID")
pool1 = sp.pool()
arts = np.array(sorted(pool1.PMCID.unique()))
rng = np.random.default_rng(15)
rng.shuffle(arts)
size = pool1.groupby("PMCID").size()
part_of_art, n_r, n_j = {}, 0, 0
for a in arts:
    if n_r < ADD:
        part_of_art[a], n_r = "reward_extra", n_r + size[a]
    elif n_j < ADD:
        part_of_art[a], n_j = "judge_extra", n_j + size[a]
    else:
        part_of_art[a] = "pool"
split2 = pool1.assign(part=pool1.PMCID.map(part_of_art))
pool2 = split2[split2.part == "pool"].drop(columns="part")
for name, frame in (("split_v2.csv", split2[["ID", "PMCID", "modality", "part"]]), ("pool_v2.csv", pool2)):
    p = DATA / name
    frame.sort_values("ID").to_csv(p, index=False)
    (p.parent / f"{name}.sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {name}", encoding="ascii")
r_extra = sorted(split2[split2.part == "reward_extra"].ID)
j_extra = sorted(split2[split2.part == "judge_extra"].ID)
r0, j0 = sp.judge1_ids(), sp.judge2_ids()
test = sp.test_items()
A = lambda ids: set(df.loc[ids].PMCID)
sides = {"reward": A(r0) | A(r_extra), "judge": A(j0) | A(j_extra), "pool": set(pool2.PMCID), "test": set(test.PMCID)}
names_ = list(sides)
shared = {f"{a}&{b}": len(sides[a] & sides[b]) for i, a in enumerate(names_) for b in names_[i + 1:]}
per_type = pool2.modality.value_counts().reindex(sp.MODS).to_dict()
split_ok = all(v == 0 for v in shared.values()) and min(per_type.values()) >= 2000
print(f"split v2: reward +{len(r_extra)} images, judge +{len(j_extra)}, pool {len(pool2)} {per_type}; "
      f"shared articles {shared}; ok={split_ok}", flush=True)


# ------------------------------------------------------------------ features for the moved images (one read per image)
def both_feats(name, ids):
    fd, fs = FEAT / f"{name}_dinov2.npy", FEAT / f"{name}_swav.npy"
    if fd.exists() and fs.exists():
        d, s = np.load(fd), np.load(fs)
        if len(d) == len(s) == len(ids):
            return d, s
    D, S, t0 = [], [], time.time()
    for k in range(0, len(ids), 400):
        ims = [rl.to_square(rl.open_image(i), 256) for i in ids[k:k + 400]]
        D.append(rl.dino_features([rl.to_square(im, 224) for im in ims]))
        S.append(rl.swav_features(ims))
        print(f"  {name}: {min(k + 400, len(ids))}/{len(ids)}  {time.time() - t0:.0f} s", flush=True)
    D, S = np.concatenate(D), np.concatenate(S)
    np.save(fd, D)
    np.save(fs, S)
    pd.Series(ids, name="ID").to_csv(FEAT / f"{name}_ids.csv", index=False)
    return D, S


t_feat = time.time()
Rx_d, Rx_s = both_feats("reward_extra", r_extra)
Jx_d, Jx_s = both_feats("judge_extra", j_extra)
R_dino = np.concatenate([np.load(sp.DER / "checker_train_feats.npy"), Rx_d])
R_swav = np.concatenate([np.load(FEAT / "reward_side_swav.npy"), Rx_s])
J_dino = np.concatenate([np.load(FEAT / "judge_side_dinov2.npy"), Jx_d])
J_swav = np.concatenate([np.load(FEAT / "judge_side_swav.npy"), Jx_s])
r_ids, j_ids = r0 + r_extra, j0 + j_extra
feat_sec = time.time() - t_feat
print(f"features ready in {feat_sec:.0f} s: reward side {len(r_ids)}, judge side {len(j_ids)}", flush=True)

# ------------------------------------------------------------------ change 2: negation-aware labels
names = pd.read_csv(ROCO / "tables" / "cui_mapping.csv")
NAME = dict(zip(names.iloc[:, 0], names.iloc[:, 1]))
manual = Counter(c for s in df.cuis_manual.dropna() for c in s.split(";") if c)
DROP = {"structure", "disorder", "entire", "finding", "device", "procedure", "region", "part", "body", "system", "of",
        "the", "and", "nos"}
FILL = {"evidence", "sign", "signs", "any", "of", "obvious", "significant", "definite", "for", "the", "a", "an"}
TRIG = {"no", "not", "without", "absent", "absence", "negative", "free", "nor", "resolved", "resolution", "excluded"}


def head(cui):
    words = [w for w in re.findall(r"[a-z]+", re.sub(r"\(.*?\)", " ", NAME.get(cui, "").lower()))
             if w not in DROP and len(w) >= 4]
    return words[-1] if words else None


HEAD = {c: head(c) for c in NAME}


def negated_heads(caption):
    out = set()
    for sent in re.split(r"[.;:!?]", str(caption).lower()):
        toks = re.findall(r"[a-z]+", sent)
        for k, t in enumerate(toks):
            is_trig = t in TRIG or (t == "ruled" and k + 1 < len(toks) and toks[k + 1] == "out")
            if not is_trig:
                continue
            nxt = [w for w in toks[k + 1:k + 12] if w not in FILL and w != "out"][:3]
            out.update(nxt)
    return out


def codes(i, neg):
    row = df.loc[i]
    cs = {c for col in ("cuis_auto", "cuis_manual") if isinstance(row[col], str) for c in row[col].split(";") if c}
    if neg:
        words = negated_heads(row.Caption)
        cs = {c for c in cs if not (HEAD.get(c) and any(w.startswith(HEAD[c]) for w in words))}
    return cs


labels = {}
for neg in (False, True):
    labels[neg] = ([codes(i, neg) for i in r_ids], [codes(i, neg) for i in j_ids], [codes(i, neg) for i in test.ID])
removed = sum(len(a) - len(b) for x, y in zip(labels[False][:2], labels[True][:2]) for a, b in zip(x, y))
print(f"negation rule removed {removed} positive labels on the two sides", flush=True)


# ------------------------------------------------------------------ checkers
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
    g, boots = np.random.default_rng(seed), []
    for _ in range(BOOT):
        k = g.integers(0, len(yte), len(yte))
        if 0 < yte[k].sum() < len(k):
            boots.append(roc_auc_score(yte[k], p[k]))
    return float(auc), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def one(c, Rc, Jc, version):
    yr = np.array([c in s for s in Rc], int)
    yj = np.array([c in s for s in Jc], int)
    a = fit_score(R_dino, yr, J_dino, yj, 1)
    b = fit_score(J_swav, yj, R_swav, yr, 2)
    return {"version": version, "CUI": c, "name": NAME.get(c, "?"),
            "category": "scan type, view, region" if c in manual else "other",
            "pos_reward_side": int(yr.sum()), "pos_judge_side": int(yj.sum()), "auc_A_dinov2": a[0], "ciA_low": a[1],
            "auc_B_swav": b[0], "ciB_low": b[1], "usable": bool(a[0] >= AUC_MIN and b[0] >= AUC_MIN)}


from joblib import Parallel, delayed  # noqa: E402
t_fit, rows, summary = time.time(), [], {"split_v2": {"reward_added": len(r_extra), "judge_added": len(j_extra),
                                                         "pool_v2_images": len(pool2), "pool_v2_per_type": per_type,
                                                         "shared_articles": shared, "checks_ok": split_ok},
                                            "negation_removed_labels": removed}
for neg, version in ((True, "main: larger sides + negation rule"), (False, "larger sides only")):
    Rc, Jc, Tc = labels[neg]
    cr, cj = Counter(c for s in Rc for c in s), Counter(c for s in Jc for c in s)
    elig = sorted(c for c in set(cr) & set(cj) if cr[c] >= MIN_POS and cj[c] >= MIN_POS)
    res = pd.DataFrame(Parallel(n_jobs=8)(delayed(one)(c, Rc, Jc, version) for c in elig))
    rows.append(res)
    uo = set(res[res.usable & res.category.eq("other")].CUI)
    tc = [s & uo for s in Tc]
    summary[version] = {"eligible": len(elig), "eligible_other": int(res.category.eq("other").sum()),
                        "usable": int(res.usable.sum()), "usable_other": len(uo),
                        "usable_other_both_ci_low_ge_0.80": int((res.usable & res.category.eq("other")
                                                                  & (res.ciA_low >= 0.8) & (res.ciB_low >= 0.8)).sum()),
                        "test_items_with_usable_other": float(np.mean([len(s) > 0 for s in tc])),
                        "mean_usable_other_per_test_item": float(np.mean([len(s) for s in tc]))}
    print(version, summary[version], flush=True)
allres = pd.concat(rows)
allres.to_csv(OUT / "concepts_v2.csv", index=False)
old = pd.read_csv(HERE.parent / "test2_concepts" / "runs" / "concepts.csv")
old16 = set(old[old.usable & old.category.eq("other")].CUI)
main = allres[allres.version.str.startswith("main")]
kept = old16 & set(main[main.usable].CUI)
n = summary["main: larger sides + negation rule"]["usable_other"]
summary["test2_16_still_usable"] = len(kept)
summary["decision_rule"] = ("20 or more: concepts become a main part" if n >= 20 else
                            "5 to 19: concepts stay secondary; pilot uses split v2" if n >= 5 else
                            "fewer than 5: investigate before the pilot")
summary["seconds"] = {"features": round(feat_sec), "classifiers": round(time.time() - t_fit),
                      "total": round(time.time() - T0)}
json.dump(summary, open(OUT / "summary_v2.json", "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
