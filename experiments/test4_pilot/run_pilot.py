"""Test 4: best of 4 picked by reward-side checkers, with a copy and marks filter, against a random choice of 4 and the
base model. Only judge-side checkers score the results.

Resumable: every stage skips work that is already saved. --smoke runs tiny settings in runs_smoke/ to check the code.
"""
import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import splits as sp  # noqa: E402
import concepts as cc  # noqa: E402
import sd  # noqa: E402

import rocolib as rl  # noqa: E402
from paths import DATA, FEATURES, MODELS  # noqa: E402
from texts import make_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="tiny settings in runs_smoke/ to check the code (no results)")
args = ap.parse_args()
S = args.smoke
RUN = HERE / ("runs_smoke" if S else "runs")
for d in ("lists", "latents/base", "latents/pick", "latents/random", "models", "cand", "test/base", "test/random",
          "test/pick", "checkers"):
    (RUN / d).mkdir(parents=True, exist_ok=True)
N_BASE, N_REW, K, BASE_STEPS, FT_STEPS = (5, 1, 4, 3, 3) if S else (250, 125, 4, 1200, 300)
DRAW_STEPS = 2 if S else 15
LOG = RUN / "pilot_log.txt"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha(path):
    p = Path(path)
    (p.parent / f"{p.name}.sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}", encoding="ascii")


def marks_share(img):
    a = np.asarray(img.convert("RGB"), np.int16)
    return float(((a.max(-1) - a.min(-1)) > 60).mean())


def has_marks(share):
    return 0.0005 < share < 0.05


log(f"=== test 4 {'SMOKE ' if S else ''}start")
M = sp.master()
MI = M.set_index("ID")
test = sp.test_items()
if S:
    test = test.groupby("modality", group_keys=False).head(2).reset_index(drop=True)
ck1 = pickle.load(open(MODELS / "checker.pkl", "rb"))
ck2 = pickle.load(open(MODELS / "judge2.pkl", "rb"))
REG = ck1["regions"]

# ------------------------------------------------------------------ S1 item lists
base_csv, rew_csv = RUN / "lists" / "base_items.csv", RUN / "lists" / "reward_items.csv"
if not base_csv.exists():
    pool = pd.read_csv(DATA / "pool_v2.csv")[["ID"]].merge(M, on="ID")
    base = pool.groupby("modality", group_keys=False).apply(lambda g: g.sample(N_BASE, random_state=0))
    rest = pool[~pool.PMCID.isin(set(base.PMCID))]
    rew = rest.groupby("modality", group_keys=False).apply(lambda g: g.sample(N_REW, random_state=1))
    for frame, path in ((base, base_csv), (rew, rew_csv)):
        frame = frame.assign(text=[make_text(r, "minim_paper") for r in frame.to_dict("records")])
        frame[["ID", "PMCID", "modality", "region", "text"]].reset_index(drop=True).to_csv(path, index=False)
        sha(path)
    log(f"S1 lists: base {len(base)}, reward captions {len(rew)}; shared articles "
        f"{len(set(base.PMCID) & set(rew.PMCID))}")
base, rew = pd.read_csv(base_csv), pd.read_csv(rew_csv)

# ------------------------------------------------------------------ S2-S3 base model
lat_base = [RUN / "latents" / "base" / f"{i}.npy" for i in base.ID]
todo = [(i, p) for i, p in zip(base.ID, lat_base) if not p.exists()]
if todo:
    sd.encode_latents([rl.open_image(i) for i, _ in todo], [p for _, p in todo])
    log(f"S2 latents for {len(todo)} base images")
sd.train(base, lat_base, BASE_STEPS, RUN / "models" / "base", seed=0, log=log)
gc.collect()

# ------------------------------------------------------------------ S4 concept checkers A (reward side) and B (judge side)
USE = cc.usable_other()[:3] if S else cc.usable_other()
ckA_p, ckB_p = RUN / "checkers" / "concept_A.pkl", RUN / "checkers" / "concept_B.pkl"
if not ckB_p.exists():
    from joblib import Parallel, delayed
    FE = FEATURES
    r_ids = sp.judge1_ids() + list(pd.read_csv(FE / "reward_extra_ids.csv").ID)
    j_ids = sp.judge2_ids() + list(pd.read_csv(FE / "judge_extra_ids.csv").ID)
    R_dino = np.concatenate([np.load(sp.DER / "checker_train_feats.npy"), np.load(FE / "reward_extra_dinov2.npy")])
    J_swav = np.concatenate([np.load(FE / "judge_side_swav.npy"), np.load(FE / "judge_extra_swav.npy")])
    if S:
        r_ids, j_ids, R_dino, J_swav = r_ids[:3000], j_ids[:3000], R_dino[:3000], J_swav[:3000]
    Rc = [cc.codes(MI.loc[i]) for i in r_ids]
    Jc = [cc.codes(MI.loc[i]) for i in j_ids]

    def fit(X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")).fit(X, y)

    A = Parallel(n_jobs=8)(delayed(fit)(R_dino, np.array([c in s for s in Rc], int)) for c in USE)
    B = Parallel(n_jobs=8)(delayed(fit)(J_swav, np.array([c in s for s in Jc], int)) for c in USE)
    pickle.dump(dict(zip(USE, A)), open(ckA_p, "wb"))
    pickle.dump(dict(zip(USE, B)), open(ckB_p, "wb"))
    log(f"S4 concept checkers trained for {len(USE)} concepts")
CA, CB = pickle.load(open(ckA_p, "rb")), pickle.load(open(ckB_p, "rb"))

# ------------------------------------------------------------------ S5 candidates from the base model
cand_jobs = [(r.text, 50000 + 4 * k + j, RUN / "cand" / f"{r.ID}_{j}.png") for k, r in enumerate(rew.itertuples())
             for j in range(K)]
if not all(p.exists() for _, _, p in cand_jobs):
    pipe = sd.pipeline(RUN / "models" / "base" / "lora.pt")
    sd.draw(pipe, cand_jobs, steps=DRAW_STEPS, log=log, label="S5 candidates")
    del pipe
    gc.collect()


# ------------------------------------------------------------------ S6 rewards, guard, picks (reward side only)
def j1_probs(F, mods, regs):
    pm = ck1["modality"].predict_proba(F)
    cls = list(ck1["modality"].classes_)
    p_mod = pm[np.arange(len(F)), [cls.index(m) for m in mods]]
    pr = ck1["region"].predict_proba(F)
    rcls = list(ck1["region"].classes_)
    p_reg = np.array([pr[k, rcls.index(r)] if (m == "X-ray" and r in rcls) else np.nan
                      for k, (m, r) in enumerate(zip(mods, regs))])
    return p_mod, p_reg


picks_csv = RUN / "lists" / "picks.csv"
if not picks_csv.exists():
    Fb_d = rl.dino_features([rl.to_square(rl.open_image(i), 256) for i in base.ID])
    Ft_d = np.load(sp.DER / "checker_test_feats.npy")
    full_test = sp.test_items()
    sim_t, _ = rl.cosine_nn(Ft_d, Fb_d)
    line_d = {m: float(np.percentile(sim_t[(full_test.modality == m).values], 95)) for m in sp.MODS}
    rows = []
    for k, r in enumerate(rew.itertuples()):
        cs = cc.codes(MI.loc[r.ID]) & set(CA)
        ims = [Image.open(RUN / "cand" / f"{r.ID}_{j}.png").convert("RGB") for j in range(K)]
        F = rl.dino_features(ims)
        p_mod, p_reg = j1_probs(F, [r.modality] * K, [r.region] * K)
        p_con = np.array([[CA[c].predict_proba(F[j:j + 1])[0, 1] for c in sorted(cs)] for j in range(K)]) \
            if cs else np.zeros((K, 0))
        sim, _ = rl.cosine_nn(F, Fb_d)
        for j in range(K):
            terms = [p_mod[j]] + ([p_reg[j]] if not np.isnan(p_reg[j]) else []) + list(p_con[j])
            ms = marks_share(ims[j])
            rows.append({"ID": r.ID, "j": j, "modality": r.modality, "reward": float(np.mean(terms)),
                         "n_terms": len(terms), "n_concepts": len(cs), "sim_dino": float(sim[j]),
                         "copy_guard": bool(sim[j] > line_d[r.modality]), "marks_share": ms,
                         "marks_guard": has_marks(ms)})
    cand = pd.DataFrame(rows)
    cand["ok"] = ~(cand.copy_guard | cand.marks_guard)
    cand.to_csv(RUN / "lists" / "candidates_scored.csv", index=False)
    g = np.random.default_rng(2)
    out = []
    for i, grp in cand.groupby("ID", sort=False):
        okg = grp[grp.ok]
        if okg.empty:
            continue
        best = okg.sort_values(["reward", "j"], ascending=[False, True]).iloc[0].j
        rnd = okg.j.values[g.integers(len(okg))]
        out.append({"ID": i, "pick_j": int(best), "random_j": int(rnd)})
    picks = pd.DataFrame(out).merge(rew[["ID", "text", "modality"]], on="ID")
    picks.to_csv(picks_csv, index=False)
    sha(picks_csv)
    json.dump({"copy_line_dinov2": line_d}, open(RUN / "lists" / "guard_lines.json", "w"), indent=1)
    log(f"S6 candidates scored: {len(cand)}; guarded copy {int(cand.copy_guard.sum())}, marks "
        f"{int(cand.marks_guard.sum())}; captions kept {len(picks)} of {len(rew)}; "
        f"pick = random for {int((picks.pick_j == picks.random_j).sum())}")
picks = pd.read_csv(picks_csv)

# ------------------------------------------------------------------ S7 fine-tune Random and Pick from Base
for arm in ("random", "pick"):
    paths = [RUN / "latents" / arm / f"{i}.npy" for i in picks.ID]
    todo = [(i, j, p) for i, j, p in zip(picks.ID, picks[f"{arm}_j"], paths) if not p.exists()]
    if todo:
        sd.encode_latents([Image.open(RUN / "cand" / f"{i}_{j}.png") for i, j, _ in todo], [p for _, _, p in todo])
        log(f"S7 latents for {len(todo)} {arm} images")
    sd.train(picks, paths, FT_STEPS, RUN / "models" / arm, init_lora=RUN / "models" / "base" / "lora.pt", seed=0,
             log=log)
    gc.collect()

# ------------------------------------------------------------------ S8 test images for the three models
ttext = [make_text(r, "minim_paper", for_eval=True) for r in test.to_dict("records")]
tseed = [1000 + int(i.split("_")[-1]) for i in test.ID]
for arm in ("base", "random", "pick"):
    jobs = [(t, s, RUN / "test" / arm / f"{i}.png") for t, s, i in zip(ttext, tseed, test.ID)]
    if not all(p.exists() for _, _, p in jobs):
        pipe = sd.pipeline(RUN / "models" / arm / "lora.pt")
        sd.draw(pipe, jobs, steps=DRAW_STEPS, log=log, label=f"S8 test {arm}")
        del pipe
        gc.collect()

# ------------------------------------------------------------------ S9 scoring (judge side; judge 1 only for inflation)
log("S9 scoring")
full_test = sp.test_items()
Fb_s = rl.swav_features([rl.to_square(rl.open_image(i), 256) for i in base.ID])
Ft_s_real = rl.swav_features([rl.to_square(rl.open_image(i), 256) for i in full_test.ID])
sim_rt, _ = rl.cosine_nn(Ft_s_real, Fb_s)
line_s = {m: float(np.percentile(sim_rt[(full_test.modality == m).values], 95)) for m in sp.MODS}
used_arts = set(base.PMCID) | set(rew.PMCID)
pool = pd.read_csv(DATA / "pool_v2.csv")
spare = pool[~pool.PMCID.isin(used_arts)].groupby("modality", group_keys=False).apply(
    lambda g: g.sample(2 if S else 100, random_state=3))
F_spare = rl.swav_features([rl.to_square(rl.open_image(i), 256) for i in spare.ID])
real_idx = [list(full_test.ID).index(i) for i in test.ID]
F_real = Ft_s_real[real_idx]
kid_rr = rl.kid(F_spare, F_real, n_subsets=100)
xr = (test.modality.eq("X-ray") & test.region.isin(REG)).values
tcodes = [cc.codes(MI.loc[i]) & set(CB) for i in test.ID]
has_c = np.array([len(s) > 0 for s in tcodes])
J, per_item = {}, {}
for arm in ("base", "random", "pick"):
    ims = [Image.open(RUN / "test" / arm / f"{i}.png").convert("RGB") for i in test.ID]
    Fs, Fd = rl.swav_features(ims), rl.dino_features(ims)
    j2m, j2r = ck2["modality"].predict(Fs), ck2["region"].predict(Fs)
    j1m = ck1["modality"].predict(Fd)
    sim, _ = rl.cosine_nn(Fs, Fb_s)
    copy = np.array([s > line_s[m] for s, m in zip(sim, test.modality)])
    mk = np.array([has_marks(marks_share(im)) for im in ims])
    hit = np.array([np.mean([CB[c].predict(Fs[k:k + 1])[0] for c in sorted(s)]) if s else np.nan
                    for k, s in enumerate(tcodes)])
    per_item[arm] = {"j2_mod": (j2m == test.modality.values).astype(float),
                     "j2_reg": (j2r == test.region.values).astype(float)[xr],
                     "j1_mod": (j1m == test.modality.values).astype(float), "copy": copy.astype(float),
                     "marks": mk.astype(float), "concept": hit[has_c]}
    J[arm] = {"scan_type_judge2": float(per_item[arm]["j2_mod"].mean()),
              "region_judge2": float(per_item[arm]["j2_reg"].mean()) if xr.any() else None,
              "scan_type_judge1": float(per_item[arm]["j1_mod"].mean()),
              "concept_hit_rate_B": float(np.nanmean(hit)) if has_c.any() else None,
              "copy_share": float(copy.mean()), "marks_share": float(mk.mean()),
              "kid_swav": rl.kid(Fs, F_real, n_subsets=100)[0]}
    pd.DataFrame({"ID": test.ID, "modality": test.modality, "seen_j2": j2m, "seen_j1": j1m, "sim_swav": sim,
                  "copy": copy, "marks": mk, "concept_hit": hit}).to_csv(RUN / f"judged_{arm}.csv", index=False)


def boot(d, n=2000, seed=4):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return {"diff": None, "low": None, "high": None, "n": 0}
    g = np.random.default_rng(seed)
    bs = [d[g.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"diff": float(d.mean()), "low": float(np.percentile(bs, 2.5)), "high": float(np.percentile(bs, 97.5)),
            "n": int(len(d))}


C = {}
for a, b in (("pick", "random"), ("pick", "base"), ("random", "base")):
    C[f"{a}-{b}"] = {k: boot(per_item[a][k] - per_item[b][k]) for k in ("j2_mod", "j2_reg", "j1_mod", "copy", "marks",
                                                                          "concept")}
infl = boot((per_item["pick"]["j1_mod"] - per_item["random"]["j1_mod"])
            - (per_item["pick"]["j2_mod"] - per_item["random"]["j2_mod"]))
P1 = C["pick-random"]["j2_mod"]["low"] is not None and C["pick-random"]["j2_mod"]["low"] > 0
P2 = J["pick"]["copy_share"] <= 0.05
P3 = C["pick-random"]["concept"]["low"] is not None and C["pick-random"]["concept"]["low"] > 0
res = {"smoke": S, "models": J, "comparisons": C, "inflation_pick_minus_random": infl, "kid_real_vs_real": kid_rr[0],
       "copy_line_swav": line_s, "verdicts": {"P1_scan_type_pick_gt_random": P1, "P2_copy_share_le_5pct": P2,
                                              "P3_concepts_pick_gt_random": P3},
       "test_items": len(test), "test_items_with_concepts": int(has_c.sum()), "captions_kept": len(picks)}
json.dump(res, open(RUN / "results.json", "w"), indent=1, default=float)
log(f"S9 done: verdicts {res['verdicts']}")
log("=== test 4 finished")
