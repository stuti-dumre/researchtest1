"""Test 1, set-up check. S0-1 redraws 4 stored test images of the earlier model, S0-2 checks both judges on real
images, S0-3 recovers judge 1's training IDs and builds the four-way split by article (data/pool_v1.csv).
The speed check (S0-4) is speed_test.py.
"""
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREP = ROOT / "prepare"
sys.path.insert(0, str(ROOT / "tools"))
import rocolib as rl  # noqa: E402
from paths import DATA, DERIVED as DER, MODELS  # noqa: E402

OUT = HERE / "runs"
OUT.mkdir(parents=True, exist_ok=True)
MODS = ["CT", "X-ray", "MRI", "Ultrasound"]
R = {}
mini = pd.read_csv(DATA / "roco_mini_v1.csv")
test = mini[mini.role == "test"].reset_index(drop=True)
ck = pickle.load(open(MODELS / "checker.pkl", "rb"))
j2 = pickle.load(open(MODELS / "judge2.pkl", "rb"))
REG = ck["regions"]


def ok(flag, msg):
    print(("OK    " if flag else "FAIL  ") + msg, flush=True)
    return bool(flag)


# ---------------------------------------------------------------- S0-1 redraw stored images
pick = test.groupby("modality", group_keys=False).head(1)
ids_csv = OUT / "redraw_ids.csv"
pick[["ID"]].to_csv(ids_csv, index=False)
red = OUT / "redraw_minim_paper_t1000"
t0 = time.time()
cmd = [sys.executable, "generate.py", "--run", "minim_paper_t1000", "--ids", str(ids_csv), "--steps", "15",
       "--cfg", "4.0", "--res", "256", "--seed-base", "1000", "--out", str(red)]
p = subprocess.run(cmd, cwd=PREP, capture_output=True, text=True)
(OUT / "redraw_generate_log.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
redraw_sec = time.time() - t0
stored_dir = MODELS / "minim_paper_t1000" / "gen"
stored_prompts = pd.read_csv(MODELS / "minim_paper_t1000" / "prompts.csv").drop_duplicates("ID").set_index("ID")
new_prompts = pd.read_csv(red / "prompts.csv").set_index("ID")
rows = []
for i in pick.ID:
    a = np.asarray(Image.open(stored_dir / f"{i}.png").convert("RGB"), np.int16)
    b = np.asarray(Image.open(red / "gen" / f"{i}.png").convert("RGB"), np.int16)
    d = np.abs(a - b)
    rows.append({"ID": i, "modality": pick.set_index("ID").modality[i], "max_diff": int(d.max()),
                 "mean_diff": float(d.mean()), "pixels_differing": int((d.max(-1) > 0).sum()),
                 "prompt_same": bool(stored_prompts.prompt[i] == new_prompts.prompt[i])})
R["S0-1"] = {"generate_exit": p.returncode, "seconds": round(redraw_sec, 1), "items": rows}
mx = max(r["max_diff"] for r in rows)
s01 = ok(p.returncode == 0 and all(r["prompt_same"] for r in rows) and mx == 0,
         f"S0-1 redraw: largest pixel difference {mx} over 4 images; prompts identical: "
         f"{all(r['prompt_same'] for r in rows)}; {redraw_sec:.0f} s including model loading")
R["S0-1"]["pass"] = s01

# ---------------------------------------------------------------- S0-2 judges
Ft1 = np.load(DER / "checker_test_feats.npy")
x97 = (test.modality.eq("X-ray") & test.region.isin(REG)).values
a1m = float((ck["modality"].predict(Ft1) == test.modality).mean())
a1r = float((ck["region"].predict(Ft1[x97]) == test.region[x97]).mean())
s02a = ok(a1m == float(ck["acc_real"]["modality"]) and a1r == float(ck["acc_real"]["region"]),
          f"S0-2a judge 1 on stored real features: scan type {a1m:.1%} (recorded {ck['acc_real']['modality']:.1%}), "
          f"region {a1r:.1%} (recorded {ck['acc_real']['region']:.1%})")
t0 = time.time()
Ft2 = rl.swav_features([rl.to_square(rl.open_image(i), 256) for i in test.ID])
a2m = float((j2["modality"].predict(Ft2) == test.modality).mean())
a2r = float((j2["region"].predict(Ft2[x97]) == test.region[x97]).mean())
rec2m, rec2r = j2["acc_real"]["scan type (400 real test images)"], j2["acc_real"]["X-ray region (97 real test images)"]
s02b = ok(abs(a2m - rec2m) <= 0.005 and abs(a2r - rec2r) <= 0.005,
          f"S0-2b judge 2 on new SwAV features: scan type {a2m:.1%} (recorded {rec2m:.1%}), region {a2r:.1%} "
          f"(recorded {rec2r:.1%}); {time.time() - t0:.0f} s for 400 images")
jc = pd.read_csv(HERE / "stored_judge1.csv")
stored_seen = jc.drop_duplicates("ID").set_index("ID").seen_mod
Fr = rl.dino_features([Image.open(red / "gen" / f"{i}.png") for i in pick.ID])
seen_new = dict(zip(pick.ID, ck["modality"].predict(Fr)))
s02c = ok(all(seen_new[i] == stored_seen[i] for i in pick.ID),
          f"S0-2c judge 1 on the redrawn images: {seen_new} vs stored {stored_seen[list(pick.ID)].to_dict()}")
R["S0-2"] = {"judge1_real": {"scan_type": a1m, "region": a1r}, "judge2_real": {"scan_type": a2m, "region": a2r},
             "judge1_on_redraw": seen_new, "pass": s02a and s02b and s02c}

# ---------------------------------------------------------------- S0-3 four-way split by article
df = pd.read_pickle(DER / "rocov2_master.pkl")
test_arts = set(test.PMCID)
used = set(mini[mini.role.isin(["gen_train", "test"])].ID)
cand = df[df.split_official.isin(["valid", "test"]) & df.modality.isin(MODS) & ~df.PMCID.isin(test_arts)
          & ~df.ID.isin(used)]
pick1 = cand.groupby("modality", group_keys=False).apply(lambda g: g.sample(min(len(g), 900), random_state=5))
xr_extra = cand[(cand.modality == "X-ray") & cand.region.isin(REG) & ~cand.ID.isin(pick1.ID)]
j1_ids = list(pd.concat([pick1, xr_extra.sample(min(len(xr_extra), 1200), random_state=5)]).ID)
Ftr = np.load(DER / "checker_train_feats.npy")
probe = [0, 1, 2, 3, len(j1_ids) - 4, len(j1_ids) - 3, len(j1_ids) - 2, len(j1_ids) - 1]
Fp = rl.dino_features([rl.to_square(rl.to_square(rl.open_image(j1_ids[k]), 256), 224) for k in probe])
feat_diff = float(np.abs(Fp - Ftr[probe]).max())
rec1 = ok(len(j1_ids) == 4800 and len(set(j1_ids)) == 4800 and feat_diff < 1e-4,
          f"S0-3 judge 1 IDs recovered: {len(j1_ids)} IDs; largest feature difference on 8 probes {feat_diff:.2e}")
j2_ids = list(j2["train_ids"])
art = df.set_index("ID").PMCID
a_test, a_j1, a_j2 = test_arts, set(art[j1_ids].dropna()), set(art[j2_ids].dropna())
a_chk = set(mini[mini.role == "checker"].PMCID)
excluded = a_test | a_j1 | a_j2 | a_chk
base = df[df.modality.isin(MODS)]
no_art = int(base.PMCID.isna().sum())
pool = base[base.PMCID.notna() & ~base.PMCID.isin(excluded)]
keep = [c for c in ["ID", "PMCID", "modality", "region", "view", "split_official", "n_tokens"] if c in pool.columns]
keep += [c for c in pool.columns if c.startswith("flag_")]
pool_path = DATA / "pool_v1.csv"
pool[keep].sort_values("ID").to_csv(pool_path, index=False)
sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
(pool_path.parent / "pool_v1.csv.sha256").write_text(f"{sha}  pool_v1.csv", encoding="ascii")
per_type = pool.modality.value_counts().reindex(MODS).to_dict()
pa = set(pool.PMCID)
no_overlap = not (pa & a_test) and not (pa & a_j1) and not (pa & a_j2)
s03 = ok(rec1 and no_overlap and min(per_type.values()) >= 2000,
         f"S0-3 pool: {len(pool):,} images from {len(pa):,} articles; per scan type {per_type}; no shared article with "
         f"test/judge 1/judge 2: {no_overlap}")
gt = mini[(mini.role == "gen_train") & (mini.tier <= 1000)]
gt_art = gt.PMCID
R["S0-3"] = {"judge1_ids": len(j1_ids), "judge1_feature_check_max_diff": feat_diff, "judge2_ids": len(j2_ids),
             "articles": {"test": len(a_test), "judge1": len(a_j1), "judge2": len(a_j2), "checker_role": len(a_chk),
                          "judge1_and_judge2_shared": len(a_j1 & a_j2)},
             "four_types_total": len(base), "no_article_id": no_art, "pool_images": len(pool),
             "pool_articles": len(pa), "pool_per_type": per_type, "pool_sha256": sha,
             "existing_generator_tier1000": {"images": len(gt),
                                             "in_judge1_articles": int(gt_art.isin(a_j1).sum()),
                                             "in_judge2_articles": int(gt_art.isin(a_j2).sum()),
                                             "in_either": int(gt_art.isin(a_j1 | a_j2).sum())},
             "pass": s03}
print("info: judge 1 and judge 2 share", len(a_j1 & a_j2), "articles; existing generator (tier 1000):",
      R["S0-3"]["existing_generator_tier1000"], flush=True)
json.dump(R, open(OUT / "setup_results.json", "w", encoding="utf-8"), indent=1, default=str)
print("saved", OUT / "setup_results.json", "| pool sha256", sha)
