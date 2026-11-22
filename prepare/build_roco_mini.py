"""Build ROCO-mini v1, split by article.

Four scan types (CT, X-ray, MRI, ultrasound), images from the valid and test zips only, shorter side at least 160 px.
15% of the articles form the test side: 100 test images per scan type. From the other articles: 150 checker images
per scan type and 1,000 generator training images per scan type, in nested tiers of 500, 1,000, 2,000 and 4,000.
Writes data/roco_mini_v1.csv and data/roco_mini_v1_test_articles.txt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rocolib as rl  # noqa: E402
from paths import DATA, DERIVED  # noqa: E402

df = pd.read_pickle(DERIVED / "rocov2_master.pkl")
flag_cols = [c for c in df.columns if c.startswith("flag_")]
avail = df[df.split_official.isin(["valid", "test"]) & df.modality.notna()].copy()


def probe(image_id):
    im = rl.open_image(image_id)
    w, h, mode = im.size[0], im.size[1], im.mode
    im.draft("RGB", (max(32, w // 8), max(32, h // 8)))          # JPEG can decode at 1/8 size: fast
    a = np.asarray(im.convert("RGB"), np.int16)
    spread = a.max(-1) - a.min(-1)                               # 0 for a grey pixel, large for a coloured one
    return w, h, mode, float(spread.mean()), float((spread > 60).mean())


cols = ["width", "height", "mode", "chan_spread", "colour_px"]
avail[cols] = pd.DataFrame([probe(i) for i in avail.ID], index=avail.index, columns=cols)
avail["min_side"] = avail[["width", "height"]].min(1)
avail["true_colour"] = avail.colour_px >= 0.05
avail["marks"] = ~avail.true_colour & (avail.colour_px > 0.0005)

MINI_MODS = ["CT", "X-ray", "MRI", "Ultrasound"]
TIERS = [500, 1000, 2000, 4000]
pool = avail[avail.split_official.isin(["valid", "test"]) & avail.modality.isin(MINI_MODS)
             & (avail.min_side >= 160)].copy()
arts = np.array(sorted(pool.PMCID.unique()))
rng = np.random.default_rng(2026)
test_arts = set(rng.choice(arts, int(0.15 * len(arts)), replace=False))
pool["side"] = np.where(pool.PMCID.isin(test_arts), "test", "train")
roles = []
for m in MINI_MODS:
    t_ = pool[(pool.side == "test") & (pool.modality == m)].sample(100, random_state=1)
    roles.append(t_.assign(role="test", tier=np.nan))
    tr = pool[(pool.side == "train") & (pool.modality == m)].sample(frac=1, random_state=2)
    roles.append(tr.iloc[:150].assign(role="checker", tier=np.nan))
    gen = tr.iloc[150:150 + TIERS[-1] // 4].copy()
    gen["tier"] = [next(t for t in TIERS if k < t // 4) for k in range(len(gen))]   # smallest tier containing it
    roles.append(gen.assign(role="gen_train"))
mini = pd.concat(roles, ignore_index=True)
assert mini.ID.is_unique
assert not set(mini[mini.role == "test"].PMCID) & set(mini[mini.role != "test"].PMCID)

keep = ["ID", "PMCID", "split_official", "modality", "region", "view", "Caption", "cuis_auto", "n_tokens",
        "width", "height", "marks", "true_colour", "role", "tier"] + flag_cols
mini[keep].to_csv(DATA / "roco_mini_v1.csv", index=False)
(DATA / "roco_mini_v1_test_articles.txt").write_text("\n".join(sorted(test_arts)))
n_new = rl.cache_images(mini.ID)
print(mini.groupby(["role", "modality"]).size().unstack()[MINI_MODS])
print(f"saved {DATA / 'roco_mini_v1.csv'}; copied {n_new} images to {rl.IMG_CACHE}")
