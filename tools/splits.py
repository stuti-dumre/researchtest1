"""The four-way split by article from test 1: test items, reward side (judge 1's training images), judge side (judge
2's training images) and the generator pool (data/pool_v1.csv). Used by every later test.
"""
import pickle
import warnings

import pandas as pd

from paths import DATA, DERIVED, MODELS

DER = DERIVED
MODS = ["CT", "X-ray", "MRI", "Ultrasound"]


def master():
    return pd.read_pickle(DER / "rocov2_master.pkl")


def mini():
    return pd.read_csv(DATA / "roco_mini_v1.csv")


def judge1_ids(df=None, mi=None):
    """Judge 1's 4,800 training IDs in the row order of checker_train_feats.npy (the selection of
    prepare/build_judge1.py, seed 5)."""
    df = master() if df is None else df
    mi = mini() if mi is None else mi
    regions = pickle.load(open(MODELS / "checker.pkl", "rb"))["regions"]
    test_arts = set(mi[mi.role == "test"].PMCID)
    used = set(mi[mi.role.isin(["gen_train", "test"])].ID)
    cand = df[df.split_official.isin(["valid", "test"]) & df.modality.isin(MODS) & ~df.PMCID.isin(test_arts)
              & ~df.ID.isin(used)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pick = cand.groupby("modality", group_keys=False).apply(lambda g: g.sample(min(len(g), 900), random_state=5))
    extra = cand[(cand.modality == "X-ray") & cand.region.isin(regions) & ~cand.ID.isin(pick.ID)]
    return list(pd.concat([pick, extra.sample(min(len(extra), 1200), random_state=5)]).ID)


def judge2_ids():
    return list(pickle.load(open(MODELS / "judge2.pkl", "rb"))["train_ids"])


def test_items():
    m = mini()
    return m[m.role == "test"].reset_index(drop=True)


def pool():
    return pd.read_csv(DATA / "pool_v1.csv")
