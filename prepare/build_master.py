"""Join the ROCOv2 tables into one table with one row per image: caption, concepts, article, scan type, body region,
view, caption length in CLIP tokens and simple caption flags. Writes data/rocov2/derived/rocov2_master.pkl.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from paths import DERIVED, ROCO  # noqa: E402

TABLES = ROCO / "tables"
DERIVED.mkdir(parents=True, exist_ok=True)

parts = []
for split in ["train", "valid", "test"]:
    cap = pd.read_csv(TABLES / f"{split}_captions.csv")
    auto = pd.read_csv(TABLES / f"{split}_concepts.csv").rename(columns={"CUIs": "cuis_auto"})
    man = pd.read_csv(TABLES / f"{split}_concepts_manual.csv").rename(columns={"CUIs": "cuis_manual"})
    d = cap.merge(auto, on="ID", how="left").merge(man, on="ID", how="left")
    d["split_official"] = split
    parts.append(d)
df = pd.concat(parts, ignore_index=True)
lic = pd.read_csv(TABLES / "license_information.csv")
df = df.merge(lic[["ID", "PMCID", "Attribution"]], on="ID", how="left")
df["cuis_manual"] = df.cuis_manual.fillna("")
df["cuis_auto"] = df.cuis_auto.fillna("")

# scan type, body region and view come from the manual concepts
MOD_CUI = {"C0040405": "CT", "C1306645": "X-ray", "C0024485": "MRI", "C0041618": "Ultrasound",
           "C0002978": "Angiogram", "C0032743": "PET"}
REGION_CUI = {"C0817096": "Chest", "C0037303": "Head", "C0000726": "Abdomen", "C0030797": "Pelvis",
              "C0037949": "Spine", "C1140618": "Upper limb", "C0023216": "Lower limb", "C0006141": "Breast"}
VIEW_CUI = {"C1999039": "AP", "C1996865": "PA", "C0205129": "Sagittal", "C0205106": "Transverse"}


def labels(s, table):
    return [table[c] for c in s.split(";") if c in table]


df["mods"] = df.cuis_manual.map(lambda s: labels(s, MOD_CUI))
df["regions"] = df.cuis_manual.map(lambda s: labels(s, REGION_CUI))
df["views"] = df.cuis_manual.map(lambda s: labels(s, VIEW_CUI))
df["modality"] = df.mods.map(lambda m: m[0] if m else None)
df["region"] = df.regions.map(lambda r: r[0] if len(r) == 1 else ("several" if r else "none"))
df["view"] = df.views.map(lambda v: v[0] if len(v) == 1 else ("several" if v else "none"))

# caption length, with the CLIP tokenizer that Stable Diffusion 1.5 uses (77 tokens at most)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
df["n_words"] = df.Caption.str.split().map(len)
df["n_tokens"] = [len(x) for x in tok(df.Caption.tolist(), add_special_tokens=True)["input_ids"]]

# rough word searches on the captions
FLAGS = {
    "Pointers": r"\barrow(?:s|head|heads)?\b|\basterisks?\b|\bstars?\b|\bcircles?\b|\bcircled\b",
    "Panel refs": r"\(\s*[a-hA-H]\s*(?:[,\-\u2013]\s*[a-hA-H]\s*)?\)|\bpanels?\b|\b(?:left|right|upper|lower|top|bottom) "
                  r"(?:image|panel|figure)\b|^[A-H][\.\)]\s",
    "Patient story": r"\b\d{1,3}[- ](?:year|yr|month|day)s?[- ]old\b|\bpatient\b|\b(?:man|woman|male|female|boy|girl)\b"
                     r"|\bpresent(?:ed|ing)\b|\bhistory\b",
    "Time": r"\bafter\b|\bbefore\b|\bfollow[- ]?up\b|\bpost[- ]?operative(?:ly)?\b|\bpre[- ]?operative\b"
            r"|\b\d+\s*(?:days?|weeks?|months?|years?)\b",
    "Size": r"\b\d+(?:\.\d+)?\s?(?:mm|cm)\b",
    "Side": r"\b(?:left|right|bilateral)\b",
}
for k, rx in FLAGS.items():
    df["flag_" + k] = df.Caption.str.contains(rx, flags=re.I, regex=True)
df["n_flags"] = df[["flag_" + k for k in FLAGS]].sum(1)

# finding and anatomy concepts (everything except scan type, region and view)
manual_set = set(MOD_CUI) | set(REGION_CUI) | set(VIEW_CUI)
df["n_find"] = df.cuis_auto.map(lambda s: sum(c not in manual_set for c in s.split(";") if c))

# how many images share the same caption
df["cap_norm"] = df.Caption.str.lower().str.replace(r"\s+", " ", regex=True).str.strip(" .")
g = df.groupby("cap_norm").agg(n=("ID", "size"), n_art=("PMCID", "nunique"))
df["cap_repeat"] = df.cap_norm.map(g.n)

df["licence"] = df.Attribution.str.extract(r"^(CC0|CC BY(?:-(?:NC|ND|SA))*)")[0].fillna("other")
df.to_pickle(DERIVED / "rocov2_master.pkl")
print(f"saved {DERIVED / 'rocov2_master.pkl'}: {len(df):,} images from {df.PMCID.nunique():,} articles")
