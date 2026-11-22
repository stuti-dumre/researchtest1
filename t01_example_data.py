"""T01: load MINIM's example data the way train_model.py does, and describe its captions and images."""
import csv
import hashlib
import re
from collections import Counter

import numpy as np
from PIL import Image
from transformers import CLIPTokenizer

from common import CLIP_TOK, EXAMPLE, read_example_csv, save

res = {}

rows = list(csv.reader(open(EXAMPLE / "example_data.csv", encoding="utf-8", newline="")))
res["csv_rows"] = len(rows) - 1
res["csv_malformed_lines"] = [i + 1 for i, r in enumerate(rows) if len(r) != 3]
try:
    from datasets import load_dataset  # same call as train_model.py
    load_dataset("csv", data_files={"train": str(EXAMPLE / "example_data.csv")})
    res["hf_datasets_load"] = "ok"
except Exception as e:
    res["hf_datasets_load"] = f"FAILS: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
print("CSV:", res["hf_datasets_load"])

header, data = read_example_csv()
res["csv_header"] = header
tok = CLIPTokenizer.from_pretrained(CLIP_TOK)
lens = [len(tok(d["Text"]).input_ids) for d in data]  # includes the start and end tokens
res["caption_tokens"] = {"min": min(lens), "mean": round(float(np.mean(lens)), 1), "max": max(lens),
                         "over_77": sum(n > 77 for n in lens)}
by_mod = {}
for d in data:
    by_mod.setdefault(d["modal"], []).append(d["Text"])
res["captions_per_modality"] = {m: {"n": len(t), "unique": len(set(t))} for m, t in by_mod.items()}
side = re.compile(r"\b(left|right|bilateral|unilateral)\b", re.I)
res["captions_with_left_right"] = sum(bool(side.search(d["Text"])) for d in data)
MOD_WORDS = {"BreastMRI": ["mri", "breast"], "BrainMRI": ["mri", "brain"], "ChestCT": ["ct"],
             "CXR": ["x-ray", "xray", "radiograph", "chest"], "Fundus": ["fundus", "retina"], "OCT": ["oct", "retina"]}
res["modality_word_in_caption"] = {m: sum(any(w in t.lower() for w in MOD_WORDS.get(m, [m.lower()])) for t in ts)
                                   for m, ts in by_mod.items()}

img_info = []
for d in data:
    im = Image.open(EXAMPLE / d["path"])
    w, h = im.size
    # share of the image left after train.sh's Resize(512) and CenterCrop(512)
    s = 512 / min(w, h)
    kept = (512 * 512) / (w * s * h * s)
    img_info.append({"modal": d["modal"], "size": [w, h], "mode": im.mode, "crop_keeps_fraction": round(kept, 3)})
res["images"] = {
    "modes": dict(Counter(i["mode"] for i in img_info)),
    "sizes": dict(Counter(f"{i['size'][0]}x{i['size'][1]}" for i in img_info)),
    "min_crop_keeps": min(i["crop_keeps_fraction"] for i in img_info),
    "n_cropped_more_than_10pct": sum(i["crop_keeps_fraction"] < 0.9 for i in img_info),
}
by = {}
for i in img_info:
    by.setdefault(i["modal"], []).append(i)
res["images_per_modality"] = {m: {"sizes": sorted({f"{x['size'][0]}x{x['size'][1]}" for x in v}),
                                  "modes": sorted({x["mode"] for x in v}),
                                  "min_crop_keeps": min(x["crop_keeps_fraction"] for x in v)} for m, v in by.items()}
hashes = Counter(hashlib.md5((EXAMPLE / d["path"]).read_bytes()).hexdigest() for d in data)
res["duplicate_image_files"] = sum(c - 1 for c in hashes.values() if c > 1)

save("t01_example_data", res)
for k, v in res.items():
    print(k, ":", v)
