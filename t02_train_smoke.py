"""T02: run MINIM's train/train_model.py with a tiny test model and check whether the `modal` column is used.

A: the example CSV. B: `modal` scrambled. C: every caption changed. A2: A again, into the same output folder.
If B ends with the same weights as A, the column is never read. C is the control.
"""
import csv
import hashlib
import random
import shutil
import subprocess
import sys
import time

from diffusers import UNet2DConditionModel
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from common import BASE, EXAMPLE, RESULTS, TINY_SD, read_example_csv, save

tiny = snapshot_download(TINY_SD)
work = RESULTS / "t02_work"
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
_, rows = read_example_csv()


def write_csv(name, rows):
    p = work / name
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["path", "Text", "modal"])
        for r in rows:
            w.writerow([str(EXAMPLE / r["path"]), r["Text"], r["modal"]])
    return p


csv_a = write_csv("A_original.csv", rows)
rng = random.Random(1)
csv_b = write_csv("B_modal_scrambled.csv",
                  [dict(r, modal=rng.choice(["XXX", "Banana", "ChestCT", "OCT"])) for r in rows])
csv_c = write_csv("C_all_captions_changed.csv", [dict(r, Text="a photo of " + r["Text"]) for r in rows])


def run(csv_path, out):
    # 15 steps x batch 4 = one pass over the 60 examples
    cmd = [sys.executable, str(BASE / "train" / "train_model.py"),
           f"--pretrained_model_name_or_path={tiny}",
           f"--train_data_dir={csv_path}",
           "--resolution=64", "--center_crop", "--random_flip",
           "--train_batch_size=4", "--max_train_steps=15",
           "--learning_rate=1e-4", "--lr_warmup_steps=0",
           "--seed=0", "--checkpointing_steps=1000",
           f"--output_dir={out.name}"]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work))
    secs = round(time.time() - t, 1)
    unet = out / "unet" / "diffusion_pytorch_model.safetensors"
    digest = hashlib.sha256(unet.read_bytes()).hexdigest()[:16] if unet.exists() else None
    tail = (p.stderr or p.stdout).strip().splitlines()[-1:] if p.returncode else []
    return {"returncode": p.returncode, "seconds": secs, "unet_sha256": digest, "error_tail": tail}


def weights(out):
    return load_file(str(out / "unet" / "diffusion_pytorch_model.safetensors"))


def max_diff(a, b):
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


res = {}
res["A_original"] = run(csv_a, work / "out_A")
res["B_modal_scrambled"] = run(csv_b, work / "out_B")
res["C_all_captions_changed"] = run(csv_c, work / "out_C")
wa, wb, wc = weights(work / "out_A"), weights(work / "out_B"), weights(work / "out_C")
w0 = UNet2DConditionModel.from_pretrained(tiny, subfolder="unet").state_dict()
res["max_weight_change_A_vs_start"] = max_diff(wa, w0)
res["max_weight_diff_A_vs_B_modal"] = max_diff(wa, wb)
res["max_weight_diff_A_vs_C_captions"] = max_diff(wa, wc)
res["A2_rerun_same_folder"] = run(csv_a, work / "out_A")
res["modal_column_has_effect"] = res["max_weight_diff_A_vs_B_modal"] > 0
res["caption_has_effect"] = res["max_weight_diff_A_vs_C_captions"] > 0
save("t02_train_smoke", res)
for k, v in res.items():
    print(k, ":", v)
