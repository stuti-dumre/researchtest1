"""T05: test MINIM's train/train_classifier.py, the small CNN used for the HER2 task.

Checks: running the file, loading the breast MRI examples, the saved "best" weights, softmax before
CrossEntropyLoss, patient overlap with an image-level split, layer counts vs the paper.
"""
import importlib.util
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import BASE, EXAMPLE, save

warnings.filterwarnings("ignore")
path = BASE / "train" / "train_classifier.py"
spec = importlib.util.spec_from_file_location("minim_cls", path)
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)
res = {}

p = subprocess.run([sys.executable, str(path), "--metadata", "does_not_exist.xlsx", "--class_num", "3"],
                   capture_output=True, text=True)
res["run_file"] = {"returncode": p.returncode, "stdout_chars": len(p.stdout), "stderr_chars": len(p.stderr),
                   "has_main_guard": "__main__" in path.read_text(encoding="utf-8")}
print("run file:", res["run_file"])

meta = pd.DataFrame({"image_path": [f"breast-mri/{i}.jpg" for i in range(10)]})
try:
    x = mc.read_images_from_df(meta, root_dir=str(EXAMPLE))
    res["load_breast_examples"] = f"ok, tensor {tuple(x.shape)}"
except Exception as e:
    res["load_breast_examples"] = f"CRASH: {type(e).__name__}: {str(e)[:150]}"
print("load examples:", res["load_breast_examples"])

# the same pattern as main(), lines 164 and 181
torch.manual_seed(0)
model = mc.Classifier(class_num=3)
best_model_wts = model.state_dict()
snapshot = {k: v.clone() for k, v in best_model_wts.items()}
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
xb = torch.rand(2, 3, 512, 512)
loss = nn.CrossEntropyLoss()(model(xb), torch.tensor([0, 1]))
loss.backward()
opt.step()  # a later training step
k0 = "cnn.0.weight"
res["state_dict_aliasing"] = {
    "saved_best_changed_after_later_training": not torch.equal(best_model_wts[k0], snapshot[k0]),
    "saved_best_equals_current_weights": torch.equal(best_model_wts[k0], model.state_dict()[k0]),
    "parameters": int(sum(p.numel() for p in model.parameters())),
    "fc1_parameters": int(model.fc[0].weight.numel() + model.fc[0].bias.numel()),
}
print("aliasing:", res["state_dict_aliasing"])

ce = nn.CrossEntropyLoss()
perfect = torch.tensor([[1.0, 0.0, 0.0]])  # a perfect, fully confident softmax output
res["double_softmax"] = {
    "loss_floor_3_classes": round(float(ce(perfect, torch.tensor([0]))), 4),
    "loss_floor_formula": "log(e^0 + e^0 + e^1) - 1 = log(2 + e) - 1",
}
g = torch.Generator().manual_seed(0)
X = torch.randn(600, 20, generator=g)
W = torch.randn(20, 3, generator=g)
y = (X @ W).argmax(1)
out = {}
for name, use_softmax in [("logits_correct", False), ("softmax_then_CE_MINIM", True)]:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))
    o = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(300):
        o.zero_grad()
        z = net(X)
        ce(F.softmax(z, 1) if use_softmax else z, y).backward()
        o.step()
    with torch.no_grad():
        z = net(X)
        out[name] = {"final_loss": round(float(ce(F.softmax(z, 1) if use_softmax else z, y)), 4),
                     "train_accuracy": round(float((z.argmax(1) == y).float().mean()), 3),
                     "first_layer_grad_norm": None}
res["double_softmax"]["toy_training_300_steps"] = out
print("double softmax:", res["double_softmax"])


def overlap(n_patients, imgs_per_patient, seed=0):
    # share of test images whose patient also has images in the training part
    pid = torch.arange(n_patients).repeat_interleave(imgs_per_patient)
    ds = torch.utils.data.TensorDataset(pid)
    tr, te = torch.utils.data.random_split(ds, [0.8, 0.2], generator=torch.Generator().manual_seed(seed))
    train_p = set(pid[tr.indices].tolist())
    return round(float(np.mean([t in train_p for t in pid[te.indices].tolist()])), 4)


res["image_level_split_overlap"] = {
    "chest_3745_patients_x_20_images": overlap(3745, 20),
    "10_images_per_patient": overlap(1000, 10),
    "3_images_per_patient": overlap(1000, 3),
    "meaning": "share of test images whose patient also has images in the training part",
}
print("split overlap:", res["image_level_split_overlap"])

res["architecture"] = {"conv_layers_in_code": sum(isinstance(m, nn.Conv2d) for m in model.modules()),
                       "linear_layers_in_code": sum(isinstance(m, nn.Linear) for m in model.modules()),
                       "paper_says": "three convolutional layers ... three linear functions",
                       "code_lr": 1e-5, "paper_lr": 1e-3, "code_epochs": 50, "paper_epochs": 20,
                       "code_batch": 32, "paper_batch": 128}
print("architecture:", res["architecture"])
save("t05_classifier", res)
