"""A look at the test 4 results after the verdicts: accuracy per scan type, the items where the pick and random models
differ, and a contact sheet of test images."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402,F401
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs"
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import splits as sp  # noqa: E402
import rocolib as rl  # noqa: E402

test = sp.test_items()
J = {a: pd.read_csv(RUN / f"judged_{a}.csv").set_index("ID") for a in ("base", "random", "pick")}
tab = pd.DataFrame({a: (J[a].seen_j2 == J[a].modality).groupby(J[a].modality).mean() for a in J}).round(3)
print("scan type accuracy by judge 2, per scan type:\n", tab)
conf = {a: pd.crosstab(J[a].modality, J[a].seen_j2) for a in J}
for a in J:
    print(f"\n{a}: what judge 2 sees (rows = asked for)\n", conf[a])
ok_p = J["pick"].seen_j2 == J["pick"].modality
ok_r = J["random"].seen_j2 == J["random"].modality
print("\nitems right for pick only:", int((ok_p & ~ok_r).sum()), "| right for random only:", int((~ok_p & ok_r).sum()))
print("marks share per scan type:\n", pd.DataFrame({a: J[a].marks.groupby(J[a].modality).mean() for a in J}).round(3))

ids = list(test.groupby("modality", group_keys=False).head(3).ID) + list((ok_p & ~ok_r)[lambda s: s].index[:3])
fig, ax = plt.subplots(len(ids), 4, figsize=(8, 2.05 * len(ids)))
for r, i in enumerate(ids):
    row = test.set_index("ID").loc[i]
    ims = [rl.to_square(rl.open_image(i), 256)] + [Image.open(RUN / "test" / a / f"{i}.png") for a in J]
    labels = ["real"] + [f"{a}: {J[a].seen_j2[i]}" for a in J]
    for c, (im, lab) in enumerate(zip(ims, labels)):
        ax[r, c].imshow(im)
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_title(lab if c else f"real ({row.modality})", fontsize=7)
    ax[r, 0].set_ylabel(str(row.Caption)[:40] + "...", fontsize=5.5)
fig.tight_layout()
fig.savefig(RUN / "contact_sheet.png", dpi=110)
print("saved", RUN / "contact_sheet.png")
