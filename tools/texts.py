"""The text the model gets for one image.

  minim_code   MINIM's released code: the caption as it is.
  minim_paper  MINIM's paper: "scan type: caption". All four tests use this one.
"""
ARMS = ["minim_code", "minim_paper"]


def make_text(row, arm, rng=None, for_eval=False):
    """row: a dict-like with Caption and modality. Neither format uses rng or for_eval."""
    cap = str(row["Caption"]).strip()
    if arm == "minim_code":
        return cap
    if arm == "minim_paper":
        return f"{row['modality']}: {cap}"
    raise ValueError(arm)
