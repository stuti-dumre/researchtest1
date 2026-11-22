import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("MINIM_DIR", HERE / "MINIM"))
RLHF = BASE / "MINIM_RLHF"
EXAMPLE = BASE / "example"
RESULTS = Path(os.environ.get("RESULTS_DIR", HERE / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)

TINY_SD = "hf-internal-testing/tiny-stable-diffusion-torch"
CLIP_TOK = "openai/clip-vit-large-patch14"


def save(name, obj):
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"[saved] {path.relative_to(HERE) if path.is_relative_to(HERE) else path}")


def rlhf_config():
    import yaml
    return yaml.safe_load((RLHF / "config" / "config.yaml").read_text())


def import_rlhf():
    # MINIM_RLHF imports its own modules as src.*
    if str(RLHF) not in sys.path:
        sys.path.insert(0, str(RLHF))


def read_example_csv():
    # some captions contain unquoted commas, so the middle fields are joined back together
    rows = list(csv.reader(open(EXAMPLE / "example_data.csv", encoding="utf-8", newline="")))
    header, out = rows[0], []
    for r in rows[1:]:
        out.append({"path": r[0], "Text": ",".join(r[1:-1]), "modal": r[-1]})
    return header, out
