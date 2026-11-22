"""Folders used by the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ROCO = DATA / "rocov2"            # tables/ and zips/ from Zenodo
DERIVED = ROCO / "derived"        # joined table and judge 1 features
FEATURES = DATA / "features"      # feature caches of tests 2 to 4
MODELS = ROOT / "models"          # the two judges and the LoRA models
