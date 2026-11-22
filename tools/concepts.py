"""Concept labels with the negation rule of test 3."""
import re
from collections import Counter

import pandas as pd

from paths import ROCO, ROOT

_names = pd.read_csv(ROCO / "tables" / "cui_mapping.csv")
NAME = dict(zip(_names.iloc[:, 0], _names.iloc[:, 1]))
DROP = {"structure", "disorder", "entire", "finding", "device", "procedure", "region", "part", "body", "system", "of",
        "the", "and", "nos"}
FILL = {"evidence", "sign", "signs", "any", "of", "obvious", "significant", "definite", "for", "the", "a", "an"}
TRIG = {"no", "not", "without", "absent", "absence", "negative", "free", "nor", "resolved", "resolution", "excluded"}


def _head(cui):
    words = [w for w in re.findall(r"[a-z]+", re.sub(r"\(.*?\)", " ", NAME.get(cui, "").lower()))
             if w not in DROP and len(w) >= 4]
    return words[-1] if words else None


HEAD = {c: _head(c) for c in NAME}


def negated_words(caption):
    out = set()
    for sent in re.split(r"[.;:!?]", str(caption).lower()):
        toks = re.findall(r"[a-z]+", sent)
        for k, t in enumerate(toks):
            if t in TRIG or (t == "ruled" and k + 1 < len(toks) and toks[k + 1] == "out"):
                out.update([w for w in toks[k + 1:k + 12] if w not in FILL and w != "out"][:3])
    return out


def codes(row, negation=True):
    """Concept codes of one master-table row (cuis_auto and cuis_manual), without negated mentions if negation=True."""
    cs = {c for col in ("cuis_auto", "cuis_manual") if isinstance(row[col], str) for c in row[col].split(";") if c}
    if negation:
        words = negated_words(row["Caption"])
        cs = {c for c in cs if not (HEAD.get(c) and any(w.startswith(HEAD[c]) for w in words))}
    return cs


def manual_codes(df):
    return set(Counter(c for s in df.cuis_manual.dropna() for c in s.split(";") if c))


def usable_other():
    """The 39 usable 'other' concepts of test 3 (main version)."""
    r = pd.read_csv(ROOT / "experiments" / "test3_concepts" / "runs" / "concepts_v2.csv")
    m = r[r.version.str.startswith("main") & r.usable & r.category.eq("other")]
    return sorted(m.CUI)
