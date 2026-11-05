"""What script a validation photo shows, against one model's answers.

    uv run python scripts/ocr_table.py --name b2

Joins results/curate/ocr_val.csv with results/curate/predictions_<name>.csv and
prints, per kind of script found, the photos, the accuracy, the country-level
accuracy and the answers. Each photo has one kind. Then the same per true city.
Draws 48 random photos of each kind (seed 0) on sheets
results/curate/ocr_check_<kind>_00, to check by eye how well the OCR read them.
"""

import argparse
import collections
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curate import OUT, draw_sheets  # noqa: E402
from rules import CITIES  # noqa: E402

COUNTRY = {"seoul": "korea", "busan": "korea", "tokyo": "japan", "osaka": "japan", "taipei": "taiwan"}
KINDS = ["hangul", "kana", "hangul+kana", "han_only", "latin_only", "none"]
VAL = ROOT / "data" / "starter" / "val"


def kind(r, least=1):
    hangul, kana, han, latin = (int(r[k]) >= least for k in ("hangul", "kana", "han", "latin"))
    if hangul and kana:
        return "hangul+kana"
    if hangul:
        return "hangul"
    if kana:
        return "kana"
    if han:
        return "han_only"
    return "latin_only" if latin else "none"


def wrong_script(r):
    """Hangul read outside Korea or kana read outside Japan: taken for a misreading, the student's rule."""
    return (int(r["hangul"]) > 0 and COUNTRY[r["city"]] != "korea") or (int(r["kana"]) > 0 and COUNTRY[r["city"]] != "japan")


def table(rows):
    print(f"{'kind':12} {'photos':>6} {'acc':>6} {'country':>7}   answers " + " ".join(f"{c:>6}" for c in CITIES))
    for k in KINDS:
        group = [r for r in rows if r["kind"] == k]
        if not group:
            continue
        acc = sum(r["answer"] == r["city"] for r in group) / len(group)
        same = sum(COUNTRY[r["answer"]] == COUNTRY[r["city"]] for r in group) / len(group)
        answers = collections.Counter(r["answer"] for r in group)
        print(f"{k:12} {len(group):6} {acc:6.3f} {same:7.3f}           "
              + " ".join(f"{answers[c] / len(group):6.3f}" for c in CITIES))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--least", type=int, default=1, help="text boxes of a script needed to count it")
    ap.add_argument("--clean", action="store_true", help="leave out the photos wrong_script finds")
    args = ap.parse_args()
    with open(OUT / f"predictions_{args.name}.csv", encoding="utf-8") as f:
        answers = {r["path"]: r["answer"] for r in csv.DictReader(f)}
    with open(OUT / "ocr_val.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.clean:
        kept = [r for r in rows if not wrong_script(r)]
        print(f"left out {len(rows) - len(kept)} photos, by true city: "
              + ", ".join(f"{c} {sum(r['city'] == c for r in rows) - sum(r['city'] == c for r in kept)}" for c in CITIES))
        rows = kept
    for r in rows:
        r["kind"], r["answer"] = kind(r, args.least), answers[r["path"]]
    print(f"all cities, {len(rows)} photos")
    table(rows)
    for city in CITIES:
        print(f"\n{city}")
        table([r for r in rows if r["city"] == city])
    if args.least > 1:
        return
    rng = random.Random(0)
    for k in KINDS:
        group = [r for r in rows if r["kind"] == k]
        pick = rng.sample(group, min(48, len(group)))
        draw_sheets(f"ocr_check_{'clean_' if args.clean else ''}{k.replace('+', '_')}",
                    [(VAL / r["path"], f"{r['city']}>{r['answer']}") for r in pick])


if __name__ == "__main__":
    main()
