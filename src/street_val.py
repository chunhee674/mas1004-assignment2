"""Validation score of trained models on the street photos only (youtube, osv5m). For analysis only.

    python src/street_val.py balanced_clean clean_street2 clean_street3

Takes results/<tag>_seed0/model.onnx. The score is computed as the official one:
the accuracy of each city, then their mean. Also prints the same for all
validation photos, so that it can be checked against experiments.csv, and for
each kind of street photo. Writes results/curate/street_val.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402
from curate import OUT, predict  # noqa: E402
from rules import CITIES  # noqa: E402

STREET = ("youtube", "osv5m")


def source(path):
    return Path(path).name.split("_")[0]


def scores(rows, answers):
    """Accuracy per city and their mean, for the rows given as (index, city)."""
    per_city = []
    for i, city in enumerate(CITIES):
        mine = [k for k, c in rows if c == city]
        per_city.append(sum(answers[k] == i for k in mine) / max(1, len(mine)))
    return sum(per_city) / len(CITIES), per_city


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--every", type=int, default=1, help="use every n-th photo only")
    args = ap.parse_args()

    photos = rules.list_photos(ROOT / "data" / "starter" / "val")[::args.every]
    groups = {
        "all": [(k, c) for k, (p, c) in enumerate(photos)],
        "street": [(k, c) for k, (p, c) in enumerate(photos) if source(p) in STREET],
    }
    for s in STREET:
        groups[s] = [(k, c) for k, (p, c) in enumerate(photos) if source(p) == s]

    print(f"{'photos':10} " + " ".join(f"{c:>6}" for c in CITIES))
    for name, rows in groups.items():
        print(f"{name:10} " + " ".join(f"{sum(c == city for _, c in rows):6}" for city in CITIES))

    out = []
    print(f"\n{'tag':18} {'photos':8} {'score':>6} " + " ".join(f"{c:>6}" for c in CITIES))
    for tag in args.tags:
        answers = predict(ROOT / "results" / f"{tag}_seed0" / "model.onnx", [p for p, _ in photos]).argmax(1)
        for name, rows in groups.items():
            score, per_city = scores(rows, answers)
            print(f"{tag:18} {name:8} {score:6.3f} " + " ".join(f"{a:6.3f}" for a in per_city), flush=True)
            out.append([tag, name, round(score, 4)] + [round(a, 4) for a in per_city])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "street_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "photos", "score"] + list(CITIES))
        w.writerows(out)


if __name__ == "__main__":
    main()
