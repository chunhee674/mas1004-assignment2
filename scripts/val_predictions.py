"""Every validation photo's answer from one model, and where its wrong answers go.

    uv run python scripts/val_predictions.py submission/model.onnx --size 448 --name b2

Writes results/curate/predictions_<name>.csv (path, city, answer, probability
of the answer) and prints, for each city, the accuracy, the share of wrong
photos whose answer is in the same country, and which cities they were called.
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

COUNTRY = {"seoul": "korea", "busan": "korea", "tokyo": "japan", "osaka": "japan", "taipei": "taiwan"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    rows = rules.list_photos(ROOT / "data" / "starter" / "val")
    probs = predict(args.onnx, [p for p, _ in rows], size=args.size)
    answers = probs.argmax(1)
    with open(OUT / f"predictions_{args.name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "city", "answer", "probability"])
        for (p, c), a, q in zip(rows, answers, probs):
            w.writerow([f"{c}/{Path(p).name}", c, CITIES[a], round(float(q[a]), 4)])

    print(f"\n{'city':8} {'photos':>6} {'right':>6} {'wrong':>6} {'same country':>13}  wrong photos called")
    wrong_all, same_all = 0, 0
    for i, city in enumerate(CITIES):
        ks = [k for k, (_, c) in enumerate(rows) if c == city]
        wrong = [k for k in ks if answers[k] != i]
        same = [k for k in wrong if COUNTRY[CITIES[answers[k]]] == COUNTRY[city]]
        wrong_all += len(wrong)
        same_all += len(same)
        called = ", ".join(f"{c} {sum(answers[k] == j for k in wrong)}" for j, c in enumerate(CITIES) if c != city)
        print(f"{city:8} {len(ks):6d} {1 - len(wrong) / len(ks):6.3f} {len(wrong):6d} {len(same):6d} ({100 * len(same) / max(1, len(wrong)):3.0f}%)  {called}")
    print(f"all wrong photos: {wrong_all}, same country {same_all} ({100 * same_all / wrong_all:.0f}%)")


if __name__ == "__main__":
    main()
