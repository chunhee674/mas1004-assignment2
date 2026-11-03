"""How often the right city is the model's first or second choice, on the validation photos.

    python src/top2.py results/baseline_seed0/model.onnx results/balanced_seed0/model.onnx
    python src/top2.py results/run_seed0/model.onnx --every 50

For each model and city it prints the share of photos whose city is the
model's first choice (the accuracy in experiments.csv), the share whose city
is its first or second choice, and the share where the model's first choice
is in another country. Photos are prepared as the grader prepares them
(curate.predict).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402
from curate import predict  # noqa: E402
from rules import CITIES  # noqa: E402

COUNTRY = {"seoul": "korea", "busan": "korea", "tokyo": "japan", "osaka": "japan", "taipei": "taiwan"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--photos", default=str(ROOT / "data" / "starter" / "val"))
    ap.add_argument("--every", type=int, default=1, help="use every n-th photo only")
    args = ap.parse_args()

    rows = rules.list_photos(args.photos)[::args.every]
    for onnx in args.onnx:
        probs = predict(onnx, [p for p, _ in rows])
        order = probs.argsort(1)[:, ::-1]
        print(f"\n{onnx}: {len(rows)} validation photos")
        print(f"{'city':8} {'photos':>6} {'top-1':>6} {'top-2':>6} {'other country':>14}")
        top1s, top2s = [], []
        for i, city in enumerate(CITIES):
            ks = [k for k, (_, c) in enumerate(rows) if c == city]
            top1 = sum(order[k, 0] == i for k in ks) / len(ks)
            top2 = sum(i in order[k, :2] for k in ks) / len(ks)
            other = sum(COUNTRY[CITIES[order[k, 0]]] != COUNTRY[city] for k in ks) / len(ks)
            top1s.append(top1)
            top2s.append(top2)
            print(f"{city:8} {len(ks):6d} {top1:6.3f} {top2:6.3f} {other:14.3f}")
        print(f"{'mean':8} {'':6} {sum(top1s) / 5:6.3f} {sum(top2s) / 5:6.3f}")


if __name__ == "__main__":
    main()
