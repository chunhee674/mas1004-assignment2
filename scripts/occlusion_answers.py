"""Which answers change when text is covered, from results/curate/occlusion_<name>.csv.

    uv run python scripts/occlusion_answers.py --name b2

Prints the share of each answer under each version, and the photos where exactly
one of the two covers keeps the right answer (the pairs a McNemar test counts).
"""

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curate import OUT  # noqa: E402
from rules import CITIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    with open(OUT / f"occlusion_{args.name}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{'answer share':12} " + " ".join(f"{c:>6}" for c in CITIES))
    for k in ("plain", "text", "control"):
        print(f"{k:12} " + " ".join(f"{sum(r[k] == c for r in rows) / len(rows):6.3f}" for c in CITIES))
    print(f"\n{'':8} {'only ctrl right':>15} {'only text right':>15} {'z':>6}")
    for city in ["all"] + CITIES:
        group = [r for r in rows if city in ("all", r["city"])]
        a = sum(r["control"] == r["city"] != r["text"] for r in group)
        b = sum(r["text"] == r["city"] != r["control"] for r in group)
        print(f"{city:8} {a:15} {b:15} {(a - b) / math.sqrt(max(1, a + b)):6.1f}")


if __name__ == "__main__":
    main()
