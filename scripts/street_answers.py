"""Where a model's answers go on street and other validation photos, per true city.

    uv run python scripts/street_answers.py --name b2

Takes results/curate/predictions_<name>.csv. Street photos are youtube and osv5m.
"""

import argparse
import csv
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
    with open(OUT / f"predictions_{args.name}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = {
        "youtube": lambda s: s == "youtube",
        "osv5m": lambda s: s == "osv5m",
        "other": lambda s: s not in ("youtube", "osv5m"),
    }
    for name, keep in groups.items():
        print(f"\n{name:8} {'photos':>6}  answers " + " ".join(f"{c:>6}" for c in CITIES))
        for city in CITIES:
            mine = [r for r in rows if r["city"] == city and keep(Path(r["path"]).name.split("_")[0])]
            print(f"{city:8} {len(mine):6}          "
                  + " ".join(f"{sum(r['answer'] == c for r in mine) / max(1, len(mine)):6.3f}" for c in CITIES))
        mine = [r for r in rows if keep(Path(r["path"]).name.split("_")[0])]
        print(f"{'answered':8} {len(mine):6}          "
              + " ".join(f"{sum(r['answer'] == c for r in mine) / len(mine):6.3f}" for c in CITIES))


if __name__ == "__main__":
    main()
