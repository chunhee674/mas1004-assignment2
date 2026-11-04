"""Training-photo accuracy of models, and sheets of the training photos they get wrong.

    uv run python scripts/train_mistakes.py balanced_seed0 balanced_clean_seed0 balanced_clean_mixup_seed0

Reads results/curate/suspects_<tag>.csv (curate.py suspects writes it, one row
per training photo with the model's answer). Only photos in the current
track_a/selection.csv are counted, so every model is measured on the same photos.
For each city it prints the accuracy and which cities the wrong photos were
called. For the last tag it draws sheets of the wrong photos of each city, the
most confidently wrong first (lowest probability of their own city):
results/curate/train_mistakes_<tag>_<city>_NN.
"""

import collections
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curate import OUT, TRAIN, draw_sheets, read_selection  # noqa: E402
from rules import CITIES  # noqa: E402

SHEETS_PER_CITY = 1


def read(tag, selection):
    with open(OUT / f"suspects_{tag}.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["path"] in selection]


def main():
    selection = set(read_selection())
    for tag in sys.argv[1:]:
        rows = read(tag, selection)
        print(f"\n{tag}: {len(rows)} training photos")
        print(f"{'city':8} {'photos':>6} {'right':>6}  wrong photos called")
        accs = []
        for city in CITIES:
            mine = [r for r in rows if r["city"] == city]
            wrong = collections.Counter(r["answer"] for r in mine if r["answer"] != city)
            acc = 1 - sum(wrong.values()) / len(mine)
            accs.append(acc)
            print(f"{city:8} {len(mine):6d} {acc:6.3f}  " + ", ".join(f"{c} {wrong[c]}" for c in CITIES if c != city))
        print(f"{'mean':8} {'':6} {sum(accs) / 5:6.3f}")

    tag = sys.argv[-1]
    rows = read(tag, selection)
    for city in CITIES:
        wrong = sorted((r for r in rows if r["city"] == city and r["answer"] != city),
                       key=lambda r: float(r["own_city_probability"]))
        items = [(TRAIN / r["path"], f"{float(r['own_city_probability']):.2f}, says {r['answer']}") for r in wrong]
        names = draw_sheets(f"train_mistakes_{tag}_{city}", items[:48 * SHEETS_PER_CITY])
        print(f"{city}: {len(wrong)} wrong training photos, the first {len(items[:48 * SHEETS_PER_CITY])} on {', '.join(names)}")


if __name__ == "__main__":
    main()
