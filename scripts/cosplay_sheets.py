"""Sheets of the training photos whose title in ATTRIBUTION.csv mentions cosplay.

    uv run python scripts/cosplay_sheets.py

Writes results/curate/cosplay_NN.png and .csv (48 per sheet), so that
`curate.py drop cosplay_00 5 17 --reason ...` takes the numbers on them.
The student looks at every sheet and names the photos to drop.
"""

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curate import TRAIN, draw_sheets  # noqa: E402

WORDS = re.compile(r"cosplay|コスプレ|角色扮演|comic ?world|fancy frontier|開拓動漫", re.I)

rows = []
with open(ROOT / "data" / "starter" / "ATTRIBUTION.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["split"] == "train" and WORDS.search(r["title"]):
            rows.append(r)
rows.sort(key=lambda r: (r["city"], r["file"]))
items = [(TRAIN / r["city"] / Path(r["file"]).name, f"{r['city']} {r['title']}") for r in rows]
names = draw_sheets("cosplay", items)
print(f"{len(items)} training photos with cosplay in the title, on {', '.join(names)}")
