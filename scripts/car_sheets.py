"""Sheets of the training photos still in the selection whose title mentions cars or a motor show.

    uv run python scripts/car_sheets.py

Writes results/curate/cand_car_NN.png and .csv (48 per sheet), so that
`curate.py drop cand_car_00 5 17 --reason ...` takes the numbers on them.
The student looks at every sheet and names the photos to drop.
"""

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curate import TRAIN, draw_sheets, read_selection  # noqa: E402

WORDS = re.compile(r"motor ?show|auto ?(show|salon)|모터쇼|자동차|モーターショー|車展|\bkia\b|hyundai|genesis"
                   r"|toyota|nissan|honda|lexus|\bcars?\b|\bconcept\b|\bev\d|ioniq", re.I)

selection = set(read_selection())
rows = []
with open(ROOT / "data" / "starter" / "ATTRIBUTION.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        path = f"{r['city']}/{Path(r['file']).name}"
        if r["split"] == "train" and path in selection and WORDS.search(r["title"]):
            rows.append((path, r["title"]))
rows.sort()
names = draw_sheets("cand_car", [(TRAIN / p, f"{p.split('/')[0]} {t}") for p, t in rows])
print(f"{len(rows)} training photos with a car word in the title, on {', '.join(names)}")
