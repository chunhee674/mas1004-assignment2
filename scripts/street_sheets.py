"""Sheets of a random sample of the street photos (youtube, osv5m) among the training photos.

    uv run python scripts/street_sheets.py

Uses the selection of balanced_clean (commit 0c8089a). The sample is drawn by
a fixed random seed, not chosen. Writes results/curate/street_<source>_<city>_00
(48 per sheet) and every street photo to results/curate/street_photos.csv.
"""

import csv
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curate import OUT, TRAIN, draw_sheets  # noqa: E402
from rules import CITIES  # noqa: E402

text = subprocess.run(["git", "show", "0c8089a:track_a/selection.csv"], cwd=ROOT, capture_output=True,
                      encoding="utf-8", check=True).stdout
paths = [r["path"] for r in csv.DictReader(text.splitlines())]


def source(p):
    return Path(p).name.split("_")[0]


street = [p for p in paths if source(p) in ("youtube", "osv5m")]
with open(OUT / "street_photos.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "source"])
    w.writerows((p, source(p)) for p in street)

rng = random.Random(0)
print(f"{'city':8} {'all':>6} {'youtube':>8} {'videos':>7} {'osv5m':>6} {'street %':>9}")
for city in CITIES:
    mine = [p for p in paths if p.split("/")[0] == city]
    yt = [p for p in street if p.split("/")[0] == city and source(p) == "youtube"]
    osv = [p for p in street if p.split("/")[0] == city and source(p) == "osv5m"]
    videos = {Path(p).stem[len("youtube_"):].rsplit("_", 1)[0] for p in yt}
    print(f"{city:8} {len(mine):6d} {len(yt):8d} {len(videos):7d} {len(osv):6d} {100 * (len(yt) + len(osv)) / len(mine):8.1f}%")
    for name, group in (("youtube", yt), ("osv5m", osv)):
        if group:
            sample = rng.sample(sorted(group), min(48, len(group)))
            draw_sheets(f"street_{name}_{city}", [(TRAIN / p, Path(p).stem[:24]) for p in sample])
