"""Candidate sheets for the Step 7 rules, made by fixed criteria, not by choosing photos.

    uv run python scripts/candidate_sheets.py

Only photos still in track_a/selection.csv are considered. The student looks
at every sheet and names the photos to drop with curate.py drop.

  rule 1 (close-ups): the title in ATTRIBUTION.csv has a word of TITLE_WORDS,
                      one set of sheets per kind (a photo goes to the first kind)
                      -> results/curate/cand_food_NN, cand_museum_NN, cand_map_NN, cand_people_NN
  rule 2 (360 panoramas): width / height between 1.95 and 2.05, the shape of
                      an equirectangular panorama
                      -> results/curate/cand_panorama_NN
  rule 3 (hard to see): the DARK_N darkest photos of each city (mean grey
                      level) and the BLUR_N least sharp (variance of the
                      Laplacian of the grey photo at 256 pixels)
                      -> results/curate/cand_dark_<city>_NN, cand_blur_<city>_NN

The numbers measured for every photo go to results/curate/photo_stats.csv.
"""

import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curate import OUT, TRAIN, draw_sheets, read_selection  # noqa: E402
from rules import CITIES  # noqa: E402

TITLE_WORDS = {
    "food": re.compile(r"\bfood\b|\bdish(es)?\b|\bmeal\b|cuisine|\bnoodles?\b|ramen|sushi|dessert|bento|banchan"
                       r"|음식|반찬|料理|食べ物|食物|小吃", re.I),
    "museum": re.compile(r"museum|exhibit|artifact|박물관|博物館", re.I),
    "map": re.compile(r"\bmaps?\b|지도|地図|地圖", re.I),
    "people": re.compile(r"press conference|portrait|selfie|기자회견|記者会見|記者會", re.I),
}
DARK_N, BLUR_N = 24, 24


def measure(rel):
    """(width, height, mean grey 0..255, sharpness) of one training photo."""
    with Image.open(TRAIN / rel) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        g = im.convert("L")
        g.thumbnail((256, 256))
        a = np.asarray(g, dtype=np.float32)
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    return w, h, float(a.mean()), float(lap.var())


def main():
    selection = read_selection()
    with ProcessPoolExecutor() as pool:
        stats = dict(zip(selection, pool.map(measure, selection, chunksize=64)))
    with open(OUT / "photo_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "width", "height", "mean_grey", "sharpness"])
        for p, (wd, ht, m, s) in stats.items():
            w.writerow([p, wd, ht, round(m, 1), round(s, 1)])

    titles = {}
    with open(ROOT / "data" / "starter" / "ATTRIBUTION.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                titles[f"{r['city']}/{Path(r['file']).name}"] = r["title"]

    taken = set()
    for kind, words in TITLE_WORDS.items():
        found = sorted(p for p in selection if p not in taken and words.search(titles.get(p, "")))
        taken.update(found)
        names = draw_sheets(f"cand_{kind}", [(TRAIN / p, f"{p.split('/')[0]} {titles[p]}") for p in found])
        print(f"rule 1, {kind}: {len(found)} photos with such a word in the title, on {', '.join(names) or '-'}")

    pano = sorted(p for p, (wd, ht, _, _) in stats.items() if 1.95 <= wd / ht <= 2.05)
    names = draw_sheets("cand_panorama", [(TRAIN / p, f"{p.split('/')[0]} {stats[p][0]}x{stats[p][1]}") for p in pano])
    print(f"rule 2: {len(pano)} photos shaped 2:1, on {', '.join(names) or '-'}")

    for city in CITIES:
        mine = [p for p in selection if p.split("/")[0] == city]
        dark = sorted(mine, key=lambda p: stats[p][2])[:DARK_N]
        blur = sorted(mine, key=lambda p: stats[p][3])[:BLUR_N]
        d = draw_sheets(f"cand_dark_{city}", [(TRAIN / p, f"grey {stats[p][2]:.0f}") for p in dark])
        b = draw_sheets(f"cand_blur_{city}", [(TRAIN / p, f"sharpness {stats[p][3]:.0f}") for p in blur])
        print(f"rule 3, {city}: darkest {DARK_N} on {d[0]}, least sharp {BLUR_N} on {b[0]}")


if __name__ == "__main__":
    main()
