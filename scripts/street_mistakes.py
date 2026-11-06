"""Sheets of wrong answers on the YouTube validation frames, from street_memory.py's output.

    uv run python scripts/street_mistakes.py clean_street2_video busan osaka

Draws at most 48 wrong frames per city, at random (seed 0), to
results/curate/street_mistakes_<tag>_<city>_00. The caption is the video and the answer.
"""

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curate import OUT, draw_sheets  # noqa: E402


def main():
    tag, cities = sys.argv[1], sys.argv[2:]
    with open(OUT / f"street_memory_{tag}.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    rng = random.Random(0)
    for city in cities:
        mine = [r for r in rows if r["city"] == city]
        wrong = [r for r in mine if r["answer"] != city]
        pick = rng.sample(wrong, min(48, len(wrong)))
        items = [(ROOT / "data" / "starter" / "val" / r["path"],
                  f"{Path(r['path']).stem[8:14]} >{r['answer']}") for r in pick]
        names = draw_sheets(f"street_mistakes_{tag}_{city}", items)
        print(f"{city}: {len(wrong)} wrong of {len(mine)}, on {', '.join(names)}")


if __name__ == "__main__":
    main()
