"""Sheets of the photos in data/mine, per city, to look through before training.

    uv run python scripts/mine_sheets.py

Writes results/curate/mine_<city>_NN.png, 48 photos each, in video order.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curate import draw_sheets  # noqa: E402
from rules import CITIES  # noqa: E402


def main():
    for city in CITIES:
        photos = sorted((ROOT / "data" / "mine" / city).glob("*.jpg"))
        if not photos:
            continue
        # the caption is the video and the frame number
        names = draw_sheets(f"mine_{city}", [(p, p.stem[len("ytmine_"):]) for p in photos])
        print(f"{city:7} {len(photos):4} photos on {len(names)} sheets: {names[0]} to {names[-1]}")


if __name__ == "__main__":
    main()
