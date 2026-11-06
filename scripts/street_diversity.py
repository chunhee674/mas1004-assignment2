"""How varied the street photos of the selection are, and how often each photo is drawn per weight.

    uv run python scripts/street_diversity.py

Counts the YouTube videos behind the frames and the frames per video, for the
selection and for the validation photos, and how many validation videos also
give training frames. Then, with the budget split equally between the cities,
how many times a street and a non-street photo are drawn at each weight.
"""

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402
from curate import read_selection  # noqa: E402
from rules import CITIES, TRACK_A_BUDGET  # noqa: E402


def source(name):
    return name.split("_")[0]


def video(name):
    return Path(name).stem[len("youtube_"):].rsplit("_", 1)[0]


def main():
    selection = read_selection()
    val = [(Path(p).name, c) for p, c in rules.list_photos(ROOT / "data" / "starter" / "val")]
    print(f"{'city':8} {'photos':>6} {'yt':>6} {'videos':>6} {'fr/vid':>6} {'top5 %':>6} {'osv5m':>6}"
          f" {'val yt':>6} {'val vid':>7} {'shared':>6}")
    per_city = {}
    for city in CITIES:
        names = [p.split("/")[1] for p in selection if p.split("/")[0] == city]
        yt = [n for n in names if source(n) == "youtube"]
        frames = collections.Counter(video(n) for n in yt)
        top5 = sum(n for _, n in frames.most_common(5)) / max(1, len(yt))
        osv = sum(source(n) == "osv5m" for n in names)
        val_yt = [n for n, c in val if c == city and source(n) == "youtube"]
        val_videos = {video(n) for n in val_yt}
        print(f"{city:8} {len(names):6} {len(yt):6} {len(frames):6} {len(yt) / max(1, len(frames)):6.1f} {100 * top5:6.1f}"
              f" {osv:6} {len(val_yt):6} {len(val_videos):7} {len(val_videos & set(frames)):6}")
        per_city[city] = (len(names), len(yt) + osv)

    share = TRACK_A_BUDGET / len(CITIES)
    print(f"\ndraws per photo, with {share:.0f} draws per city")
    print(f"{'city':8} " + " ".join(f"{f'x{w} st':>7} {f'x{w} non':>7}" for w in (1, 2, 3)))
    for city, (photos, street) in per_city.items():
        cells = []
        for w in (1, 2, 3):
            unit = share / (w * street + photos - street)
            cells.append(f"{w * unit:7.2f} {unit:7.2f}")
        print(f"{city:8} " + " ".join(cells))


if __name__ == "__main__":
    main()
