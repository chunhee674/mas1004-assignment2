"""Does a model know the YouTube frames it trained on better than new ones? For analysis only.

    python src/street_memory.py clean_street2_video

Takes results/<tag>_seed0/model.onnx. Runs it on the YouTube frames in
track_a/selection.csv and on the YouTube validation frames. Prints, per true
city, the accuracy and where the answers go, and the same with every city
counting equally. Writes results/curate/street_memory_<tag>.csv, one row per photo.
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402
from curate import OUT, TRAIN, predict, read_selection  # noqa: E402
from rules import CITIES  # noqa: E402


def table(name, rows):
    print(f"\n{name:8} {'photos':>6} {'acc':>6}  answers " + " ".join(f"{c:>6}" for c in CITIES))
    shares = []
    for city in CITIES:
        mine = [a for c, a in rows if c == city]
        share = [sum(a == other for a in mine) / max(1, len(mine)) for other in CITIES]
        shares.append(share)
        print(f"{city:8} {len(mine):6} {share[CITIES.index(city)]:6.3f}          " + " ".join(f"{s:6.3f}" for s in share))
    mean = [sum(s[k] for s in shares) / len(CITIES) for k in range(len(CITIES))]
    score = sum(shares[k][k] for k in range(len(CITIES))) / len(CITIES)
    print(f"{'equal':8} {len(rows):6} {score:6.3f}          " + " ".join(f"{s:6.3f}" for s in mean))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--every", type=int, default=1, help="use every n-th photo only")
    args = ap.parse_args()

    train = [(TRAIN / p, p.split("/")[0]) for p in read_selection() if Path(p).name.startswith("youtube_")]
    val = [(p, c) for p, c in rules.list_photos(ROOT / "data" / "starter" / "val")
           if Path(p).name.startswith("youtube_")]
    model = ROOT / "results" / f"{args.tag}_seed0" / "model.onnx"

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"street_memory_{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "city", "answer"])
        for name, photos in (("train", train[::args.every]), ("val", val[::args.every])):
            answers = [CITIES[k] for k in predict(model, [p for p, _ in photos]).argmax(1)]
            table(name, [(c, a) for (_, c), a in zip(photos, answers)])
            for (p, c), a in zip(photos, answers):
                w.writerow([name, f"{c}/{Path(p).name}", c, a])
            f.flush()


if __name__ == "__main__":
    main()
