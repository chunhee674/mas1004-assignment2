"""Score models on data/streetval, street frames from YouTube channels in neither the starter data nor data/mine.

    python src/new_val.py baseline balanced_clean clean2 b2

A tag is a Track A run (results/<tag>_seed0/model.onnx, 224), a Track B run
(results/<tag>/model.onnx, at the image size in its submission.json) or
"submission" (submission/model.onnx, the Track B release). The
photos were made by scripts/video_frames.py from scripts/val_videos.csv: 140 a
city, at most 20 a video, like the leaderboard. Prints the score (the mean over
cities) and each city, and writes results/curate/new_val.csv, one row per model
(rows of earlier models are kept).
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402
from curate import OUT, predict  # noqa: E402
from rules import CITIES  # noqa: E402

PHOTOS = ROOT / "data" / "streetval"


def model_of(tag):
    a = ROOT / "results" / f"{tag}_seed0" / "model.onnx"
    if a.exists():
        return a, rules.TRACK_A_IMAGE_SIZE
    b = ROOT / "submission" if tag == "submission" else ROOT / "results" / tag
    return b / "model.onnx", json.loads((b / "submission.json").read_text())["image_size"]


def main():
    photos = rules.list_photos(PHOTOS)
    print(f"{len(photos)} photos in {PHOTOS}")
    out = OUT / "new_val.csv"
    rows = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            rows = {r["tag"]: r for r in csv.DictReader(f)}
    for tag in sys.argv[1:]:
        onnx, size = model_of(tag)
        answers = predict(onnx, [p for p, _ in photos], size=size).argmax(1)
        per_city = []
        for i, city in enumerate(CITIES):
            mine = [k for k, (_, c) in enumerate(photos) if c == city]
            per_city.append(sum(answers[k] == i for k in mine) / max(1, len(mine)))
        score = sum(per_city) / len(CITIES)
        print(f"{tag:28} {score:.3f}  " + "  ".join(f"{c} {v:.3f}" for c, v in zip(CITIES, per_city)), flush=True)
        rows[tag] = {"tag": tag, "size": size, "score": round(score, 4),
                     **{c: round(v, 4) for c, v in zip(CITIES, per_city)}}
        OUT.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["tag", "size", "score", *CITIES])
            w.writeheader()
            w.writerows(rows.values())


if __name__ == "__main__":
    main()
