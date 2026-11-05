"""Where text is in each validation photo, found with EasyOCR's detector. For analysis only.

    python src/ocr_boxes.py

The model that is handed in does not use this, and nothing here chooses photos.
Photos are opened the way the grader opens them, so the boxes fit what the model
sees. Nothing is read, only found. Writes results/curate/ocr_boxes.csv, one row
per photo, with the boxes as a JSON list of [left, top, right, bottom] in pixels.
A run that stops continues where it stopped.
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402

OUT = ROOT / "results" / "curate" / "ocr_boxes.csv"
FIELDS = ["path", "city", "boxes"]


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "easyocr"], check=True)
    import easyocr

    print("easyocr", easyocr.__version__, flush=True)
    reader = easyocr.Reader(["en"], gpu=True, recognizer=False, verbose=False)

    rows = rules.list_photos(ROOT / "data" / "starter" / "val")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            done = {r["path"] for r in csv.DictReader(f)}
    else:
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, FIELDS).writeheader()
    print(f"{len(rows)} photos, {len(done)} done before", flush=True)

    start = time.time()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        for n, (path, city) in enumerate(rows, start=1):
            rel = f"{city}/{Path(path).name}"
            if rel in done:
                continue
            horizontal, free = reader.detect(np.array(rules.load_photo(path)))
            boxes = [[int(x0), int(y0), int(x1), int(y1)] for x0, x1, y0, y1 in horizontal[0]]
            for points in free[0]:
                xs, ys = zip(*points)
                boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
            w.writerow({"path": rel, "city": city, "boxes": json.dumps(boxes)})
            if n % 200 == 0:
                f.flush()
                print(f"  {n}/{len(rows)} ({time.time() - start:.0f}s)", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
