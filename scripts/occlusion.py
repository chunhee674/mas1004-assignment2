"""Does a model look at the text in a photo? Cover it and see whether the answer changes.

    uv run python scripts/occlusion.py submission/model.onnx --size 448 --name b2

Takes results/curate/ocr_boxes.csv. For each validation photo with a text box in
the middle square the model sees, three versions go through the model: the
photo as it is, the text boxes painted grey, and boxes of the same sizes painted
grey at random places away from any text box (the control; seed 0). A photo where
the control boxes do not fit is left out. Prints, for each group, how often the
answer changes and how often a right answer becomes wrong under each cover.
Writes results/curate/occlusion_<name>.csv and sheets occlusion_<name>_check_00
(random photos, each text cover followed by its control).
"""

import argparse
import csv
import json
import random
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402
from curate import OUT, draw_sheets  # noqa: E402
from rules import CITIES  # noqa: E402

VAL = ROOT / "data" / "starter" / "val"
GREY = (128, 128, 128)
TRIES = 200


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def square(width, height):
    """The middle square of a photo, the part the model sees."""
    side = min(width, height)
    x, y = (width - side) // 2, (height - side) // 2
    return x, y, x + side, y + side


def clip(box, area):
    """A box cut to the middle square, or None if nothing of it is inside."""
    out = [max(box[0], area[0]), max(box[1], area[1]), min(box[2], area[2]), min(box[3], area[3])]
    return out if out[0] < out[2] and out[1] < out[3] else None


def control_boxes(boxes, area, rng):
    """Boxes of the same sizes inside the square, away from every text box and each other."""
    placed = []
    for b in sorted(boxes, key=lambda b: (b[0] - b[2]) * (b[3] - b[1])):
        w, h = b[2] - b[0], b[3] - b[1]
        for _ in range(TRIES):
            x, y = rng.randint(area[0], area[2] - w), rng.randint(area[1], area[3] - h)
            c = [x, y, x + w, y + h]
            if not any(overlaps(c, o) for o in boxes + placed):
                placed.append(c)
                break
        else:
            return None
    return placed


def covered(path, boxes):
    im = rules.load_photo(path).copy()
    draw = ImageDraw.Draw(im)
    for b in boxes:
        draw.rectangle([b[0], b[1], b[2] - 1, b[3] - 1], fill=GREY)
    return im


def answers(session, jobs, size, batch=32):
    """The answer of the model for each (path, boxes) job."""
    out = []
    with ThreadPoolExecutor(4) as pool:
        for i in range(0, len(jobs), batch):
            x = np.stack(list(pool.map(lambda j: rules.grader_input(covered(*j), size), jobs[i:i + batch])))
            out += session.run(["logits"], {"image": x})[0].argmax(1).tolist()
            print(f"\r  {min(i + batch, len(jobs))}/{len(jobs)}", end="", flush=True)
    print()
    return out


def report(name, group):
    if not group:
        return
    n = len(group)
    right = [r for r in group if r["plain"] == r["city"]]
    line = f"{name:22} {n:6} {sum(r['coverage'] for r in group) / n:6.3f} {len(right) / n:6.3f}"
    for k in ("text", "control"):
        line += f" {sum(r[k] == r['city'] for r in group) / n:6.3f}"
    for k in ("text", "control"):
        line += f" {sum(r[k] != r['plain'] for r in group) / n:7.3f}"
    for k in ("text", "control"):
        line += f" {sum(r[k] != r['city'] for r in right) / max(1, len(right)):7.3f}"
    print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--name", required=True)
    ap.add_argument("--every", type=int, default=1, help="use every n-th photo only")
    args = ap.parse_args()

    import onnxruntime
    session = onnxruntime.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    with open(OUT / "ocr_boxes.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[::args.every]

    rng = random.Random(0)
    photos, no_text, no_room = [], 0, 0
    for r in rows:
        path = VAL / r["path"]
        area = square(*rules.load_photo(path).size)
        boxes = [c for c in (clip(b, area) for b in json.loads(r["boxes"])) if c]
        if not boxes:
            no_text += 1
            continue
        control = control_boxes(boxes, area, rng)
        if control is None:
            no_room += 1
            continue
        side = area[2] - area[0]
        coverage = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / side ** 2
        photos.append({"path": r["path"], "city": r["city"], "boxes": boxes, "control_boxes": control,
                       "coverage": coverage})
    print(f"{len(rows)} photos: {no_text} without a text box in the square, {no_room} where the control "
          f"did not fit, {len(photos)} used")

    for key, boxes_key in (("plain", None), ("text", "boxes"), ("control", "control_boxes")):
        print(key)
        got = answers(session, [(VAL / p["path"], p[boxes_key] if boxes_key else []) for p in photos], args.size)
        for p, a in zip(photos, got):
            p[key] = CITIES[a]

    with open(OUT / f"occlusion_{args.name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "city", "coverage", "plain", "text", "control"])
        for p in photos:
            w.writerow([p["path"], p["city"], round(p["coverage"], 5), p["plain"], p["text"], p["control"]])

    print(f"\n{'':22} {'photos':>6} {'cover':>6} {'plain':>6} {'text':>6} {'ctrl':>6} "
          f"{'chg txt':>7} {'chg ctl':>7} {'r>w txt':>7} {'r>w ctl':>7}")
    report("all", photos)
    for city in CITIES:
        report(city, [p for p in photos if p["city"] == city])
    for low, high in ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 1.01)):
        report(f"cover {low}-{high}", [p for p in photos if low <= p["coverage"] < high])

    pick = random.Random(0).sample(photos, min(24, len(photos)))
    with tempfile.TemporaryDirectory() as tmp:
        items = []
        for k, p in enumerate(pick):
            for key, boxes_key in (("text", "boxes"), ("control", "control_boxes")):
                out = Path(tmp) / f"{k}_{key}.jpg"
                covered(VAL / p["path"], p[boxes_key]).save(out)
                items.append((out, f"{key} {p['city']}>{p[key]} was {p['plain']}"))
        draw_sheets(f"occlusion_{args.name}_check", items)


if __name__ == "__main__":
    main()
