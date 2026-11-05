"""Which scripts are written in each validation photo, read with EasyOCR. For analysis only.

    python src/ocr_scripts.py

The model that is handed in does not use this, and nothing here chooses photos.
Text boxes are found once. Each box is then read by the Korean, the Japanese
and the Traditional Chinese recogniser, because EasyOCR cannot load them
together, and the most confident reading counts. Readings below MIN_CONFIDENCE
are ignored. Writes results/curate/ocr_val.csv, one row per photo, with the
number of boxes containing Hangul, kana, Han characters and Latin letters.
A run that stops continues where it stopped.
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rules  # noqa: E402

OUT = ROOT / "results" / "curate" / "ocr_val.csv"
MIN_CONFIDENCE = 0.4
FIELDS = ["path", "city", "boxes", "hangul", "kana", "han", "latin", "text"]
HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
KANA = re.compile(r"[぀-ヿ]")
HAN = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=1, help="use every n-th photo only")
    args = ap.parse_args()

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "easyocr"], check=True)
    import easyocr
    from easyocr.utils import reformat_input

    print("easyocr", easyocr.__version__, flush=True)
    readers = [easyocr.Reader([lang, "en"], gpu=True, verbose=False) for lang in ("ko", "ja", "ch_tra")]

    rows = rules.list_photos(ROOT / "data" / "starter" / "val")[::args.every]
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
            img, grey = reformat_input(str(path))
            horizontal, free = readers[0].detect(img)
            horizontal, free = horizontal[0], free[0]
            best = {}
            if horizontal or free:
                for reader in readers:
                    for box, text, conf in reader.recognize(grey, horizontal_list=horizontal, free_list=free):
                        key = repr(box)
                        if key not in best or conf > best[key][1]:
                            best[key] = (text, conf)
            texts = [t for t, c in best.values() if c >= MIN_CONFIDENCE]
            w.writerow({"path": rel, "city": city, "boxes": len(texts),
                        "hangul": sum(bool(HANGUL.search(t)) for t in texts),
                        "kana": sum(bool(KANA.search(t)) for t in texts),
                        "han": sum(bool(HAN.search(t)) for t in texts),
                        "latin": sum(bool(LATIN.search(t)) for t in texts),
                        "text": " | ".join(texts)[:200]})
            if n % 200 == 0:
                f.flush()
                print(f"  {n}/{len(rows)} ({time.time() - start:.0f}s)", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
