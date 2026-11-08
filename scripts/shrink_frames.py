"""Save video frames the way the starter frames are saved: 854 x 480, JPEG quality 85.

    uv run python scripts/shrink_frames.py data/mine data/streetval

Our frames came out at 1280 x 720 and almost without compression, while the
starter frames of every city are 854 x 480 (some 640 x 360) at about quality
85. Because only Busan, Osaka and Taipei got our frames, sharpness alone told
the model the city (b3). Frames already 480 high or lower are left as they are,
so running this again changes nothing.
"""

import sys
from pathlib import Path

from PIL import Image

HEIGHT, QUALITY = 480, 85


def shrink(path):
    """Shrink one frame in place; True if it was changed."""
    with Image.open(path) as im:
        if im.height <= HEIGHT:
            return False
        small = im.convert("RGB").resize((round(im.width * HEIGHT / im.height), HEIGHT), Image.LANCZOS)
    small.save(path, quality=QUALITY)
    return True


def main():
    for folder in sys.argv[1:]:
        for city in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
            photos = sorted(city.glob("*.jpg"))
            changed = sum(shrink(p) for p in photos)
            print(f"{city}: {changed} of {len(photos)} frames shrunk", flush=True)


if __name__ == "__main__":
    main()
