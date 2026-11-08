"""Take evenly spaced frames from YouTube videos listed as city,video_id.

    uv run --with yt-dlp --with imageio-ffmpeg python scripts/video_frames.py
    uv run --with yt-dlp --with imageio-ffmpeg python scripts/video_frames.py \\
        --videos scripts/val_videos.csv --out data/streetval --per-city 140

Without --per-city, one frame a minute (training photos, data/mine). With
--per-city, every city gets that many frames, shared as evenly as possible over
its videos, like the leaderboard photos. The first and last 5% of a video are
skipped, where intros and end screens are. Only the moments that are kept are
read from the stream (at most 720p), so no whole video is downloaded. Frames go
to <out>/<city>/ytmine_<video>_<k>.jpg, which git does not track, saved at
854 x 480 and quality 85 like the starter frames (see shrink_frames.py). Videos whose
frames are already all there are skipped.
"""

import argparse
import csv
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import yt_dlp

from shrink_frames import shrink

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def stream(video):
    opts = {"quiet": True, "no_warnings": True, "format": "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video}", download=False)
    return info["url"], info["duration"], info.get("height")


def grab(url, t, path):
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-ss", f"{t:.1f}", "-i", url, "-frames:v", "1",
                    "-q:v", "2", str(path)], check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "scripts" / "extra_videos.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "mine"))
    ap.add_argument("--every", type=float, default=60, help="seconds between two frames")
    ap.add_argument("--per-city", type=int, help="frames for each city, split over its videos")
    args = ap.parse_args()
    with open(args.videos, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per_city = defaultdict(list)
    for row in rows:
        per_city[row["city"]].append(row["video_id"])
    # photos the student dropped from data/mine stay dropped when this runs again
    dropped = set()
    if Path(args.out) == ROOT / "data" / "mine" and (ROOT / "scripts" / "mine_dropped.csv").exists():
        with open(ROOT / "scripts" / "mine_dropped.csv", encoding="utf-8") as f:
            dropped = {r["photo"] for r in csv.DictReader(f)}

    for row in rows:
        city, video = row["city"], row["video_id"]
        out = Path(args.out) / city
        out.mkdir(parents=True, exist_ok=True)
        try:
            url, duration, height = stream(video)
        except Exception as e:  # a private or removed video
            print(f"{city:7} {video}  failed: {e}", flush=True)
            continue
        if args.per_city:
            share, extra = divmod(args.per_city, len(per_city[city]))
            n = share + (per_city[city].index(video) < extra)
        else:
            n = max(1, int(0.9 * duration / args.every))
        keep = [k for k in range(n) if f"ytmine_{video}_{k:03d}.jpg" not in dropped]
        if len(list(out.glob(f"ytmine_{video}_*.jpg"))) == len(keep):
            continue
        for old in out.glob(f"ytmine_{video}_*.jpg"):
            old.unlink()
        with ThreadPoolExecutor(4) as pool:
            for k in keep:
                t = duration * (0.05 + 0.9 * (k + 0.5) / n)
                pool.submit(grab, url, t, out / f"ytmine_{video}_{k:03d}.jpg")
        for frame in out.glob(f"ytmine_{video}_*.jpg"):
            shrink(frame)
        got = len(list(out.glob(f"ytmine_{video}_*.jpg")))
        print(f"{city:7} {video}  {duration / 60:5.1f} min  {height}p  {got}/{len(keep)} frames", flush=True)


if __name__ == "__main__":
    main()
