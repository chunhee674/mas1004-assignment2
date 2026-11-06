"""Train another recipe with src/official.py, so that one Colab session can run several.

    python src/run_variant.py track_a_variants/color --tag clean_street2_video_color --seed 0

The folder holds a recipe.py. It runs with track_a/selection.csv, in a temporary
folder that official.py is pointed to, and everything after the folder goes to
official.py. Nothing in track_a/ changes. To release a variant, its recipe.py
has to be copied to track_a/ first.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder, rest = ROOT / sys.argv[1], sys.argv[2:]
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(folder / "recipe.py", tmp)
        shutil.copy(ROOT / "track_a" / "selection.csv", tmp)
        env = {**os.environ, "MAS1004_RECIPE_DIR": tmp}
        sys.exit(subprocess.run([sys.executable, "-u", str(ROOT / "src" / "official.py"), *rest], env=env).returncode)


if __name__ == "__main__":
    main()
