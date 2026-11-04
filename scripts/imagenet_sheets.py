"""Candidate sheets by what an ImageNet classifier sees in a photo, not by its title.

    uv run python scripts/imagenet_sheets.py

Every photo still in track_a/selection.csv goes through torchvision's ResNet18
with its ImageNet-1k weights (no training, nothing about the cities). A photo
is a candidate when the classifier's first answer is in one of the KINDS below
with probability at least MIN_PROB. Candidates go on sheets per kind and city,
the most certain first:
results/curate/cand_in_<kind>_<city>_NN (48 per sheet).
The answers for every photo go to results/curate/imagenet_top1.csv, so a
second run only draws the sheets. The student looks at every sheet and names
the photos to drop.
"""

import csv
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curate import OUT, TRAIN, draw_sheets, read_selection  # noqa: E402
from rules import CITIES  # noqa: E402

# ImageNet-1k class numbers
KINDS = {
    "food": [809, 923] + list(range(924, 970)),        # soup bowl, plate, dishes and food
    "animal": list(range(0, 398)),                     # all animals
    "car": [436, 468, 479, 511, 581, 609, 627, 656, 661, 675, 717, 734, 751, 817, 864, 867],
}
MIN_PROB = 0.5
CACHE = OUT / "imagenet_top1.csv"


class Photos(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(TRAIN / self.paths[i]) as im:
            return self.tf(ImageOps.exif_transpose(im).convert("RGB"))


def classify(paths):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights).eval()
    loader = torch.utils.data.DataLoader(Photos(paths), batch_size=64, num_workers=4)
    out = []
    with torch.inference_mode():
        for k, x in enumerate(loader):
            p, c = net(x).softmax(1).max(1)
            out += list(zip(c.tolist(), p.tolist()))
            if k % 50 == 0:
                print(f"  {len(out)}/{len(paths)}", flush=True)
    names = weights.meta["categories"]
    with open(CACHE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "name", "probability"])
        for path, (c, p) in zip(paths, out):
            w.writerow([path, c, names[c], round(p, 4)])


def main():
    selection = read_selection()
    if not CACHE.exists():
        classify(selection)
    with open(CACHE, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["path"] in set(selection)]
    for kind, classes in KINDS.items():
        classes = set(classes)
        for city in CITIES:
            found = sorted((r for r in rows if r["path"].split("/")[0] == city
                            and int(r["class"]) in classes and float(r["probability"]) >= MIN_PROB),
                           key=lambda r: -float(r["probability"]))
            names = draw_sheets(f"cand_in_{kind}_{city}",
                                [(TRAIN / r["path"], f"{r['name']} {float(r['probability']):.2f}") for r in found])
            print(f"{kind}, {city}: {len(found)} photos, on {', '.join(names) or '-'}")


if __name__ == "__main__":
    main()
