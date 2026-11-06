"""Your Track A recipe: how the fixed model is trained on your selection.

src/official.py imports this file and calls the functions below. Change
anything in it. The grader copies this one file and your selection.csv to its
own computer and runs src/official.py with them, so everything your recipe
needs has to be in this file: it cannot read other files, download anything,
or load another model.

What you cannot change, because official.py decides it:
  - the model: ResNet18 with the ImageNet weights, a new last layer of 5
  - the photos: only the ones in selection.csv, with the city of their folder
  - how many photos go through the network in training: rules.TRACK_A_BUDGET
    (250,000). A bigger batch means fewer steps, not more photos.
  - the image size the model sees, 224 x 224
  - mixed precision (float16), and the seed

This is the recipe the starter code came with. It is the baseline on the
leaderboard.
"""

import collections

import torch
import torch.nn.functional as F
from torchvision import transforms as T

BATCH_SIZE = 64
STREET_SOURCES = ("youtube", "osv5m")  # the first word of a photo's file name
STREET_WEIGHT = 2.0  # within a city, how much more often a street photo is drawn
BACKBONE_LR = 1e-4  # the layers that come with the ImageNet weights
LAST_LAYER_LR = 1e-3  # the new last layer


def train_transform():
    """A function that turns one training photo (a PIL image) into a 224 x 224 one.

    It may return a PIL image or a float tensor (3, 224, 224) with values
    0..1. official.py normalises it afterwards. It runs once every time a
    photo is drawn, so the same photo looks different every time.
    """
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.35, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.05),
    ])


def sample_weights(cities, paths):
    """How often each photo in your selection is drawn.

    cities and paths are lists, one entry per row of selection.csv. Return a
    list of positive numbers of the same length, where a photo with weight 2
    is drawn twice as often as one with weight 1. Return None to go through
    the selection in a new random order each time, every photo once per round.
    """
    names = [p.split("/")[-1] for p in paths]
    videos = [n[len("youtube_"):].rsplit("_", 1)[0] if n.startswith("youtube_") else None for n in names]
    frames = collections.Counter(v for v in videos if v)
    video_city = {v: c for v, c in zip(videos, cities) if v}
    per_video = collections.defaultdict(list)
    for v, n in frames.items():
        per_video[video_city[v]].append(n)
    # a video's frames share one amount, the mean size of a video in the city, so every video is drawn equally
    mean_frames = {c: sum(n) / len(n) for c, n in per_video.items()}
    raw = []
    for n, v, c in zip(names, videos, cities):
        if v:
            raw.append(STREET_WEIGHT * mean_frames[c] / frames[v])
        else:
            raw.append(STREET_WEIGHT if n.split("_")[0] in STREET_SOURCES else 1.0)
    totals = collections.Counter()
    for c, w in zip(cities, raw):
        totals[c] += w
    return [w / totals[c] for c, w in zip(cities, raw)]


def make_optimizer(model):
    last = list(model.fc.parameters())
    chosen = {id(p) for p in last}
    rest = [p for p in model.parameters() if id(p) not in chosen]
    return torch.optim.AdamW([{"params": rest, "lr": BACKBONE_LR}, {"params": last, "lr": LAST_LAYER_LR}],
                             weight_decay=0.05)


def make_scheduler(optimizer, total_steps):
    """Stepped once after every training step. total_steps is BUDGET // BATCH_SIZE."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[BACKBONE_LR, LAST_LAYER_LR], total_steps=total_steps, pct_start=0.15)


def mix(images, labels):
    """Change a batch on the GPU before the model sees it.

    images is (BATCH_SIZE, 3, 224, 224), normalised; labels is (BATCH_SIZE,)
    city numbers. Return images of the same shape and the targets for
    loss_fn: the labels, or probabilities (BATCH_SIZE, 5) if you mix photos.
    """
    return images, labels


def loss_fn(logits, targets):
    return F.cross_entropy(logits, targets, label_smoothing=0.1)


def on_step(step, total_steps, model, optimizer):
    """Called after every step, with the step number from 0. Returns nothing."""


def final_model(model):
    """The model to hand in, at the end of training. It must still be the same ResNet18."""
    return model
