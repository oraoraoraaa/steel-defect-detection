# Tools

Standalone data-preparation utilities. Everything here is Python 3.10+ stdlib-only,
so it runs before any ML environment (PyTorch, ultralytics, …) is installed.

## voc_yolo.py — VOC ↔ YOLO annotation converter

The contest training set arrives as PASCAL VOC XML (one `.xml` per image,
absolute pixel corner boxes `xmin, ymin, xmax, ymax`). YOLO-family trainers
need one `.txt` per image with `class_id cx cy w h` normalized to `[0, 1]`.
This tool converts in both directions, validates boxes along the way
(clamps out-of-bounds coordinates, drops degenerate boxes, flags unknown
classes), and prints dataset statistics.

### Convert contest VOC data to a YOLO dataset

```sh
python3 tools/voc_yolo.py voc2yolo \
    --images path/to/JPEGImages \
    --annotations path/to/Annotations \
    --out dataset/contest-yolo \
    --link-images
```

- Class order is discovered automatically (alphabetical) unless pinned with
  `--classes a,b,c`. **Pin it once the real data arrives** so every split and
  every teammate uses the same class ids.
- `--link-images` symlinks images instead of copying (the contest images are
  ~4096×3000; copying 3,200 of them wastes gigabytes).
- Writes `images/`, `labels/`, and a `data.yaml` into `--out`.

### Convert YOLO labels back to VOC XML

```sh
python3 tools/voc_yolo.py yolo2voc \
    --images dataset/NEU-DET-yolo26/valid/images \
    --labels dataset/NEU-DET-yolo26/valid/labels \
    --data dataset/NEU-DET-yolo26/data.yaml \
    --out out/voc_valid
```

Used to generate VOC fixtures for testing (NEU-DET-yolo26 ships YOLO-only)
and to export predictions/labels for VOC-based tooling.

### Dataset statistics

```sh
# YOLO layout (a split dir containing images/ and labels/)
python3 tools/voc_yolo.py stats --yolo dataset/NEU-DET-yolo26/train \
    --data dataset/NEU-DET-yolo26/data.yaml

# VOC layout (a dir of .xml files)
python3 tools/voc_yolo.py stats --voc path/to/Annotations
```

Reports per-class box counts, class share, and bbox size distribution
(min/median/max of √area in pixels) — run this first on the contest data to
see the long-tail imbalance and pick tiling/anchor strategies.

### Verified

Round-trip test on NEU-DET-yolo26 `valid` (360 images, 556 boxes):
`yolo2voc` → `voc2yolo` reproduces the original labels with zero coordinate
error and zero class mismatches.
