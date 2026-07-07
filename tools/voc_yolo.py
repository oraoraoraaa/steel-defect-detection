#!/usr/bin/env python3
"""Convert object-detection annotations between PASCAL VOC XML and YOLO txt.

The 2026AIC contest ships its training set as PASCAL VOC XML (absolute pixel
corner boxes); YOLO-family trainers consume one .txt per image with normalized
center-format boxes. This tool converts in both directions and reports dataset
statistics. Stdlib-only on purpose: it must run before any ML environment is
set up.

Subcommands:
    voc2yolo   VOC (images + XML annotations) -> YOLO (images/ labels/ data.yaml)
    yolo2voc   YOLO (images/ labels/ + class list) -> VOC XML annotations
    stats      Per-class counts and bbox size distribution for either format

Run `python3 tools/voc_yolo.py <subcommand> -h` for options.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Box:
    """One annotated defect. Coordinates are absolute pixels, 0-based,
    (xmin, ymin) top-left inclusive, (xmax, ymax) bottom-right exclusive-ish;
    the contest spec only guarantees 0 <= v <= width/height."""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class ImageAnnotation:
    image_name: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Image size probing (stdlib replacement for PIL, JPEG/PNG/BMP only)
# ---------------------------------------------------------------------------

def read_image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) by parsing the file header."""
    with open(path, "rb") as f:
        head = f.read(26)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            w, h = struct.unpack(">II", head[16:24])
            return w, h
        if head.startswith(b"BM"):
            w, h = struct.unpack("<ii", head[18:26])
            return w, abs(h)
        if head.startswith(b"\xff\xd8"):
            f.seek(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    break
                # SOF0..SOF15 except DHT(C4)/DAC(CC)/RST have the frame size
                if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                    f.read(3)  # segment length (2) + bit depth (1)
                    h, w = struct.unpack(">HH", f.read(4))
                    return w, h
                (seg_len,) = struct.unpack(">H", f.read(2))
                f.seek(seg_len - 2, 1)
    raise ValueError(f"unsupported or corrupt image file: {path}")


# ---------------------------------------------------------------------------
# VOC side
# ---------------------------------------------------------------------------

def parse_voc_xml(xml_path: Path) -> ImageAnnotation:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path}: missing <size>")
    ann = ImageAnnotation(
        image_name=root.findtext("filename", default=xml_path.stem),
        width=int(float(size.findtext("width"))),
        height=int(float(size.findtext("height"))),
    )
    for obj in root.iter("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        ann.boxes.append(
            Box(
                name=obj.findtext("name", default="unknown").strip(),
                xmin=float(bnd.findtext("xmin")),
                ymin=float(bnd.findtext("ymin")),
                xmax=float(bnd.findtext("xmax")),
                ymax=float(bnd.findtext("ymax")),
            )
        )
    return ann


def write_voc_xml(ann: ImageAnnotation, xml_path: Path, depth: int = 1) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = ann.image_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(ann.width)
    ET.SubElement(size, "height").text = str(ann.height)
    ET.SubElement(size, "depth").text = str(depth)
    for box in ann.boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = box.name
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(round(box.xmin)))
        ET.SubElement(bnd, "ymin").text = str(int(round(box.ymin)))
        ET.SubElement(bnd, "xmax").text = str(int(round(box.xmax)))
        ET.SubElement(bnd, "ymax").text = str(int(round(box.ymax)))
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(xml_path, encoding="unicode")


# ---------------------------------------------------------------------------
# YOLO side
# ---------------------------------------------------------------------------

def parse_yolo_txt(txt_path: Path, classes: list[str], width: int, height: int) -> list[Box]:
    boxes = []
    for line_no, line in enumerate(txt_path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{txt_path}:{line_no}: expected 'cls cx cy w h'")
        cls_id = int(parts[0])
        if not 0 <= cls_id < len(classes):
            raise ValueError(f"{txt_path}:{line_no}: class id {cls_id} out of range")
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append(
            Box(
                name=classes[cls_id],
                xmin=(cx - w / 2) * width,
                ymin=(cy - h / 2) * height,
                xmax=(cx + w / 2) * width,
                ymax=(cy + h / 2) * height,
            )
        )
    return boxes


def yolo_line(box: Box, classes: list[str], width: int, height: int) -> str:
    cls_id = classes.index(box.name)
    cx = (box.xmin + box.xmax) / 2 / width
    cy = (box.ymin + box.ymax) / 2 / height
    w = (box.xmax - box.xmin) / width
    h = (box.ymax - box.ymin) / height
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def read_classes_from_data_yaml(yaml_path: Path) -> list[str]:
    """Minimal parser for the `names:` entry of an ultralytics data.yaml.
    Handles flow style (names: ['a', 'b']) and block style (- a) lists."""
    lines = yaml_path.read_text().splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        rest = stripped[len("names:"):].strip()
        if rest.startswith("["):
            items = rest.strip("[]").split(",")
            return [it.strip().strip("'\"") for it in items if it.strip()]
        names = []
        for follow in lines[i + 1:]:
            fs = follow.strip()
            if fs.startswith("- "):
                names.append(fs[2:].strip().strip("'\""))
            elif fs and not fs.startswith("#"):
                break
        if names:
            return names
    raise ValueError(f"could not find a 'names:' list in {yaml_path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def sanitize_boxes(ann: ImageAnnotation, source: str, warnings: list[str]) -> list[Box]:
    """Clamp boxes to image bounds and drop degenerate ones, recording why."""
    kept = []
    for box in ann.boxes:
        clamped = Box(
            box.name,
            min(max(box.xmin, 0), ann.width),
            min(max(box.ymin, 0), ann.height),
            min(max(box.xmax, 0), ann.width),
            min(max(box.ymax, 0), ann.height),
        )
        if (clamped.xmin, clamped.ymin, clamped.xmax, clamped.ymax) != (
            box.xmin, box.ymin, box.xmax, box.ymax,
        ):
            warnings.append(f"{source}: clamped out-of-bounds box for '{box.name}'")
        if clamped.xmax - clamped.xmin < 1 or clamped.ymax - clamped.ymin < 1:
            warnings.append(f"{source}: dropped degenerate box for '{box.name}'")
            continue
        kept.append(clamped)
    return kept


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def cmd_voc2yolo(args: argparse.Namespace) -> int:
    ann_dir, img_dir, out_dir = Path(args.annotations), Path(args.images), Path(args.out)
    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        print(f"error: no .xml files in {ann_dir}", file=sys.stderr)
        return 1

    annotations = [(p, parse_voc_xml(p)) for p in xml_files]
    if args.classes:
        classes = args.classes.split(",")
    else:
        classes = sorted({b.name for _, a in annotations for b in a.boxes})
        print(f"discovered {len(classes)} classes: {classes}")

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    n_boxes = skipped = 0
    for xml_path, ann in annotations:
        image_path = find_image(img_dir, xml_path.stem)
        if image_path is None:
            warnings.append(f"{xml_path.name}: no matching image in {img_dir}, skipped")
            skipped += 1
            continue
        unknown = [b for b in ann.boxes if b.name not in classes]
        for b in unknown:
            warnings.append(f"{xml_path.name}: unknown class '{b.name}', box dropped")
        ann.boxes = [b for b in ann.boxes if b.name in classes]
        boxes = sanitize_boxes(ann, xml_path.name, warnings)
        lines = [yolo_line(b, classes, ann.width, ann.height) for b in boxes]
        (out_dir / "labels" / f"{xml_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        if args.link_images:
            dest = out_dir / "images" / image_path.name
            if not dest.exists():
                dest.symlink_to(image_path.resolve())
        else:
            shutil.copy2(image_path, out_dir / "images" / image_path.name)
        n_boxes += len(lines)

    names_flow = ", ".join(f"'{c}'" for c in classes)
    (out_dir / "data.yaml").write_text(
        f"train: images\n\nnc: {len(classes)}\nnames: [{names_flow}]\n"
    )
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(
        f"wrote {len(annotations) - skipped} label files ({n_boxes} boxes, "
        f"{len(warnings)} warnings) to {out_dir}"
    )
    return 0


def cmd_yolo2voc(args: argparse.Namespace) -> int:
    img_dir, lbl_dir, out_dir = Path(args.images), Path(args.labels), Path(args.out)
    if args.data:
        classes = read_classes_from_data_yaml(Path(args.data))
    elif args.classes:
        classes = args.classes.split(",")
    else:
        print("error: provide --data data.yaml or --classes a,b,c", file=sys.stderr)
        return 1

    txt_files = sorted(lbl_dir.glob("*.txt"))
    if not txt_files:
        print(f"error: no .txt files in {lbl_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    n_boxes = skipped = 0
    for txt_path in txt_files:
        image_path = find_image(img_dir, txt_path.stem)
        if image_path is None:
            print(f"warning: {txt_path.name}: no matching image, skipped", file=sys.stderr)
            skipped += 1
            continue
        width, height = read_image_size(image_path)
        ann = ImageAnnotation(image_path.name, width, height)
        ann.boxes = parse_yolo_txt(txt_path, classes, width, height)
        write_voc_xml(ann, out_dir / f"{txt_path.stem}.xml")
        n_boxes += len(ann.boxes)
    print(f"wrote {len(txt_files) - skipped} xml files ({n_boxes} boxes) to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def cmd_stats(args: argparse.Namespace) -> int:
    annotations: list[ImageAnnotation] = []
    if args.voc:
        for xml_path in sorted(Path(args.voc).glob("*.xml")):
            annotations.append(parse_voc_xml(xml_path))
    else:
        yolo_dir = Path(args.yolo)
        data_yaml = Path(args.data) if args.data else yolo_dir / "data.yaml"
        classes = read_classes_from_data_yaml(data_yaml)
        img_dir, lbl_dir = yolo_dir / "images", yolo_dir / "labels"
        for txt_path in sorted(lbl_dir.glob("*.txt")):
            image_path = find_image(img_dir, txt_path.stem)
            if image_path is None:
                continue
            width, height = read_image_size(image_path)
            ann = ImageAnnotation(image_path.name, width, height)
            ann.boxes = parse_yolo_txt(txt_path, classes, width, height)
            annotations.append(ann)

    if not annotations:
        print("error: no annotations found", file=sys.stderr)
        return 1

    counts = Counter(b.name for a in annotations for b in a.boxes)
    total = sum(counts.values())
    sizes: dict[str, list[float]] = {}
    for a in annotations:
        for b in a.boxes:
            sizes.setdefault(b.name, []).append(
                ((b.xmax - b.xmin) * (b.ymax - b.ymin)) ** 0.5
            )

    print(f"images: {len(annotations)}   boxes: {total}   "
          f"boxes/image: {total / len(annotations):.2f}")
    print(f"{'class':<28}{'boxes':>7}{'share':>8}{'min√A':>8}{'med√A':>8}{'max√A':>8}")
    for name, cnt in counts.most_common():
        s = sorted(sizes[name])
        print(f"{name:<28}{cnt:>7}{cnt / total:>7.1%}"
              f"{s[0]:>8.0f}{s[len(s) // 2]:>8.0f}{s[-1]:>8.0f}")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("voc2yolo", help="convert VOC XML annotations to YOLO layout")
    p.add_argument("--images", required=True, help="directory with image files")
    p.add_argument("--annotations", required=True, help="directory with VOC .xml files")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--classes", help="comma-separated class order (default: discover, sorted)")
    p.add_argument("--link-images", action="store_true",
                   help="symlink images into the output instead of copying")
    p.set_defaults(func=cmd_voc2yolo)

    p = sub.add_parser("yolo2voc", help="convert YOLO labels to VOC XML annotations")
    p.add_argument("--images", required=True, help="directory with image files")
    p.add_argument("--labels", required=True, help="directory with YOLO .txt files")
    p.add_argument("--out", required=True, help="output directory for .xml files")
    p.add_argument("--data", help="data.yaml to read class names from")
    p.add_argument("--classes", help="comma-separated class names (alternative to --data)")
    p.set_defaults(func=cmd_yolo2voc)

    p = sub.add_parser("stats", help="per-class counts and bbox size distribution")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--voc", help="directory with VOC .xml files")
    g.add_argument("--yolo", help="YOLO split directory containing images/ and labels/")
    p.add_argument("--data", help="data.yaml for --yolo (default: <yolo>/data.yaml)")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
