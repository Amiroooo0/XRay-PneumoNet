#!/usr/bin/env python3
"""Audit chest X-ray JPEGs and optionally export padded grayscale inputs."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it with: python -m pip install Pillow") from exc


CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def images_in(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def audit(root: Path) -> list[dict[str, object]]:
    rows = []
    for split in ("train", "val", "test"):
        for label in CLASSES:
            folder = root / split / label
            if not folder.is_dir():
                raise SystemExit(f"Expected class folder not found: {folder}")
            for path in images_in(folder):
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        width, height = image.size
                        mode = image.mode
                        image.load()
                    rows.append({
                        "split": split,
                        "label": label,
                        "path": path.relative_to(root).as_posix(),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "bytes": path.stat().st_size,
                        "status": "ok",
                    })
                except Exception as exc:  # report a broken file and continue the audit
                    rows.append({
                        "split": split,
                        "label": label,
                        "path": path.relative_to(root).as_posix(),
                        "width": "",
                        "height": "",
                        "mode": "",
                        "bytes": path.stat().st_size,
                        "status": f"error: {type(exc).__name__}: {exc}",
                    })
    return rows


def write_audit(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = ("split", "label", "path", "width", "height", "mode", "bytes", "status")
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def report(rows: list[dict[str, object]]) -> None:
    counts = Counter((str(row["split"]), str(row["label"])) for row in rows)
    print(f"JPEG files audited: {len(rows)}")
    print("Counts by split and class:")
    for split in ("train", "val", "test"):
        print(f"  {split:5} " + "  ".join(f"{label}={counts[(split, label)]}" for label in CLASSES))

    valid = [row for row in rows if row["status"] == "ok"]
    modes = Counter(str(row["mode"]) for row in valid)
    sizes = Counter((int(row["width"]), int(row["height"])) for row in valid)
    ratios = [int(row["width"]) / int(row["height"]) for row in valid]
    bytes_list = [int(row["bytes"]) for row in valid]
    print(f"Readable images: {len(valid)}; failures: {len(rows) - len(valid)}")
    print(f"Modes: {dict(modes)}; distinct dimensions: {len(sizes)}")
    if valid:
        print(
            "Dimensions: "
            f"width {min(int(r['width']) for r in valid)}-{max(int(r['width']) for r in valid)}, "
            f"height {min(int(r['height']) for r in valid)}-{max(int(r['height']) for r in valid)}; "
            f"median aspect ratio {statistics.median(ratios):.3f}"
        )
        print(
            "File size: "
            f"median {statistics.median(bytes_list):,.0f} bytes, "
            f"range {min(bytes_list):,}-{max(bytes_list):,} bytes"
        )
    if len(sizes) <= 30:
        print(f"Dimension frequencies: {dict(sizes)}")


def make_manifest(root: Path, rows: list[dict[str, object]], destination: Path,
                  validation_fraction: float, seed: int) -> list[dict[str, str]]:
    # Existing train and tiny val form development data; test is kept isolated.
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok" and row["split"] in ("train", "val"):
            by_label[str(row["label"])].append(str(row["path"]))

    rng = random.Random(seed)
    manifest: list[dict[str, str]] = []
    for label in CLASSES:
        paths = by_label[label]
        rng.shuffle(paths)
        validation_count = max(1, round(len(paths) * validation_fraction))
        for index, path in enumerate(paths):
            manifest.append({
                "path": path,
                "label": label,
                "split": "val" if index < validation_count else "train",
            })

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "label", "split"))
        writer.writeheader()
        writer.writerows(sorted(manifest, key=lambda row: (row["split"], row["label"], row["path"])))
    return manifest


def preprocess(image: Image.Image, size: int, resize_mode: str = "pad",
               padding_mode: str = "black") -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("L")
    if resize_mode == "crop":
        return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS,
                            centering=(0.5, 0.5))

    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    fill = 0
    if padding_mode == "edge":
        pixels = image.load()
        border = [pixels[x, 0] for x in range(image.width)]
        border.extend(pixels[x, image.height - 1] for x in range(image.width))
        border.extend(pixels[0, y] for y in range(1, image.height - 1))
        border.extend(pixels[image.width - 1, y] for y in range(1, image.height - 1))
        fill = round(statistics.median(border))
    canvas = Image.new("L", (size, size), color=fill)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def export_images(root: Path, output: Path, manifest: list[dict[str, str]], size: int,
                  include_test: bool, resize_mode: str, padding_mode: str) -> None:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty; choose a new path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    items = list(manifest)
    if include_test:
        for label in CLASSES:
            for source in images_in(root / "test" / label):
                items.append({
                    "path": source.relative_to(root).as_posix(),
                    "label": label,
                    "split": "test",
                })

    index_rows = []
    for item in items:
        source = root / Path(item["path"])
        relative = Path(item["split"]) / item["label"] / f"{source.stem}.png"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            processed = preprocess(image, size, resize_mode, padding_mode)
            processed.save(destination, format="PNG", optimize=True)
        index_rows.append({**item, "output_path": relative.as_posix()})

    with (output / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "label", "split", "output_path"))
        writer.writeheader()
        writer.writerows(index_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="dataset root containing train/val/test")
    parser.add_argument("--audit", type=Path, default=Path("data_audit.csv"),
                        help="write the image audit CSV here")
    parser.add_argument("--manifest", type=Path, default=Path("split_manifest.csv"),
                        help="write a reproducible train/validation manifest here")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", type=Path,
                        help="optional new output folder for resized grayscale PNGs")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--resize-mode", choices=("pad", "crop"), default="pad",
                        help="preserve the full image with padding, or center-crop to square")
    parser.add_argument("--padding-mode", choices=("black", "edge"), default="black",
                        help="fill padding with black or median border intensity")
    parser.add_argument("--include-test", action="store_true",
                        help="also export untouched test membership to output/test")
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.size < 16:
        parser.error("--size must be at least 16")

    rows = audit(args.root.resolve())
    write_audit(rows, args.audit)
    report(rows)
    manifest = make_manifest(args.root.resolve(), rows, args.manifest,
                             args.validation_fraction, args.seed)
    split_counts = Counter((row["split"], row["label"]) for row in manifest)
    print("Development manifest counts:")
    for split in ("train", "val"):
        print(f"  {split:5} " + "  ".join(f"{label}={split_counts[(split, label)]}" for label in CLASSES))
    print(f"Audit CSV: {args.audit.resolve()}")
    print(f"Manifest: {args.manifest.resolve()}")
    if args.export:
        export_images(args.root.resolve(), args.export.resolve(), manifest, args.size,
                      args.include_test, args.resize_mode, args.padding_mode)
        print(f"Processed PNGs: {args.export.resolve()}")


if __name__ == "__main__":
    main()
