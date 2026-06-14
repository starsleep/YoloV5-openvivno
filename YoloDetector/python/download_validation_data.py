"""Download the COCO128 images and YOLO-format ground-truth labels."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download COCO128 validation data")
    parser.add_argument("--output", default="../validation_data")
    return parser.parse_args()


def download_coco128(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "coco128"
    images_dir = dataset_dir / "images" / "train2017"
    labels_dir = dataset_dir / "labels" / "train2017"
    if images_dir.exists() and labels_dir.exists():
        return dataset_dir

    archive_path = output_dir / "coco128.zip"
    if not archive_path.exists():
        print(f"Downloading: {COCO128_URL}")
        urlretrieve(COCO128_URL, archive_path)

    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (output_dir / member.filename).resolve()
            if output_root not in destination.parents and destination != output_root:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
        archive.extractall(output_dir)

    return dataset_dir


def main() -> None:
    args = parse_args()
    output_dir = (Path(__file__).resolve().parent / args.output).resolve()
    dataset_dir = download_coco128(output_dir)
    image_count = len(list((dataset_dir / "images" / "train2017").glob("*.jpg")))
    label_count = len(list((dataset_dir / "labels" / "train2017").glob("*.txt")))
    print(f"Dataset: {dataset_dir}")
    print(f"Images : {image_count}")
    print(f"Labels : {label_count}")


if __name__ == "__main__":
    main()
