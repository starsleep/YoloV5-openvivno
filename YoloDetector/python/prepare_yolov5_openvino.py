"""Download a trained YOLOv5 model and create OpenVINO optimization variants."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import openvino as ov
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEST_IMAGES = {
    "bus.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "zidane.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
}
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YOLOv5 and export FP32, FP16, and optional INT8 OpenVINO IR models."
    )
    parser.add_argument(
        "--model",
        default="yolov5s",
        choices=["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"],
    )
    parser.add_argument("--output", default="../models", help="OpenVINO model root")
    parser.add_argument("--downloads", default="../downloads", help="Downloaded .pt storage")
    parser.add_argument("--test-images", default="../test_images", help="YOLOv5 sample images")
    parser.add_argument(
        "--validation-data",
        default="../validation_data",
        help="Directory for COCO128 images and ground-truth labels",
    )
    parser.add_argument(
        "--skip-validation-data",
        action="store_true",
        help="Do not download the COCO128 validation dataset",
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument(
        "--calibration-dir",
        help="Image directory used for INT8 post-training quantization",
    )
    parser.add_argument("--calibration-count", type=int, default=100)
    return parser.parse_args()


def download_and_load(model_name: str, downloads_dir: Path):
    downloads_dir.mkdir(parents=True, exist_ok=True)
    weights_path = downloads_dir / f"{model_name}.pt"

    previous_cwd = Path.cwd()
    try:
        os.chdir(downloads_dir)
        loaded_model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(weights_path),
            autoshape=False,
            trust_repo=True,
        )
    finally:
        os.chdir(previous_cwd)

    model = loaded_model.model if hasattr(loaded_model, "model") else loaded_model
    model = model.float().cpu().eval()
    for module in model.modules():
        if module.__class__.__name__ == "Detect":
            module.inplace = False
            module.dynamic = False
            module.export = True
    return model, weights_path


def export_onnx(model, path: Path, image_size: int) -> None:
    dummy = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["images"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
    )


def save_variant(
    model: ov.Model,
    variant_dir: Path,
    model_name: str,
    labels: list[str],
    image_size: int,
    optimization: str,
    compress_to_fp16: bool,
    file_name: str,
) -> Path:
    variant_dir.mkdir(parents=True, exist_ok=True)
    xml_path = variant_dir / f"{file_name}.xml"
    ov.save_model(model, xml_path, compress_to_fp16=compress_to_fp16)

    metadata = {
        "model_name": model_name,
        "optimization": optimization,
        "input_size": image_size,
        "input_layout": "NCHW",
        "labels": labels,
    }
    xml_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return xml_path


def download_test_images(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for file_name, url in TEST_IMAGES.items():
        destination = output_dir / file_name
        if not destination.exists():
            print(f"  Downloading test image: {file_name}")
            urlretrieve(url, destination)
        downloaded.append(destination)
    return downloaded


def download_coco128(output_dir: Path) -> Path:
    dataset_dir = output_dir / "coco128"
    images_dir = dataset_dir / "images" / "train2017"
    labels_dir = dataset_dir / "labels" / "train2017"
    if images_dir.exists() and labels_dir.exists():
        return dataset_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "coco128.zip"
    if not archive_path.exists():
        print("  Downloading COCO128 validation images and labels...")
        urlretrieve(COCO128_URL, archive_path)

    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (output_dir / member.filename).resolve()
            if output_root not in destination.parents and destination != output_root:
                raise RuntimeError(f"Unsafe path in COCO128 archive: {member.filename}")
        archive.extractall(output_dir)

    if not images_dir.exists() or not labels_dir.exists():
        raise RuntimeError("COCO128 압축 해제 후 images/labels 폴더를 찾지 못했습니다.")
    return dataset_dir


def letterbox(image: np.ndarray, image_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0


def build_int8_model(
    fp32_model: ov.Model,
    calibration_dir: Path,
    image_size: int,
    calibration_count: int,
) -> ov.Model:
    try:
        import nncf
    except ImportError as exc:
        raise RuntimeError("INT8 변환에는 nncf 패키지가 필요합니다.") from exc

    image_paths = [
        path
        for path in calibration_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ][:calibration_count]
    if not image_paths:
        raise RuntimeError(f"캘리브레이션 이미지를 찾지 못했습니다: {calibration_dir}")

    def transform(path: Path) -> np.ndarray:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"이미지를 읽지 못했습니다: {path}")
        return letterbox(image, image_size)

    dataset = nncf.Dataset(image_paths, transform)
    return nncf.quantize(fp32_model, dataset)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_root = (script_dir / args.output).resolve()
    downloads_dir = (script_dir / args.downloads).resolve()
    test_images_dir = (script_dir / args.test_images).resolve()
    validation_data_dir = (script_dir / args.validation_data).resolve()

    print(f"[1/6] Downloading/loading {args.model}...")
    pytorch_model, weights_path = download_and_load(args.model, downloads_dir)
    labels = (
        list(pytorch_model.names.values())
        if isinstance(pytorch_model.names, dict)
        else list(pytorch_model.names)
    )

    output_root.mkdir(parents=True, exist_ok=True)
    onnx_path = output_root / f"{args.model}.onnx"

    print("[2/6] Downloading YOLOv5 test images...")
    test_images = download_test_images(test_images_dir)

    validation_dataset = None
    if not args.skip_validation_data:
        print("[3/6] Downloading COCO128 validation dataset...")
        validation_dataset = download_coco128(validation_data_dir)
    else:
        print("[3/6] COCO128 validation dataset skipped.")

    print("[4/6] Exporting ONNX...")
    export_onnx(pytorch_model, onnx_path, args.img_size)
    openvino_model = ov.convert_model(onnx_path)

    print("[5/6] Saving non-optimized FP32 and optimized FP16 models...")
    fp32_path = save_variant(
        openvino_model,
        output_root,
        args.model,
        labels,
        args.img_size,
        "none-fp32",
        False,
        f"{args.model}_fp32",
    )
    fp16_path = save_variant(
        openvino_model,
        output_root,
        args.model,
        labels,
        args.img_size,
        "weight-compression-fp16",
        True,
        f"{args.model}_fp16",
    )

    int8_path = None
    if args.calibration_dir:
        calibration_dir = Path(args.calibration_dir).resolve()
        print("[6/6] Quantizing INT8 with NNCF...")
        int8_model = build_int8_model(
            openvino_model, calibration_dir, args.img_size, args.calibration_count
        )
        int8_path = save_variant(
            int8_model,
            output_root,
            args.model,
            labels,
            args.img_size,
            "post-training-int8",
            False,
            f"{args.model}_int8",
        )
    else:
        print("[6/6] INT8 skipped (use --calibration-dir to enable).")

    print("\nCompleted")
    print(f"Downloaded weights : {weights_path}")
    print(f"FP32 baseline      : {fp32_path}")
    print(f"FP16 optimized     : {fp16_path}")
    print(f"Test images        : {', '.join(str(path) for path in test_images)}")
    if validation_dataset:
        print(f"Validation dataset : {validation_dataset}")
    if int8_path:
        print(f"INT8 optimized     : {int8_path}")


if __name__ == "__main__":
    main()
