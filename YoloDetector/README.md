# YOLOv5 OpenVINO WPF MVVM

Python은 학습된 YOLOv5 가중치를 다운로드하고 OpenVINO 모델을 생성합니다.
C# WPF 앱은 Python을 호출하지 않고 OpenVINO C API의 .NET 바인딩으로 모델을
직접 로드하고 추론합니다.

## 프로젝트 흐름

```text
Python
  YOLOv5 .pt 다운로드
    -> ONNX export
    -> OpenVINO FP32 (최적화 없음)
    -> OpenVINO FP16 (가중치 압축)
    -> OpenVINO INT8 (선택, NNCF 캘리브레이션)

C# WPF
  모델 로드 -> 이미지 로드 -> 전처리 -> OpenVINO 추론
  -> YOLOv5 decode/NMS -> 바운딩 박스 -> latency/FPS
```

## 1. Python 환경

`YoloDetector` 폴더에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python\requirements.txt
```

## 2. 모델 다운로드 및 변환

### FP32와 FP16

```powershell
python python\prepare_yolov5_openvino.py --model yolov5s --output ..\models
```

생성 구조:

```text
downloads/
  yolov5s.pt
models/
  yolov5s.onnx
  yolov5s_fp32.xml
  yolov5s_fp32.bin
  yolov5s_fp32.json
  yolov5s_fp16.xml
  yolov5s_fp16.bin
  yolov5s_fp16.json
  yolov5s_int8.xml
  yolov5s_int8.bin
  yolov5s_int8.json
test_images/
  bus.jpg
  zidane.jpg
validation_data/
  coco128/
    images/train2017/*.jpg
    labels/train2017/*.txt
```

- `*_fp32`: OpenVINO 변환만 수행한 FP32 비교 기준 모델
- `*_fp16`: 가중치를 FP16으로 압축한 최적화 모델
- `*_int8`: 캘리브레이션 데이터로 양자화한 INT8 모델
- `test_images`: YOLOv5 공식 저장소의 대표 검출 이미지
- `validation_data/coco128`: 128개 이미지와 YOLO 형식 정답 bounding-box 라벨
- `.json`: 클래스 이름 및 최적화 종류

`--model`에는 `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`를 사용할 수
있습니다.

기본 실행 시 COCO128 검증 데이터도 자동 다운로드됩니다. 모델만 준비하고
검증 데이터는 받지 않으려면 `--skip-validation-data`를 추가합니다.

검증 데이터만 별도로 받으려면 다음 명령을 사용합니다.

```powershell
python python\download_validation_data.py
```

YOLO 라벨 한 줄의 형식은 다음과 같습니다.

```text
class_id center_x center_y width height
```

좌표는 이미지 크기를 기준으로 0~1 사이로 정규화되어 있습니다. 이 라벨을
모델 검출 결과와 IoU로 매칭하면 precision, recall, mAP를 계산할 수 있습니다.

### INT8 추가 생성

INT8은 실제 입력과 비슷한 캘리브레이션 이미지가 필요합니다.

```powershell
python python\prepare_yolov5_openvino.py `
  --model yolov5s `
  --output ..\models `
  --calibration-dir "C:\dataset\calibration" `
  --calibration-count 100
```

결과는 `models\yolov5s_int8.xml`, `.bin`, `.json`으로 생성됩니다.

## 3. WPF 실행

Visual Studio 2022에서 `YoloDetector.sln`을 열고 `YoloDetector.Wpf`를 시작
프로젝트로 지정합니다.

필요 조건:

- .NET 8 Desktop Development
- x64 빌드
- Intel GPU를 선택할 경우 적절한 Intel 그래픽 드라이버

먼저 상단에서 `CPU` 또는 `GPU`를 선택하고 FP32, FP16, INT8 중 사용할 XML
모델을 로드합니다.

### Single Image Prediction

1. `이미지 로드`
2. `추론`

이미지와 검출 bounding box, 객체 이름, confidence score가 화면에 표시됩니다.

### Benchmark

1. `폴더 선택`에서 `validation_data\coco128` 루트 폴더 선택
2. `Benchmark 실행`

앱이 하위 `images`와 `labels` 폴더를 자동 탐색합니다. 모든 validation 이미지를
순서대로 로드하고 추론하며, 현재 이미지와 bounding box를 Benchmark 화면에
계속 표시합니다.

평가가 끝나면 선택한 COCO128 폴더 바로 아래의 `benchmark.csv`에 다음 순서로
결과가 저장됩니다.

```text
model_name,device,latency_average_ms,precision,recall,f1_score,iou
```

Precision, Recall, F1-score와 IoU는 confidence 0.25, matching IoU 0.50 기준입니다.
같은 모델 또는 다른 FP32/FP16/INT8 모델과 CPU/GPU 조합으로 다시 실행하면
`benchmark.csv`에 결과가 새 행으로 계속 추가됩니다.

모델 장치를 변경한 경우 선택한 장치에 맞게 모델을 다시 로드해야 합니다.
표시 latency는 장치 초기화를 위한 warm-up 1회 이후의 순수 추론 시간이며,
FPS는 `1000 / latency_ms`입니다.

## MVVM 구조

```text
YoloDetector.Wpf/
  Models/       검출 결과와 모델 메타데이터
  ViewModels/   화면 상태, LoadModel/LoadImage/Detect 명령
  Services/     파일 대화상자와 OpenVINO 직접 추론
  Controls/     바운딩 박스 렌더링 전용 컨트롤
  MainWindow    XAML 바인딩과 ViewModel 조립
```

핵심 추론 구현은
`YoloDetector.Wpf\Services\OpenVinoYoloService.cs`에 있습니다. 이 서비스가
OpenVINO 모델 컴파일, NCHW 전처리, 추론, YOLOv5 decode와 NMS를 수행합니다.

OpenVINO는 공식 C# API를 제공하지 않으므로 앱은 공식 OpenVINO C API를 감싸는
`OpenVINO.CSharp.API`와 Windows 네이티브 런타임 NuGet 패키지를 사용합니다.
