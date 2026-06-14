using OpenVinoSharp;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using YoloDetector.Wpf.Models;

namespace YoloDetector.Wpf.Services;

public sealed class OpenVinoYoloService : IOpenVinoYoloService
{
    private readonly object _sync = new();
    private Core? _core;
    private Model? _model;
    private CompiledModel? _compiledModel;
    private InferRequest? _inferRequest;
    private string[] _labels = [];
    private int _inputWidth;
    private int _inputHeight;
    private bool _isWarmedUp;

    public string LoadedDevice { get; private set; } = string.Empty;
    public string ModelDescription { get; private set; } = string.Empty;

    public Task LoadModelAsync(string modelPath, string device)
    {
        return Task.Run(() =>
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException("OpenVINO XML 모델을 찾을 수 없습니다.", modelPath);
            }

            lock (_sync)
            {
                DisposeModel();
                _core ??= new Core();

                List<string> devices = _core.get_available_devices();
                if (!devices.Any(item =>
                        item.Equals(device, StringComparison.OrdinalIgnoreCase) ||
                        item.StartsWith(device + ".", StringComparison.OrdinalIgnoreCase)))
                {
                    throw new InvalidOperationException(
                        $"{device} 장치를 사용할 수 없습니다. 사용 가능: {string.Join(", ", devices)}");
                }

                _model = _core.read_model(modelPath);
                // The wrapper treats an empty dictionary as an unsupported property count.
                _compiledModel = _core.compile_model(_model, device);
                _inferRequest = _compiledModel.create_infer_request();

                using Input input = _compiledModel.input();
                using Shape inputShape = input.get_shape();
                long[] dimensions = inputShape.shape.get_dims();
                if (dimensions.Length != 4)
                {
                    throw new InvalidOperationException(
                        $"NCHW 4차원 입력 모델만 지원합니다: {inputShape.to_string()}");
                }

                _inputHeight = checked((int)dimensions[2]);
                _inputWidth = checked((int)dimensions[3]);
                ModelMetadata metadata = LoadMetadata(modelPath);
                _labels = metadata.Labels.Count > 0 ? [.. metadata.Labels] : CocoLabels;
                LoadedDevice = device;
                _isWarmedUp = false;
                ModelDescription = string.IsNullOrWhiteSpace(metadata.Optimization)
                    ? Path.GetFileNameWithoutExtension(modelPath)
                    : $"{metadata.ModelName} / {metadata.Optimization}";
            }
        });
    }

    public Task<InferenceResult> DetectAsync(
        BitmapSource image,
        float confidenceThreshold,
        float iouThreshold)
    {
        int imageWidth = image.PixelWidth;
        int imageHeight = image.PixelHeight;
        byte[] sourcePixels = CopyBgraPixels(image);

        return Task.Run(() =>
        {
            lock (_sync)
            {
                if (_inferRequest is null)
                {
                    throw new InvalidOperationException("먼저 OpenVINO 모델을 로드하세요.");
                }

                PreprocessResult preprocessed = Preprocess(
                    sourcePixels, imageWidth, imageHeight, _inputWidth, _inputHeight);

                using Tensor inputTensor = _inferRequest.get_input_tensor();
                inputTensor.set_data(preprocessed.Tensor);

                if (!_isWarmedUp)
                {
                    _inferRequest.infer();
                    _isWarmedUp = true;
                }
                var stopwatch = Stopwatch.StartNew();
                _inferRequest.infer();
                stopwatch.Stop();

                using Tensor outputTensor = _inferRequest.get_output_tensor();
                using Shape outputShape = outputTensor.get_shape();
                long[] shape = outputShape.shape.get_dims();
                float[] output = outputTensor.get_float_data(checked((int)outputTensor.get_size()));

                IReadOnlyList<Detection> detections = Decode(
                    output,
                    shape,
                    confidenceThreshold,
                    iouThreshold,
                    preprocessed,
                    imageWidth,
                    imageHeight);
                double latencyMs = stopwatch.Elapsed.TotalMilliseconds;

                return new InferenceResult(
                    detections,
                    latencyMs,
                    latencyMs > 0 ? 1000.0 / latencyMs : 0,
                    imageWidth,
                    imageHeight);
            }
        });
    }

    public Task<BenchmarkResult> BenchmarkAsync(
        BitmapSource image,
        int runs,
        float confidenceThreshold,
        float iouThreshold)
    {
        if (runs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(runs));
        }

        int imageWidth = image.PixelWidth;
        int imageHeight = image.PixelHeight;
        byte[] sourcePixels = CopyBgraPixels(image);

        return Task.Run(() =>
        {
            lock (_sync)
            {
                if (_inferRequest is null)
                {
                    throw new InvalidOperationException("먼저 OpenVINO 모델을 로드하세요.");
                }

                PreprocessResult preprocessed = Preprocess(
                    sourcePixels, imageWidth, imageHeight, _inputWidth, _inputHeight);
                using Tensor inputTensor = _inferRequest.get_input_tensor();
                inputTensor.set_data(preprocessed.Tensor);

                _inferRequest.infer();
                var latencies = new double[runs];
                double detectionCountTotal = 0;
                double confidenceTotal = 0;
                int confidenceCount = 0;
                IReadOnlyList<Detection> lastDetections = [];

                for (int index = 0; index < runs; index++)
                {
                    var stopwatch = Stopwatch.StartNew();
                    _inferRequest.infer();
                    stopwatch.Stop();
                    latencies[index] = stopwatch.Elapsed.TotalMilliseconds;

                    using Tensor outputTensor = _inferRequest.get_output_tensor();
                    using Shape outputShape = outputTensor.get_shape();
                    long[] shape = outputShape.shape.get_dims();
                    float[] output = outputTensor.get_float_data(
                        checked((int)outputTensor.get_size()));
                    lastDetections = Decode(
                        output,
                        shape,
                        confidenceThreshold,
                        iouThreshold,
                        preprocessed,
                        imageWidth,
                        imageHeight);
                    detectionCountTotal += lastDetections.Count;
                    confidenceTotal += lastDetections.Sum(item => item.Confidence);
                    confidenceCount += lastDetections.Count;
                }

                double average = latencies.Average();
                double variance = latencies.Average(value => Math.Pow(value - average, 2));
                return new BenchmarkResult(
                    runs,
                    average,
                    latencies.Min(),
                    latencies.Max(),
                    Math.Sqrt(variance),
                    average > 0 ? 1000.0 / average : 0,
                    detectionCountTotal / runs,
                    confidenceCount > 0 ? confidenceTotal / confidenceCount : 0,
                    lastDetections);
            }
        });
    }

    private IReadOnlyList<Detection> Decode(
        float[] output,
        long[] shape,
        float confidenceThreshold,
        float iouThreshold,
        PreprocessResult preprocessed,
        int imageWidth,
        int imageHeight)
    {
        if (shape.Length != 3)
        {
            throw new InvalidOperationException(
                $"지원하지 않는 YOLO 출력 shape: [{string.Join(", ", shape)}]");
        }

        int first = checked((int)shape[1]);
        int second = checked((int)shape[2]);
        bool transposed = first <= 128 && first < second;
        int rows = transposed ? second : first;
        int columns = transposed ? first : second;
        if (columns < 6)
        {
            throw new InvalidOperationException("YOLOv5 detection 출력을 해석할 수 없습니다.");
        }

        var candidates = new List<Candidate>();
        for (int row = 0; row < rows; row++)
        {
            float objectness = ValueAt(output, row, 4, rows, columns, transposed);
            int classId = 0;
            float bestClassScore = 0;
            for (int column = 5; column < columns; column++)
            {
                float classScore = ValueAt(output, row, column, rows, columns, transposed);
                if (classScore > bestClassScore)
                {
                    bestClassScore = classScore;
                    classId = column - 5;
                }
            }

            float score = objectness * bestClassScore;
            if (score < confidenceThreshold)
            {
                continue;
            }

            float centerX = ValueAt(output, row, 0, rows, columns, transposed);
            float centerY = ValueAt(output, row, 1, rows, columns, transposed);
            float width = ValueAt(output, row, 2, rows, columns, transposed);
            float height = ValueAt(output, row, 3, rows, columns, transposed);
            candidates.Add(new Candidate(
                classId,
                score,
                centerX - width / 2,
                centerY - height / 2,
                centerX + width / 2,
                centerY + height / 2));
        }

        var selected = new List<Candidate>();
        foreach (IGrouping<int, Candidate> group in candidates.GroupBy(item => item.ClassId))
        {
            var ordered = group.OrderByDescending(item => item.Score).ToList();
            while (ordered.Count > 0)
            {
                Candidate best = ordered[0];
                selected.Add(best);
                ordered.RemoveAt(0);
                ordered.RemoveAll(item => IntersectionOverUnion(best, item) > iouThreshold);
            }
        }

        return selected
            .OrderByDescending(item => item.Score)
            .Select(item => new Detection(
                item.ClassId,
                item.ClassId < _labels.Length ? _labels[item.ClassId] : item.ClassId.ToString(),
                item.Score,
                Clamp((item.X1 - preprocessed.PaddingX) / preprocessed.Scale, 0, imageWidth),
                Clamp((item.Y1 - preprocessed.PaddingY) / preprocessed.Scale, 0, imageHeight),
                Clamp((item.X2 - preprocessed.PaddingX) / preprocessed.Scale, 0, imageWidth),
                Clamp((item.Y2 - preprocessed.PaddingY) / preprocessed.Scale, 0, imageHeight)))
            .ToList();
    }

    private static float ValueAt(
        float[] data, int row, int column, int rows, int columns, bool transposed)
    {
        return transposed
            ? data[column * rows + row]
            : data[row * columns + column];
    }

    private static float IntersectionOverUnion(Candidate first, Candidate second)
    {
        float left = Math.Max(first.X1, second.X1);
        float top = Math.Max(first.Y1, second.Y1);
        float right = Math.Min(first.X2, second.X2);
        float bottom = Math.Min(first.Y2, second.Y2);
        float intersection = Math.Max(0, right - left) * Math.Max(0, bottom - top);
        float firstArea = Math.Max(0, first.X2 - first.X1) * Math.Max(0, first.Y2 - first.Y1);
        float secondArea = Math.Max(0, second.X2 - second.X1) * Math.Max(0, second.Y2 - second.Y1);
        return intersection / Math.Max(firstArea + secondArea - intersection, 1e-7f);
    }

    private static byte[] CopyBgraPixels(BitmapSource source)
    {
        BitmapSource converted = source.Format == PixelFormats.Bgra32
            ? source
            : new FormatConvertedBitmap(source, PixelFormats.Bgra32, null, 0);
        int stride = converted.PixelWidth * 4;
        byte[] pixels = new byte[stride * converted.PixelHeight];
        converted.CopyPixels(pixels, stride, 0);
        return pixels;
    }

    private static PreprocessResult Preprocess(
        byte[] source, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
    {
        float scale = Math.Min(
            (float)targetWidth / sourceWidth,
            (float)targetHeight / sourceHeight);
        int resizedWidth = (int)MathF.Round(sourceWidth * scale);
        int resizedHeight = (int)MathF.Round(sourceHeight * scale);
        int paddingX = (targetWidth - resizedWidth) / 2;
        int paddingY = (targetHeight - resizedHeight) / 2;
        int planeSize = targetWidth * targetHeight;
        float[] tensor = Enumerable.Repeat(114f / 255f, planeSize * 3).ToArray();

        for (int y = 0; y < resizedHeight; y++)
        {
            float sourceY = (y + 0.5f) / scale - 0.5f;
            int nearestY = Math.Clamp((int)MathF.Round(sourceY), 0, sourceHeight - 1);
            for (int x = 0; x < resizedWidth; x++)
            {
                float sourceX = (x + 0.5f) / scale - 0.5f;
                int nearestX = Math.Clamp((int)MathF.Round(sourceX), 0, sourceWidth - 1);
                int sourceIndex = (nearestY * sourceWidth + nearestX) * 4;
                int targetIndex = (y + paddingY) * targetWidth + x + paddingX;
                tensor[targetIndex] = source[sourceIndex + 2] / 255f;
                tensor[planeSize + targetIndex] = source[sourceIndex + 1] / 255f;
                tensor[planeSize * 2 + targetIndex] = source[sourceIndex] / 255f;
            }
        }

        return new PreprocessResult(tensor, scale, paddingX, paddingY);
    }

    private static ModelMetadata LoadMetadata(string modelPath)
    {
        string metadataPath = Path.ChangeExtension(modelPath, ".json");
        if (!File.Exists(metadataPath))
        {
            return new ModelMetadata();
        }

        return JsonSerializer.Deserialize<ModelMetadata>(
                   File.ReadAllText(metadataPath),
                   new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
               ?? new ModelMetadata();
    }

    private void DisposeModel()
    {
        _inferRequest?.Dispose();
        _compiledModel?.Dispose();
        _model?.Dispose();
        _inferRequest = null;
        _compiledModel = null;
        _model = null;
        _isWarmedUp = false;
    }

    public void Dispose()
    {
        lock (_sync)
        {
            DisposeModel();
            _core?.Dispose();
            _core = null;
        }
    }

    private static float Clamp(float value, float minimum, float maximum) =>
        Math.Min(Math.Max(value, minimum), maximum);

    private sealed record PreprocessResult(
        float[] Tensor, float Scale, float PaddingX, float PaddingY);

    private sealed record Candidate(
        int ClassId, float Score, float X1, float Y1, float X2, float Y2);

    private static readonly string[] CocoLabels =
    [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ];
}
