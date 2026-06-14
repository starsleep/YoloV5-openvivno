using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Windows.Media.Imaging;
using YoloDetector.Wpf.Models;

namespace YoloDetector.Wpf.Services;

public sealed class YoloValidationService : IValidationService
{
    private static readonly HashSet<string> ImageExtensions =
        new(StringComparer.OrdinalIgnoreCase) { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };

    public async Task<ValidationResult> ValidateAsync(
        string cocoDatasetFolder,
        IOpenVinoYoloService detector,
        IProgress<ValidationProgress>? progress = null)
    {
        string datasetPath = Path.GetFullPath(cocoDatasetFolder);
        if (!Directory.Exists(datasetPath))
        {
            throw new DirectoryNotFoundException($"COCO 데이터셋 폴더를 찾을 수 없습니다: {datasetPath}");
        }

        (string imagesPath, string labelsPath) = ResolveDatasetFolders(datasetPath);

        string[] imagePaths = Directory
            .EnumerateFiles(imagesPath, "*.*", SearchOption.AllDirectories)
            .Where(path => ImageExtensions.Contains(Path.GetExtension(path)))
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (imagePaths.Length == 0)
        {
            throw new InvalidOperationException("선택한 폴더에 검증 이미지가 없습니다.");
        }

        var groundTruths = new List<GroundTruth>();
        var predictions = new List<Prediction>();
        var latencies = new List<double>(imagePaths.Length);
        int labelFileCount = 0;
        var totalTimer = Stopwatch.StartNew();

        for (int index = 0; index < imagePaths.Length; index++)
        {
            string imagePath = imagePaths[index];
            string imageId = Path.GetRelativePath(imagesPath, imagePath);
            BitmapSource image = LoadBitmap(imagePath);
            string relativeLabel = Path.ChangeExtension(imageId, ".txt");
            string labelPath = Path.Combine(labelsPath, relativeLabel);
            if (File.Exists(labelPath))
            {
                labelFileCount++;
                groundTruths.AddRange(ReadLabels(
                    labelPath, imageId, image.PixelWidth, image.PixelHeight));
            }

            InferenceResult result = await detector.DetectAsync(image, 0.001f, 0.45f);
            latencies.Add(result.LatencyMs);
            predictions.AddRange(result.Detections.Select(detection =>
                new Prediction(imageId, detection)));
            progress?.Report(new ValidationProgress(
                index + 1,
                imagePaths.Length,
                imagePath,
                image,
                result.Detections.Where(item => item.Confidence >= 0.25f).ToList()));
        }

        totalTimer.Stop();
        OperatingMetrics operating = CalculateOperatingMetrics(
            predictions.Where(item => item.Detection.Confidence >= 0.25f).ToList(),
            groundTruths,
            0.5);
        double map50 = CalculateMap(predictions, groundTruths, [0.5]);
        double map50To95 = CalculateMap(
            predictions,
            groundTruths,
            Enumerable.Range(0, 10).Select(index => 0.5 + index * 0.05).ToArray());
        double averageLatency = latencies.Average();

        return new ValidationResult(
            imagePaths.Length,
            labelFileCount,
            groundTruths.Count,
            predictions.Count,
            operating.Precision,
            operating.Recall,
            operating.F1,
            operating.MeanIou,
            map50,
            map50To95,
            averageLatency,
            latencies.Min(),
            latencies.Max(),
            averageLatency > 0 ? 1000.0 / averageLatency : 0,
            totalTimer.Elapsed.TotalSeconds);
    }

    private static (string ImagesPath, string LabelsPath) ResolveDatasetFolders(
        string datasetPath)
    {
        DirectoryInfo root = new(datasetPath);
        DirectoryInfo? imagesRoot = root.Name.Equals("images", StringComparison.OrdinalIgnoreCase)
            ? root
            : root.EnumerateDirectories("images", SearchOption.AllDirectories)
                .OrderBy(directory => directory.FullName.Length)
                .FirstOrDefault();
        DirectoryInfo? labelsRoot = root.Name.Equals("labels", StringComparison.OrdinalIgnoreCase)
            ? root
            : root.EnumerateDirectories("labels", SearchOption.AllDirectories)
                .OrderBy(directory => directory.FullName.Length)
                .FirstOrDefault();
        if (imagesRoot is null || labelsRoot is null)
        {
            throw new DirectoryNotFoundException(
                "선택한 폴더 내부에서 images와 labels 폴더를 찾지 못했습니다.");
        }

        string imagesPath = SelectDataLeaf(imagesRoot);
        string relative = Path.GetRelativePath(imagesRoot.FullName, imagesPath);
        string labelsPath = Path.Combine(labelsRoot.FullName, relative);
        if (!Directory.Exists(labelsPath))
        {
            labelsPath = labelsRoot.FullName;
        }
        return (imagesPath, labelsPath);
    }

    private static string SelectDataLeaf(DirectoryInfo root)
    {
        if (root.EnumerateFiles().Any(file => ImageExtensions.Contains(file.Extension)))
        {
            return root.FullName;
        }

        DirectoryInfo? leaf = root
            .EnumerateDirectories("*", SearchOption.AllDirectories)
            .FirstOrDefault(directory =>
                directory.EnumerateFiles().Any(file => ImageExtensions.Contains(file.Extension)));
        return leaf?.FullName ?? root.FullName;
    }

    private static BitmapSource LoadBitmap(string path)
    {
        var bitmap = new BitmapImage();
        bitmap.BeginInit();
        bitmap.CacheOption = BitmapCacheOption.OnLoad;
        bitmap.UriSource = new Uri(path, UriKind.Absolute);
        bitmap.EndInit();
        bitmap.Freeze();
        return bitmap;
    }

    private static IEnumerable<GroundTruth> ReadLabels(
        string labelPath,
        string imageId,
        int imageWidth,
        int imageHeight)
    {
        foreach (string line in File.ReadLines(labelPath))
        {
            string[] parts = line.Split(
                (char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 5)
            {
                continue;
            }

            int classId = int.Parse(parts[0], CultureInfo.InvariantCulture);
            double centerX = double.Parse(parts[1], CultureInfo.InvariantCulture) * imageWidth;
            double centerY = double.Parse(parts[2], CultureInfo.InvariantCulture) * imageHeight;
            double width = double.Parse(parts[3], CultureInfo.InvariantCulture) * imageWidth;
            double height = double.Parse(parts[4], CultureInfo.InvariantCulture) * imageHeight;
            yield return new GroundTruth(
                imageId,
                classId,
                centerX - width / 2,
                centerY - height / 2,
                centerX + width / 2,
                centerY + height / 2);
        }
    }

    private static OperatingMetrics CalculateOperatingMetrics(
        IReadOnlyList<Prediction> predictions,
        IReadOnlyList<GroundTruth> groundTruths,
        double iouThreshold)
    {
        MatchResult match = MatchPredictions(predictions, groundTruths, iouThreshold);
        double precision = match.TruePositive + match.FalsePositive > 0
            ? (double)match.TruePositive / (match.TruePositive + match.FalsePositive)
            : 0;
        double recall = groundTruths.Count > 0
            ? (double)match.TruePositive / groundTruths.Count
            : 0;
        double f1 = precision + recall > 0
            ? 2 * precision * recall / (precision + recall)
            : 0;
        return new OperatingMetrics(precision, recall, f1, match.MeanIou);
    }

    private static double CalculateMap(
        IReadOnlyList<Prediction> predictions,
        IReadOnlyList<GroundTruth> groundTruths,
        IReadOnlyList<double> iouThresholds)
    {
        int[] classes = groundTruths.Select(item => item.ClassId).Distinct().ToArray();
        if (classes.Length == 0)
        {
            return 0;
        }

        var averagePrecisions = new List<double>();
        foreach (double threshold in iouThresholds)
        {
            foreach (int classId in classes)
            {
                List<GroundTruth> classTruths =
                    groundTruths.Where(item => item.ClassId == classId).ToList();
                List<Prediction> classPredictions = predictions
                    .Where(item => item.Detection.ClassId == classId)
                    .OrderByDescending(item => item.Detection.Confidence)
                    .ToList();
                averagePrecisions.Add(CalculateAveragePrecision(
                    classPredictions, classTruths, threshold));
            }
        }

        return averagePrecisions.Average();
    }

    private static double CalculateAveragePrecision(
        IReadOnlyList<Prediction> predictions,
        IReadOnlyList<GroundTruth> groundTruths,
        double iouThreshold)
    {
        if (groundTruths.Count == 0)
        {
            return 0;
        }

        var matched = new HashSet<GroundTruth>();
        var truePositives = new double[predictions.Count];
        var falsePositives = new double[predictions.Count];
        for (int index = 0; index < predictions.Count; index++)
        {
            Prediction prediction = predictions[index];
            GroundTruth? best = groundTruths
                .Where(item => item.ImageId == prediction.ImageId && !matched.Contains(item))
                .Select(item => new { Item = item, Iou = IoU(prediction.Detection, item) })
                .Where(item => item.Iou >= iouThreshold)
                .OrderByDescending(item => item.Iou)
                .Select(item => item.Item)
                .FirstOrDefault();
            if (best is not null)
            {
                matched.Add(best);
                truePositives[index] = 1;
            }
            else
            {
                falsePositives[index] = 1;
            }
        }

        double cumulativeTp = 0;
        double cumulativeFp = 0;
        var recalls = new double[predictions.Count];
        var precisions = new double[predictions.Count];
        for (int index = 0; index < predictions.Count; index++)
        {
            cumulativeTp += truePositives[index];
            cumulativeFp += falsePositives[index];
            recalls[index] = cumulativeTp / groundTruths.Count;
            precisions[index] = cumulativeTp / Math.Max(cumulativeTp + cumulativeFp, 1);
        }

        double ap = 0;
        for (int point = 0; point <= 100; point++)
        {
            double recallLevel = point / 100.0;
            double precision = recalls
                .Select((recall, index) => new { recall, precision = precisions[index] })
                .Where(item => item.recall >= recallLevel)
                .Select(item => item.precision)
                .DefaultIfEmpty(0)
                .Max();
            ap += precision / 101.0;
        }
        return ap;
    }

    private static MatchResult MatchPredictions(
        IReadOnlyList<Prediction> predictions,
        IReadOnlyList<GroundTruth> groundTruths,
        double iouThreshold)
    {
        var matched = new HashSet<GroundTruth>();
        int truePositive = 0;
        int falsePositive = 0;
        double matchedIouTotal = 0;
        foreach (Prediction prediction in predictions.OrderByDescending(
                     item => item.Detection.Confidence))
        {
            var best = groundTruths
                .Where(item =>
                    item.ImageId == prediction.ImageId &&
                    item.ClassId == prediction.Detection.ClassId &&
                    !matched.Contains(item))
                .Select(item => new { Item = item, Iou = IoU(prediction.Detection, item) })
                .Where(item => item.Iou >= iouThreshold)
                .OrderByDescending(item => item.Iou)
                .FirstOrDefault();
            if (best is not null)
            {
                matched.Add(best.Item);
                truePositive++;
                matchedIouTotal += best.Iou;
            }
            else
            {
                falsePositive++;
            }
        }
        return new MatchResult(
            truePositive,
            falsePositive,
            truePositive > 0 ? matchedIouTotal / truePositive : 0);
    }

    private static double IoU(Detection detection, GroundTruth truth)
    {
        double left = Math.Max(detection.X1, truth.X1);
        double top = Math.Max(detection.Y1, truth.Y1);
        double right = Math.Min(detection.X2, truth.X2);
        double bottom = Math.Min(detection.Y2, truth.Y2);
        double intersection = Math.Max(0, right - left) * Math.Max(0, bottom - top);
        double detectionArea =
            Math.Max(0, detection.X2 - detection.X1) * Math.Max(0, detection.Y2 - detection.Y1);
        double truthArea = Math.Max(0, truth.X2 - truth.X1) * Math.Max(0, truth.Y2 - truth.Y1);
        return intersection / Math.Max(detectionArea + truthArea - intersection, 1e-9);
    }

    private sealed record GroundTruth(
        string ImageId, int ClassId, double X1, double Y1, double X2, double Y2);
    private sealed record Prediction(string ImageId, Detection Detection);
    private sealed record MatchResult(int TruePositive, int FalsePositive, double MeanIou);
    private sealed record OperatingMetrics(
        double Precision, double Recall, double F1, double MeanIou);
}
