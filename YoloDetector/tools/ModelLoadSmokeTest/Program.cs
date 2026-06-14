using System;
using System.IO;
using System.Linq;
using System.Windows.Media.Imaging;
using YoloDetector.Wpf.Services;

string modelPath = args[0];
using var service = new OpenVinoYoloService();
await service.LoadModelAsync(modelPath, "CPU");
Console.WriteLine($"loaded: {service.ModelDescription} / {service.LoadedDevice}");

if (args.Length > 1)
{
    if (args.Length > 2 && args[2].Equals("validate", StringComparison.OrdinalIgnoreCase))
    {
        var validation = new YoloValidationService();
        var progress = new Progress<YoloDetector.Wpf.Models.ValidationProgress>(item =>
            Console.WriteLine(
                $"{item.Current}/{item.Total}: {Path.GetFileName(item.ImagePath)} " +
                $"({item.Detections.Count} detections)"));
        var metrics = await validation.ValidateAsync(args[1], service, progress);
        Console.WriteLine(
            $"validation: {metrics.ImageCount} images, precision {metrics.Precision:P2}, " +
            $"recall {metrics.Recall:P2}, mAP50 {metrics.Map50:P2}, " +
            $"mAP50-95 {metrics.Map50To95:P2}, IoU {metrics.MeanMatchedIou:P2}, " +
            $"{metrics.AverageLatencyMs:F2} ms");
        return;
    }

    var image = new BitmapImage();
    image.BeginInit();
    image.CacheOption = BitmapCacheOption.OnLoad;
    image.UriSource = new Uri(Path.GetFullPath(args[1]));
    image.EndInit();
    image.Freeze();

    if (args.Length > 2 && args[2].Equals("benchmark", StringComparison.OrdinalIgnoreCase))
    {
        var benchmark = await service.BenchmarkAsync(image, 100, 0.25f, 0.45f);
        Console.WriteLine(
            $"benchmark: {benchmark.Runs} runs, avg {benchmark.AverageLatencyMs:F2} ms, " +
            $"{benchmark.AverageFps:F2} FPS, confidence {benchmark.AverageConfidence:P2}");
        return;
    }

    var result = await service.DetectAsync(image, 0.25f, 0.45f);
    Console.WriteLine(
        $"detections: {result.Detections.Count}, latency: {result.LatencyMs:F1} ms, fps: {result.Fps:F1}");
    foreach (var detection in result.Detections.Take(5))
    {
        Console.WriteLine($"{detection.Label}: {detection.Confidence:P1}");
    }
}
