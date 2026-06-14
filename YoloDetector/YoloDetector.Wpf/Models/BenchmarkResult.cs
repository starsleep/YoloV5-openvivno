namespace YoloDetector.Wpf.Models;

public sealed record BenchmarkResult(
    int Runs,
    double AverageLatencyMs,
    double MinimumLatencyMs,
    double MaximumLatencyMs,
    double LatencyStandardDeviationMs,
    double AverageFps,
    double AverageDetectionCount,
    double AverageConfidence,
    IReadOnlyList<Detection> LastDetections);
