namespace YoloDetector.Wpf.Models;

public sealed record InferenceResult(
    IReadOnlyList<Detection> Detections,
    double LatencyMs,
    double Fps,
    int ImageWidth,
    int ImageHeight);
