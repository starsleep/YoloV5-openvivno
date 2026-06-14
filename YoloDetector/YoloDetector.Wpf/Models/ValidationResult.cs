namespace YoloDetector.Wpf.Models;

public sealed record ValidationResult(
    int ImageCount,
    int LabelFileCount,
    int GroundTruthCount,
    int PredictionCount,
    double Precision,
    double Recall,
    double F1Score,
    double MeanMatchedIou,
    double Map50,
    double Map50To95,
    double AverageLatencyMs,
    double MinimumLatencyMs,
    double MaximumLatencyMs,
    double AverageFps,
    double TotalSeconds);
