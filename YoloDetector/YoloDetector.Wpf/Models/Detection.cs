namespace YoloDetector.Wpf.Models;

public sealed record Detection(
    int ClassId,
    string Label,
    float Confidence,
    float X1,
    float Y1,
    float X2,
    float Y2);
