using System.Windows.Media.Imaging;

namespace YoloDetector.Wpf.Models;

public sealed record ValidationProgress(
    int Current,
    int Total,
    string ImagePath,
    BitmapSource Image,
    IReadOnlyList<Detection> Detections);
