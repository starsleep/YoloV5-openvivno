using System.Windows.Media.Imaging;
using YoloDetector.Wpf.Models;

namespace YoloDetector.Wpf.Services;

public interface IOpenVinoYoloService : IDisposable
{
    string LoadedDevice { get; }
    string ModelDescription { get; }

    Task LoadModelAsync(string modelPath, string device);

    Task<InferenceResult> DetectAsync(
        BitmapSource image,
        float confidenceThreshold,
        float iouThreshold);

    Task<BenchmarkResult> BenchmarkAsync(
        BitmapSource image,
        int runs,
        float confidenceThreshold,
        float iouThreshold);
}
