using YoloDetector.Wpf.Models;

namespace YoloDetector.Wpf.Services;

public interface IValidationService
{
    Task<ValidationResult> ValidateAsync(
        string cocoDatasetFolder,
        IOpenVinoYoloService detector,
        IProgress<ValidationProgress>? progress = null);
}
