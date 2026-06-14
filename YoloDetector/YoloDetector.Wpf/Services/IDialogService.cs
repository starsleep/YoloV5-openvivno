namespace YoloDetector.Wpf.Services;

public interface IDialogService
{
    string? SelectModel();
    string? SelectImage();
    string? SelectValidationImageFolder();
    void ShowError(string message);
    void ShowInformation(string message);
}
