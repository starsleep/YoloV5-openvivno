using Microsoft.Win32;
using System.Windows;

namespace YoloDetector.Wpf.Services;

public sealed class DialogService : IDialogService
{
    public string? SelectModel()
    {
        var dialog = new OpenFileDialog
        {
            Title = "OpenVINO IR 모델 선택",
            Filter = "OpenVINO model (*.xml)|*.xml|All files (*.*)|*.*"
        };
        return dialog.ShowDialog() == true ? dialog.FileName : null;
    }

    public string? SelectImage()
    {
        var dialog = new OpenFileDialog
        {
            Title = "추론 이미지 선택",
            Filter = "Image files|*.jpg;*.jpeg;*.png;*.bmp;*.webp|All files (*.*)|*.*"
        };
        return dialog.ShowDialog() == true ? dialog.FileName : null;
    }

    public string? SelectValidationImageFolder()
    {
        var dialog = new OpenFolderDialog
        {
            Title = "COCO128 루트 폴더 선택"
        };
        return dialog.ShowDialog() == true ? dialog.FolderName : null;
    }

    public void ShowError(string message)
    {
        MessageBox.Show(message, "YOLOv5 OpenVINO", MessageBoxButton.OK, MessageBoxImage.Error);
    }

    public void ShowInformation(string message)
    {
        MessageBox.Show(message, "YOLOv5 OpenVINO", MessageBoxButton.OK, MessageBoxImage.Information);
    }
}
