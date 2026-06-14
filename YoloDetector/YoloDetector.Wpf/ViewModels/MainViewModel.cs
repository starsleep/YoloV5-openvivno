using System.IO;
using System.Globalization;
using System.Text;
using System.Windows.Media.Imaging;
using YoloDetector.Wpf.Models;
using YoloDetector.Wpf.Services;

namespace YoloDetector.Wpf.ViewModels;

public sealed class MainViewModel : ObservableObject, IDisposable
{
    private readonly IOpenVinoYoloService _detector;
    private readonly IDialogService _dialogs;
    private readonly IValidationService _validationService;
    private string _selectedDevice = "CPU";
    private string _modelPath = string.Empty;
    private string _imagePath = string.Empty;
    private string _validationImageFolder = string.Empty;
    private string _status = "모델을 로드하세요.";
    private string _latencyText = "- ms";
    private string _fpsText = "- FPS";
    private BitmapSource? _sourceImage;
    private IReadOnlyList<Detection> _detections = [];
    private bool _isModelLoaded;
    private bool _isBusy;

    public MainViewModel(
        IOpenVinoYoloService detector,
        IDialogService dialogs,
        IValidationService validationService)
    {
        _detector = detector;
        _dialogs = dialogs;
        _validationService = validationService;
        LoadModelCommand = new AsyncRelayCommand(LoadModelAsync, () => !IsBusy);
        LoadImageCommand = new RelayCommand(LoadImage, () => !IsBusy);
        DetectCommand = new AsyncRelayCommand(
            DetectAsync,
            () => IsModelLoaded && SourceImage is not null && !IsBusy);
        SelectValidationFolderCommand = new RelayCommand(
            SelectValidationFolder, () => !IsBusy);
        ValidateFolderCommand = new AsyncRelayCommand(
            ValidateFolderAsync,
            () => IsModelLoaded && Directory.Exists(ValidationImageFolder) && !IsBusy);
    }

    public IReadOnlyList<string> Devices { get; } = ["CPU", "GPU"];

    public string SelectedDevice
    {
        get => _selectedDevice;
        set
        {
            if (!SetProperty(ref _selectedDevice, value))
            {
                return;
            }

            if (IsModelLoaded && !_detector.LoadedDevice.Equals(value, StringComparison.OrdinalIgnoreCase))
            {
                IsModelLoaded = false;
                Status = "장치가 변경되었습니다. 모델을 다시 로드하세요.";
            }
        }
    }

    public string ModelPath
    {
        get => _modelPath;
        private set => SetProperty(ref _modelPath, value);
    }

    public string ImagePath
    {
        get => _imagePath;
        private set => SetProperty(ref _imagePath, value);
    }

    public string ValidationImageFolder
    {
        get => _validationImageFolder;
        private set
        {
            if (SetProperty(ref _validationImageFolder, value))
            {
                RaiseCommandStates();
            }
        }
    }

    public string Status
    {
        get => _status;
        private set => SetProperty(ref _status, value);
    }

    public string LatencyText
    {
        get => _latencyText;
        private set => SetProperty(ref _latencyText, value);
    }

    public string FpsText
    {
        get => _fpsText;
        private set => SetProperty(ref _fpsText, value);
    }

    public BitmapSource? SourceImage
    {
        get => _sourceImage;
        private set
        {
            if (SetProperty(ref _sourceImage, value))
            {
                OnPropertyChanged(nameof(HasImage));
                RaiseCommandStates();
            }
        }
    }

    public bool HasImage => SourceImage is not null;

    public IReadOnlyList<Detection> Detections
    {
        get => _detections;
        private set => SetProperty(ref _detections, value);
    }

    public bool IsModelLoaded
    {
        get => _isModelLoaded;
        private set
        {
            if (SetProperty(ref _isModelLoaded, value))
            {
                RaiseCommandStates();
            }
        }
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (SetProperty(ref _isBusy, value))
            {
                RaiseCommandStates();
            }
        }
    }

    public AsyncRelayCommand LoadModelCommand { get; }
    public RelayCommand LoadImageCommand { get; }
    public AsyncRelayCommand DetectCommand { get; }
    public RelayCommand SelectValidationFolderCommand { get; }
    public AsyncRelayCommand ValidateFolderCommand { get; }

    private async Task LoadModelAsync()
    {
        string? path = _dialogs.SelectModel();
        if (path is null)
        {
            return;
        }

        try
        {
            IsBusy = true;
            IsModelLoaded = false;
            Status = $"{SelectedDevice}에 모델을 로드하는 중...";
            await _detector.LoadModelAsync(path, SelectedDevice);
            ModelPath = path;
            IsModelLoaded = true;
            Status = $"모델 로드 완료: {_detector.ModelDescription} / {SelectedDevice}";
        }
        catch (Exception exception)
        {
            Status = "모델 로드 실패";
            _dialogs.ShowError(exception.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void LoadImage()
    {
        string? path = _dialogs.SelectImage();
        if (path is null)
        {
            return;
        }

        try
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.UriSource = new Uri(path, UriKind.Absolute);
            bitmap.EndInit();
            bitmap.Freeze();

            ImagePath = path;
            SourceImage = bitmap;
            Detections = [];
            LatencyText = "- ms";
            FpsText = "- FPS";
            Status = $"이미지 로드 완료: {Path.GetFileName(path)}";
        }
        catch (Exception exception)
        {
            _dialogs.ShowError(exception.Message);
        }
    }

    private async Task DetectAsync()
    {
        if (SourceImage is null)
        {
            return;
        }

        try
        {
            IsBusy = true;
            Status = $"{SelectedDevice} 추론 중...";
            InferenceResult result = await _detector.DetectAsync(SourceImage, 0.25f, 0.45f);
            Detections = result.Detections;
            LatencyText = $"{result.LatencyMs:F1} ms";
            FpsText = $"{result.Fps:F1} FPS";
            Status = $"{result.Detections.Count}개 객체 검출 / {_detector.ModelDescription}";
        }
        catch (Exception exception)
        {
            Status = "추론 실패";
            _dialogs.ShowError(exception.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void SelectValidationFolder()
    {
        string? folder = _dialogs.SelectValidationImageFolder();
        if (folder is null)
        {
            return;
        }

        ValidationImageFolder = folder;
        Status = $"COCO 데이터셋 선택: {folder}";
    }

    private async Task ValidateFolderAsync()
    {
        try
        {
            IsBusy = true;
            Status = "Benchmark 준비 중...";
            var progress = new Progress<ValidationProgress>(item =>
            {
                ImagePath = item.ImagePath;
                SourceImage = item.Image;
                Detections = item.Detections;
                Status =
                    $"Benchmark {item.Current}/{item.Total} · {Path.GetFileName(item.ImagePath)}";
            });
            ValidationResult result = await _validationService.ValidateAsync(
                ValidationImageFolder, _detector, progress);
            string csvPath = Path.Combine(ValidationImageFolder, "benchmark.csv");
            SaveValidationCsv(csvPath, result);

            LatencyText = $"{result.AverageLatencyMs:F1} ms";
            FpsText = $"{result.AverageFps:F1} FPS";
            Status =
                $"Benchmark 완료 · Precision {result.Precision:P1} · Recall {result.Recall:P1}";
            _dialogs.ShowInformation(
                $"Benchmark 결과를 누적 저장했습니다.\n{csvPath}\n\n" +
                $"이미지: {result.ImageCount}\n" +
                $"Precision: {result.Precision:P2}\n" +
                $"Recall: {result.Recall:P2}\n" +
                $"F1-score: {result.F1Score:P2}\n" +
                $"Mean IoU: {result.MeanMatchedIou:P2}\n" +
                $"평균 latency: {result.AverageLatencyMs:F2} ms");
        }
        catch (Exception exception)
        {
            Status = "Benchmark 실패";
            _dialogs.ShowError(exception.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void SaveValidationCsv(string csvPath, ValidationResult result)
    {
        string[] headers =
        [
            "model_name", "device", "latency_average_ms", "precision",
            "recall", "f1_score", "iou"
        ];
        string[] values =
        [
            Path.GetFileNameWithoutExtension(ModelPath),
            SelectedDevice,
            result.AverageLatencyMs.ToString("F4", CultureInfo.InvariantCulture),
            result.Precision.ToString("F6", CultureInfo.InvariantCulture),
            result.Recall.ToString("F6", CultureInfo.InvariantCulture),
            result.F1Score.ToString("F6", CultureInfo.InvariantCulture),
            result.MeanMatchedIou.ToString("F6", CultureInfo.InvariantCulture)
        ];

        var csv = new StringBuilder();
        if (!File.Exists(csvPath) || new FileInfo(csvPath).Length == 0)
        {
            csv.AppendLine(string.Join(",", headers.Select(EscapeCsv)));
        }
        csv.AppendLine(string.Join(",", values.Select(EscapeCsv)));
        File.AppendAllText(csvPath, csv.ToString(), new UTF8Encoding(true));
    }

    private static string EscapeCsv(string value)
    {
        return $"\"{value.Replace("\"", "\"\"")}\"";
    }

    private void RaiseCommandStates()
    {
        LoadModelCommand.RaiseCanExecuteChanged();
        LoadImageCommand.RaiseCanExecuteChanged();
        DetectCommand.RaiseCanExecuteChanged();
        SelectValidationFolderCommand.RaiseCanExecuteChanged();
        ValidateFolderCommand.RaiseCanExecuteChanged();
    }

    public void Dispose() => _detector.Dispose();
}
