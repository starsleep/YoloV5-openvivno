using System.Windows;
using YoloDetector.Wpf.Services;
using YoloDetector.Wpf.ViewModels;

namespace YoloDetector.Wpf;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DataContext = new MainViewModel(
            new OpenVinoYoloService(),
            new DialogService(),
            new YoloValidationService());
        Closed += (_, _) => (DataContext as IDisposable)?.Dispose();
    }
}
