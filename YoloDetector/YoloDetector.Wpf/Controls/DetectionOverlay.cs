using System.Windows;
using System.Windows.Media;
using YoloDetector.Wpf.Models;

namespace YoloDetector.Wpf.Controls;

public sealed class DetectionOverlay : FrameworkElement
{
    public static readonly DependencyProperty DetectionsProperty =
        DependencyProperty.Register(
            nameof(Detections),
            typeof(IReadOnlyList<Detection>),
            typeof(DetectionOverlay),
            new FrameworkPropertyMetadata(
                Array.Empty<Detection>(),
                FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty ImageWidthProperty =
        DependencyProperty.Register(
            nameof(ImageWidth),
            typeof(double),
            typeof(DetectionOverlay),
            new FrameworkPropertyMetadata(0d, FrameworkPropertyMetadataOptions.AffectsRender));

    public static readonly DependencyProperty ImageHeightProperty =
        DependencyProperty.Register(
            nameof(ImageHeight),
            typeof(double),
            typeof(DetectionOverlay),
            new FrameworkPropertyMetadata(0d, FrameworkPropertyMetadataOptions.AffectsRender));

    public IReadOnlyList<Detection> Detections
    {
        get => (IReadOnlyList<Detection>)GetValue(DetectionsProperty);
        set => SetValue(DetectionsProperty, value);
    }

    public double ImageWidth
    {
        get => (double)GetValue(ImageWidthProperty);
        set => SetValue(ImageWidthProperty, value);
    }

    public double ImageHeight
    {
        get => (double)GetValue(ImageHeightProperty);
        set => SetValue(ImageHeightProperty, value);
    }

    protected override void OnRender(DrawingContext drawingContext)
    {
        base.OnRender(drawingContext);
        if (ImageWidth <= 0 || ImageHeight <= 0 || ActualWidth <= 0 || ActualHeight <= 0)
        {
            return;
        }

        double scale = Math.Min(ActualWidth / ImageWidth, ActualHeight / ImageHeight);
        double offsetX = (ActualWidth - ImageWidth * scale) / 2;
        double offsetY = (ActualHeight - ImageHeight * scale) / 2;
        double pixelsPerDip = VisualTreeHelper.GetDpi(this).PixelsPerDip;

        foreach (Detection detection in Detections)
        {
            Brush brush = BrushesForClass[detection.ClassId % BrushesForClass.Length];
            double left = offsetX + detection.X1 * scale;
            double top = offsetY + detection.Y1 * scale;
            double width = Math.Max(1, (detection.X2 - detection.X1) * scale);
            double height = Math.Max(1, (detection.Y2 - detection.Y1) * scale);
            drawingContext.DrawRectangle(
                null,
                new Pen(brush, 3),
                new Rect(left, top, width, height));

            var text = new FormattedText(
                $"{detection.Label} {detection.Confidence:P1}",
                System.Globalization.CultureInfo.CurrentCulture,
                FlowDirection.LeftToRight,
                new Typeface("Segoe UI Semibold"),
                14,
                Brushes.White,
                pixelsPerDip);
            var background = new Rect(
                left,
                Math.Max(offsetY, top - text.Height - 8),
                text.Width + 10,
                text.Height + 6);
            drawingContext.DrawRectangle(brush, null, background);
            drawingContext.DrawText(text, new Point(background.X + 5, background.Y + 3));
        }
    }

    private static readonly Brush[] BrushesForClass =
    [
        Freeze(new SolidColorBrush(Color.FromRgb(45, 108, 223))),
        Freeze(new SolidColorBrush(Color.FromRgb(224, 80, 104))),
        Freeze(new SolidColorBrush(Color.FromRgb(52, 190, 139))),
        Freeze(new SolidColorBrush(Color.FromRgb(237, 165, 54))),
        Freeze(new SolidColorBrush(Color.FromRgb(148, 103, 189))),
        Freeze(new SolidColorBrush(Color.FromRgb(35, 171, 196)))
    ];

    private static Brush Freeze(Brush brush)
    {
        brush.Freeze();
        return brush;
    }
}
