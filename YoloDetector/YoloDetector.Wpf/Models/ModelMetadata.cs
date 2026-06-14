using System.Text.Json.Serialization;

namespace YoloDetector.Wpf.Models;

public sealed class ModelMetadata
{
    [JsonPropertyName("model_name")]
    public string ModelName { get; set; } = string.Empty;

    [JsonPropertyName("optimization")]
    public string Optimization { get; set; } = string.Empty;

    [JsonPropertyName("labels")]
    public List<string> Labels { get; set; } = [];
}
