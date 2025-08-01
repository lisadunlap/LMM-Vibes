# Configuration

Learn how to configure LMM-Vibes for your specific needs.

## Configuration Files

LMM-Vibes supports configuration through YAML files. Create a `config.yaml` file in your project root:

```yaml
# config.yaml
evaluation:
  metrics: ["accuracy", "bleu", "rouge"]
  batch_size: 32
  save_results: true
  output_dir: "./results"

data:
  input_format: "jsonl"
  output_format: "json"
  validation: true

visualization:
  theme: "default"
  save_plots: true
  plot_format: "png"

logging:
  level: "INFO"
  file: "lmmvibes.log"
```

## Loading Configuration

```python
from lmmvibes.config import load_config

# Load from file
config = load_config("config.yaml")

# Use in evaluation
from lmmvibes.evaluation import evaluate_model
results = evaluate_model(data=data, config=config)
```

## Configuration Options

### Evaluation Settings

```yaml
evaluation:
  # List of metrics to compute
  metrics: ["accuracy", "bleu", "rouge", "bert_score"]
  
  # Batch size for processing
  batch_size: 32
  
  # Whether to save results
  save_results: true
  
  # Output directory for results
  output_dir: "./results"
  
  # Model-specific settings
  model:
    name: "gpt-4"
    temperature: 0.0
    max_tokens: 1000
```

### Data Settings

```yaml
data:
  # Input data format
  input_format: "jsonl"  # or "json", "csv"
  
  # Output format for results
  output_format: "json"  # or "csv", "yaml"
  
  # Enable data validation
  validation: true
  
  # Required fields in data
  required_fields: ["question", "answer", "model_output"]
  
  # Optional metadata fields
  metadata_fields: ["category", "difficulty", "source"]
```

### Visualization Settings

```yaml
visualization:
  # Plot theme
  theme: "default"  # or "dark", "light"
  
  # Save plots automatically
  save_plots: true
  
  # Plot format
  plot_format: "png"  # or "pdf", "svg"
  
  # Plot size
  figure_size: [10, 6]
  
  # Color palette
  colors: ["#1f77b4", "#ff7f0e", "#2ca02c"]
```

### Logging Settings

```yaml
logging:
  # Log level
  level: "INFO"  # or "DEBUG", "WARNING", "ERROR"
  
  # Log file
  file: "lmmvibes.log"
  
  # Console output
  console: true
  
  # Log format
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Environment Variables

You can also configure LMM-Vibes using environment variables:

```bash
# Set environment variables
export LMMVIBES_CONFIG_FILE="my_config.yaml"
export LMMVIBES_LOG_LEVEL="DEBUG"
export LMMVIBES_OUTPUT_DIR="./my_results"
```

## Programmatic Configuration

```python
from lmmvibes.config import EvaluationConfig, DataConfig, VisualizationConfig

# Create configuration programmatically
eval_config = EvaluationConfig(
    metrics=["accuracy", "bleu"],
    batch_size=64,
    save_results=True
)

data_config = DataConfig(
    input_format="jsonl",
    validation=True
)

# Combine configurations
config = {
    "evaluation": eval_config,
    "data": data_config
}
```

## Configuration Validation

LMM-Vibes validates your configuration:

```python
from lmmvibes.config import validate_config

# Validate configuration
try:
    validate_config(config)
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Default Configuration

If no configuration is provided, LMM-Vibes uses sensible defaults:

```python
default_config = {
    "evaluation": {
        "metrics": ["accuracy"],
        "batch_size": 32,
        "save_results": False
    },
    "data": {
        "input_format": "jsonl",
        "validation": True
    },
    "visualization": {
        "theme": "default",
        "save_plots": False
    }
}
```

## Configuration Inheritance

You can inherit from a base configuration:

```yaml
# base_config.yaml
evaluation:
  metrics: ["accuracy", "bleu"]
  batch_size: 32

# my_config.yaml
inherit: "base_config.yaml"
evaluation:
  metrics: ["accuracy", "bleu", "rouge"]  # Override metrics
  batch_size: 64  # Override batch size
```

## Best Practices

1. **Use Configuration Files**: Store settings in YAML files for reproducibility
2. **Version Control**: Include configuration files in version control
3. **Environment-Specific**: Use different configs for development/production
4. **Documentation**: Document your configuration choices
5. **Validation**: Always validate configurations before use

## Next Steps

- Learn about [Basic Usage](basic-usage.md) with your configuration
- Explore the [API Reference](../api/core.md) for advanced configuration options
- Check out [Contributing](../development/contributing.md) guidelines 