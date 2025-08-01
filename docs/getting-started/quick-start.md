# Quick Start

Get up and running with LMM-Vibes in minutes.

## Basic Usage

### 1. Import the Package

```python
import lmmvibes
from lmmvibes.evaluation import evaluate_model
from lmmvibes.data import load_dataset
```

### 2. Load Your Data

```python
# Load a dataset
data = load_dataset("path/to/your/data.jsonl")

# Or create sample data
sample_data = [
    {"question": "What is 2+2?", "answer": "4", "model_output": "The answer is 4."},
    {"question": "Explain gravity", "answer": "Gravity is a force", "model_output": "Gravity is a fundamental force..."}
]
```

### 3. Run Evaluation

```python
# Basic evaluation
results = evaluate_model(
    data=sample_data,
    metrics=["accuracy", "bleu", "rouge"]
)

print(f"Accuracy: {results['accuracy']:.2f}")
print(f"BLEU Score: {results['bleu']:.2f}")
```

### 4. Visualize Results

```python
from lmmvibes.visualization import plot_results

# Create a simple plot
plot_results(results, title="Model Performance")
```

## Advanced Usage

### Custom Metrics

```python
from lmmvibes.metrics import custom_metric

def my_custom_metric(prediction, reference):
    # Your custom logic here
    return score

results = evaluate_model(
    data=data,
    metrics=["accuracy", my_custom_metric]
)
```

### Batch Processing

```python
# Process multiple models
models = ["model1", "model2", "model3"]
all_results = {}

for model in models:
    results = evaluate_model(data=data, model_name=model)
    all_results[model] = results
```

## Configuration

Create a configuration file `config.yaml`:

```yaml
evaluation:
  metrics: ["accuracy", "bleu", "rouge"]
  batch_size: 32
  
data:
  input_format: "jsonl"
  output_format: "json"
  
visualization:
  theme: "default"
  save_plots: true
```

## Next Steps

- Check out the [User Guide](../user-guide/basic-usage.md) for detailed usage
- Explore the [API Reference](../api/core.md) for all available functions
- Learn about [Configuration](../user-guide/configuration.md) options 