# Basic Usage

Learn the fundamentals of using LMM-Vibes for model evaluation and analysis.

## Core Concepts

LMM-Vibes is built around a few key concepts:

- **Datasets**: Your input data containing questions, answers, and model outputs
- **Metrics**: Evaluation measures like accuracy, BLEU, ROUGE, etc.
- **Results**: Structured output containing evaluation scores and metadata
- **Visualizations**: Charts and plots for analyzing results

## Data Format

LMM-Vibes expects data in a specific format:

```python
# Each item should be a dictionary with these keys:
{
    "question": "What is the capital of France?",
    "answer": "Paris",
    "model_output": "The capital of France is Paris.",
    "metadata": {
        "category": "geography",
        "difficulty": "easy"
    }
}
```

## Basic Evaluation

### Simple Evaluation

```python
from lmmvibes.evaluation import evaluate_model

# Load your data
data = load_dataset("your_data.jsonl")

# Run evaluation
results = evaluate_model(
    data=data,
    metrics=["accuracy", "bleu", "rouge"]
)

print(results)
```

### Evaluation with Configuration

```python
from lmmvibes.config import EvaluationConfig

config = EvaluationConfig(
    metrics=["accuracy", "bleu", "rouge"],
    batch_size=32,
    save_results=True,
    output_dir="./results"
)

results = evaluate_model(data=data, config=config)
```

## Working with Results

### Accessing Results

```python
# Get specific metrics
accuracy = results["accuracy"]
bleu_score = results["bleu"]

# Get detailed breakdown
detailed_results = results["detailed"]
per_item_scores = detailed_results["per_item"]
```

### Saving and Loading Results

```python
from lmmvibes.utils import save_results, load_results

# Save results
save_results(results, "my_evaluation_results.json")

# Load results
loaded_results = load_results("my_evaluation_results.json")
```

## Visualization

### Basic Plots

```python
from lmmvibes.visualization import plot_metrics, plot_comparison

# Plot single model results
plot_metrics(results, title="Model Performance")

# Compare multiple models
plot_comparison([results1, results2], labels=["Model A", "Model B"])
```

### Interactive Dashboards

```python
from lmmvibes.visualization import create_dashboard

# Create an interactive dashboard
dashboard = create_dashboard(results)
dashboard.show()
```

## Advanced Features

### Custom Metrics

```python
from lmmvibes.metrics import Metric

class CustomMetric(Metric):
    def compute(self, predictions, references):
        # Your custom computation logic
        return score

# Use custom metric
results = evaluate_model(
    data=data,
    metrics=["accuracy", CustomMetric()]
)
```

### Batch Processing

```python
from lmmvibes.evaluation import batch_evaluate

# Evaluate multiple models
models = ["gpt-3.5", "gpt-4", "claude"]
all_results = batch_evaluate(
    data=data,
    models=models,
    metrics=["accuracy", "bleu"]
)
```

## Error Handling

```python
from lmmvibes.exceptions import EvaluationError

try:
    results = evaluate_model(data=data)
except EvaluationError as e:
    print(f"Evaluation failed: {e}")
    # Handle the error appropriately
```

## Best Practices

1. **Data Validation**: Always validate your input data format
2. **Error Handling**: Use try-catch blocks for robust evaluation
3. **Configuration**: Use configuration files for reproducible experiments
4. **Documentation**: Document your evaluation setup and parameters
5. **Versioning**: Keep track of model versions and evaluation parameters

## Next Steps

- Learn about [Configuration](configuration.md) options
- Explore the [API Reference](../api/core.md) for detailed function documentation
- Check out [Contributing](../development/contributing.md) guidelines 