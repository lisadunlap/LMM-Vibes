# Core API Reference

Reference documentation for the core LMM-Vibes functions and classes.

## Evaluation Functions

### `evaluate_model`

Main function for evaluating model performance.

```python
from lmmvibes.evaluation import evaluate_model

evaluate_model(
    data: List[Dict],
    metrics: List[str] = ["accuracy"],
    config: Optional[Dict] = None,
    model_name: Optional[str] = None
) -> Dict
```

**Parameters:**
- `data`: List of dictionaries containing evaluation data
- `metrics`: List of metric names to compute
- `config`: Optional configuration dictionary
- `model_name`: Optional name for the model being evaluated

**Returns:**
- Dictionary containing evaluation results

**Example:**
```python
data = [
    {"question": "What is 2+2?", "answer": "4", "model_output": "The answer is 4."}
]

results = evaluate_model(
    data=data,
    metrics=["accuracy", "bleu", "rouge"],
    model_name="gpt-4"
)
```

### `batch_evaluate`

Evaluate multiple models or datasets.

```python
from lmmvibes.evaluation import batch_evaluate

batch_evaluate(
    data: List[Dict],
    models: List[str],
    metrics: List[str] = ["accuracy"],
    config: Optional[Dict] = None
) -> Dict[str, Dict]
```

**Parameters:**
- `data`: List of dictionaries containing evaluation data
- `models`: List of model names to evaluate
- `metrics`: List of metric names to compute
- `config`: Optional configuration dictionary

**Returns:**
- Dictionary mapping model names to their evaluation results

## Data Functions

### `load_dataset`

Load data from various file formats.

```python
from lmmvibes.data import load_dataset

load_dataset(
    file_path: str,
    format: Optional[str] = None,
    validation: bool = True
) -> List[Dict]
```

**Parameters:**
- `file_path`: Path to the data file
- `format`: File format (auto-detected if None)
- `validation`: Whether to validate data format

**Returns:**
- List of dictionaries containing the loaded data

**Supported Formats:**
- JSONL (`.jsonl`)
- JSON (`.json`)
- CSV (`.csv`)

### `save_dataset`

Save data to various file formats.

```python
from lmmvibes.data import save_dataset

save_dataset(
    data: List[Dict],
    file_path: str,
    format: Optional[str] = None
) -> None
```

**Parameters:**
- `data`: List of dictionaries to save
- `file_path`: Path where to save the file
- `format`: File format (auto-detected if None)

## Configuration Classes

### `EvaluationConfig`

Configuration class for evaluation settings.

```python
from lmmvibes.config import EvaluationConfig

config = EvaluationConfig(
    metrics: List[str] = ["accuracy"],
    batch_size: int = 32,
    save_results: bool = False,
    output_dir: str = "./results"
)
```

**Attributes:**
- `metrics`: List of metrics to compute
- `batch_size`: Batch size for processing
- `save_results`: Whether to save results to disk
- `output_dir`: Directory to save results

### `DataConfig`

Configuration class for data settings.

```python
from lmmvibes.config import DataConfig

config = DataConfig(
    input_format: str = "jsonl",
    output_format: str = "json",
    validation: bool = True,
    required_fields: List[str] = ["question", "answer", "model_output"]
)
```

**Attributes:**
- `input_format`: Format of input data
- `output_format`: Format for output data
- `validation`: Whether to validate data
- `required_fields`: Required fields in data

## Utility Functions

### `save_results`

Save evaluation results to file.

```python
from lmmvibes.utils import save_results

save_results(
    results: Dict,
    file_path: str,
    format: str = "json"
) -> None
```

**Parameters:**
- `results`: Evaluation results dictionary
- `file_path`: Path where to save results
- `format`: Output format ("json", "yaml", "csv")

### `load_results`

Load evaluation results from file.

```python
from lmmvibes.utils import load_results

load_results(file_path: str) -> Dict
```

**Parameters:**
- `file_path`: Path to the results file

**Returns:**
- Dictionary containing the loaded results

## Visualization Functions

### `plot_metrics`

Create plots of evaluation metrics.

```python
from lmmvibes.visualization import plot_metrics

plot_metrics(
    results: Dict,
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None
```

**Parameters:**
- `results`: Evaluation results dictionary
- `metrics`: Specific metrics to plot (all if None)
- `title`: Plot title
- `save_path`: Path to save the plot

### `plot_comparison`

Compare multiple models or results.

```python
from lmmvibes.visualization import plot_comparison

plot_comparison(
    results_list: List[Dict],
    labels: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None
) -> None
```

**Parameters:**
- `results_list`: List of evaluation results
- `labels`: Labels for each result set
- `metrics`: Specific metrics to compare
- `title`: Plot title

## Metric Classes

### `Metric`

Base class for custom metrics.

```python
from lmmvibes.metrics import Metric

class CustomMetric(Metric):
    def compute(self, predictions: List[str], references: List[str]) -> float:
        # Your custom computation logic
        return score
```

**Methods:**
- `compute(predictions, references)`: Compute the metric score

### Built-in Metrics

Available built-in metrics:

- `accuracy`: Exact match accuracy
- `bleu`: BLEU score for text similarity
- `rouge`: ROUGE score for text similarity
- `bert_score`: BERT-based semantic similarity
- `exact_match`: Exact string matching
- `f1_score`: F1 score for classification tasks

## Error Classes

### `EvaluationError`

Raised when evaluation fails.

```python
from lmmvibes.exceptions import EvaluationError

try:
    results = evaluate_model(data=data)
except EvaluationError as e:
    print(f"Evaluation failed: {e}")
```

### `DataValidationError`

Raised when data validation fails.

```python
from lmmvibes.exceptions import DataValidationError

try:
    data = load_dataset("data.jsonl")
except DataValidationError as e:
    print(f"Data validation failed: {e}")
```

## Next Steps

- Check out [Utilities](utilities.md) for additional helper functions
- Learn about [Basic Usage](../user-guide/basic-usage.md) for practical examples
- Explore [Configuration](../user-guide/configuration.md) for advanced setup 