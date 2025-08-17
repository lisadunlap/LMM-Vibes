# Utilities API Reference

Reference documentation for utility functions and helper classes in LMM-Vibes.

## Data Utilities

### `validate_data_format`

Validate that data follows the expected format.

```python
from stringsight.utils import validate_data_format

validate_data_format(
    data: List[Dict],
    required_fields: List[str] = ["question", "answer", "model_output"],
    optional_fields: List[str] = []
) -> bool
```

**Parameters:**
- `data`: List of data dictionaries to validate
- `required_fields`: Fields that must be present in each item
- `optional_fields`: Fields that may be present

**Returns:**
- `True` if data is valid, raises `DataValidationError` otherwise

### `convert_data_format`

Convert data between different formats.

```python
from stringsight.utils import convert_data_format

convert_data_format(
    data: List[Dict],
    target_format: str,
    field_mapping: Optional[Dict[str, str]] = None
) -> List[Dict]
```

**Parameters:**
- `data`: Input data
- `target_format`: Target format ("jsonl", "json", "csv")
- `field_mapping`: Optional mapping of field names

**Returns:**
- Data in the target format

## File Utilities

### `ensure_directory`

Ensure a directory exists, creating it if necessary.

```python
from stringsight.utils import ensure_directory

ensure_directory(path: str) -> None
```

**Parameters:**
- `path`: Directory path to ensure exists

### `get_file_extension`

Get the file extension from a path.

```python
from stringsight.utils import get_file_extension

get_file_extension(file_path: str) -> str
```

**Parameters:**
- `file_path`: Path to the file

**Returns:**
- File extension (e.g., ".json", ".csv")

### `sanitize_filename`

Create a safe filename from a string.

```python
from stringsight.utils import sanitize_filename

sanitize_filename(filename: str) -> str
```

**Parameters:**
- `filename`: Original filename

**Returns:**
- Sanitized filename safe for filesystem

## Text Utilities

### `normalize_text`

Normalize text for consistent processing.

```python
from stringsight.utils import normalize_text

normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    remove_whitespace: bool = False
) -> str
```

**Parameters:**
- `text`: Text to normalize
- `lowercase`: Whether to convert to lowercase
- `remove_punctuation`: Whether to remove punctuation
- `remove_whitespace`: Whether to normalize whitespace

**Returns:**
- Normalized text

### `tokenize_text`

Tokenize text into words or subwords.

```python
from stringsight.utils import tokenize_text

tokenize_text(
    text: str,
    method: str = "word",
    language: str = "en"
) -> List[str]
```

**Parameters:**
- `text`: Text to tokenize
- `method`: Tokenization method ("word", "subword", "sentence")
- `language`: Language code for tokenization

**Returns:**
- List of tokens

## Metric Utilities

### `compute_metric`

Compute a single metric on predictions and references.

```python
from stringsight.utils import compute_metric

compute_metric(
    metric_name: str,
    predictions: List[str],
    references: List[str],
    **kwargs
) -> float
```

**Parameters:**
- `metric_name`: Name of the metric to compute
- `predictions`: List of model predictions
- `references`: List of reference answers
- `**kwargs`: Additional arguments for the metric

**Returns:**
- Computed metric score

### `aggregate_metrics`

Aggregate multiple metric scores.

```python
from stringsight.utils import aggregate_metrics

aggregate_metrics(
    scores: List[float],
    method: str = "mean"
) -> float
```

**Parameters:**
- `scores`: List of metric scores
- `method`: Aggregation method ("mean", "median", "max", "min")

**Returns:**
- Aggregated score

## Configuration Utilities

### `load_config_file`

Load configuration from a file.

```python
from stringsight.utils import load_config_file

load_config_file(
    file_path: str,
    validate: bool = True
) -> Dict
```

**Parameters:**
- `file_path`: Path to configuration file
- `validate`: Whether to validate the configuration

**Returns:**
- Configuration dictionary

### `save_config_file`

Save configuration to a file.

```python
from stringsight.utils import save_config_file

save_config_file(
    config: Dict,
    file_path: str,
    format: str = "yaml"
) -> None
```

**Parameters:**
- `config`: Configuration dictionary
- `file_path`: Path where to save the configuration
- `format`: Output format ("yaml", "json")

## Logging Utilities

### `setup_logging`

Set up logging configuration.

```python
from stringsight.utils import setup_logging

setup_logging(
    level: str = "INFO",
    file_path: Optional[str] = None,
    format_string: Optional[str] = None
) -> None
```

**Parameters:**
- `level`: Logging level
- `file_path`: Optional log file path
- `format_string`: Optional log format string

### `get_logger`

Get a logger instance.

```python
from stringsight.utils import get_logger

get_logger(name: str = "stringsight") -> logging.Logger
```

**Parameters:**
- `name`: Logger name

**Returns:**
- Logger instance

## Time Utilities

### `Timer`

Context manager for timing operations.

```python
from stringsight.utils import Timer

with Timer("operation_name"):
    # Your code here
    pass
```

### `format_duration`

Format a duration in human-readable format.

```python
from stringsight.utils import format_duration

format_duration(seconds: float) -> str
```

**Parameters:**
- `seconds`: Duration in seconds

**Returns:**
- Formatted duration string (e.g., "2m 30s")

## Progress Utilities

### `ProgressBar`

Simple progress bar for long-running operations.

```python
from stringsight.utils import ProgressBar

with ProgressBar(total=100, desc="Processing") as pbar:
    for i in range(100):
        # Your processing code
        pbar.update(1)
```

**Parameters:**
- `total`: Total number of items
- `desc`: Description of the operation

## Validation Utilities

### `validate_metric_name`

Validate that a metric name is supported.

```python
from stringsight.utils import validate_metric_name

validate_metric_name(metric_name: str) -> bool
```

**Parameters:**
- `metric_name`: Name of the metric to validate

**Returns:**
- `True` if metric is supported, `False` otherwise

### `validate_file_format`

Validate that a file format is supported.

```python
from stringsight.utils import validate_file_format

validate_file_format(format_name: str) -> bool
```

**Parameters:**
- `format_name`: Name of the file format to validate

**Returns:**
- `True` if format is supported, `False` otherwise

## Error Handling Utilities

### `handle_errors`

Decorator for consistent error handling.

```python
from stringsight.utils import handle_errors

@handle_errors
def my_function():
    # Your function code
    pass
```

### `retry_on_failure`

Decorator for retrying failed operations.

```python
from stringsight.utils import retry_on_failure

@retry_on_failure(max_attempts=3, delay=1.0)
def my_function():
    # Your function code
    pass
```

**Parameters:**
- `max_attempts`: Maximum number of retry attempts
- `delay`: Delay between retries in seconds

## Next Steps

- Check out [Core API](core.md) for main functions and classes
- Learn about [Basic Usage](../user-guide/basic-usage.md) for practical examples
- Explore [Configuration](../user-guide/configuration.md) for advanced setup 