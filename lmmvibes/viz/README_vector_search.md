# Vector Search for LMM-Vibes

This module provides semantic search capabilities for behavioral properties extracted by the LMM-Vibes pipeline.

## Overview

The vector search functionality allows users to find relevant behavioral properties using natural language queries. It uses OpenAI embeddings to compute semantic similarity between user queries and property descriptions.

## Features

- **Semantic Search**: Find properties using natural language descriptions
- **Model Filtering**: Search within specific models or across all models
- **Similarity Scoring**: Results ranked by cosine similarity
- **Example Retrieval**: View actual conversations for each property
- **Caching**: Precomputed embeddings for fast search
- **Flexible Queries**: Support for various search strategies

## Usage

### In the Streamlit App

1. **Launch the app**:
   ```bash
   streamlit run lmmvibes/viz/pipeline_results_app.py -- --results_dir path/to/results/
   ```

2. **Navigate to Vector Search tab**: Click on the "🔎 Vector Search" tab

3. **Enter your query**: Use natural language to describe the behavioral property you're looking for

4. **Adjust settings**:
   - **Max results**: Number of results to return (5-50)
   - **Min similarity**: Similarity threshold (0.0-1.0)
   - **Model filter**: Optionally filter by specific models

5. **View results**: Each result shows:
   - Property description
   - Model name
   - Cluster information
   - Similarity score
   - Evidence (if available)
   - Option to view example conversations

### Precomputing Embeddings

For faster search performance, you can precompute embeddings:

```bash
python -m lmmvibes.viz.precompute_embeddings --results_dir path/to/results/
```

Options:
- `--embedding_model`: Choose embedding model (default: "openai")
- `--force_recompute`: Regenerate existing embeddings

## Search Strategies

### Effective Query Examples

**Specific behaviors:**
- "step-by-step reasoning"
- "creative problem solving"
- "formal academic tone"
- "user-friendly explanations"

**Combined concepts:**
- "detailed reasoning with examples"
- "concise but accurate responses"
- "helpful and thorough explanations"

**Behavioral patterns:**
- "admits uncertainty"
- "provides multiple solutions"
- "asks clarifying questions"

### Understanding Similarity Scores

- **0.9+**: Very similar properties
- **0.7-0.9**: Related properties
- **0.5-0.7**: Somewhat related properties
- **<0.5**: Weakly related (filtered out by default)

## Technical Details

### Architecture

1. **PropertyVectorSearch**: Main search engine class
2. **Embedding Computation**: Uses OpenAI's text-embedding-3-large model
3. **Similarity Calculation**: Cosine similarity between normalized embeddings
4. **Caching**: LMDB-based caching for embeddings
5. **Metadata Indexing**: Efficient property metadata lookup

### File Structure

```
results/
├── clustered_results.json          # Pipeline results with properties
├── model_stats.json               # Model statistics
├── property_embeddings.npy        # Precomputed embeddings (generated)
└── ...
```

### Performance Considerations

- **First run**: Embeddings computed on-demand (may take time)
- **Subsequent runs**: Load precomputed embeddings (fast)
- **Memory usage**: ~6MB per 1000 properties (1536-dimensional embeddings)
- **Search speed**: ~100ms for 10k properties

## API Reference

### PropertyVectorSearch

```python
from lmmvibes.viz.vector_search import PropertyVectorSearch

# Initialize
search_engine = PropertyVectorSearch(results_path)

# Search across all models
results = search_engine.search(query, top_k=10, min_similarity=0.5)

# Search within specific models
results = search_engine.search_by_model(query, models, top_k=10, min_similarity=0.5)

# Get examples for a property
examples = search_engine.get_property_examples(property_description, max_examples=5)

# Get statistics
stats = search_engine.get_statistics()
```

### SearchResult

```python
@dataclass
class SearchResult:
    property_description: str
    model: str
    cluster_id: str
    cluster_label: str
    similarity_score: float
    question_id: str
    evidence: Optional[str] = None
    category: Optional[str] = None
    impact: Optional[str] = None
    type: Optional[str] = None
```

## Troubleshooting

### Common Issues

1. **"No results found"**
   - Try different keywords
   - Lower the similarity threshold
   - Check spelling and terminology

2. **"Failed to initialize vector search"**
   - Ensure `clustered_results.json` exists
   - Check file permissions
   - Verify OpenAI API access

3. **Slow search performance**
   - Precompute embeddings using the utility script
   - Check network connectivity for OpenAI API
   - Consider using local sentence transformers

### Error Messages

- **FileNotFoundError**: Missing required files
- **APIError**: OpenAI API issues
- **MemoryError**: Large dataset without sufficient RAM

## Future Enhancements

- [ ] Support for local embedding models (SentenceTransformers)
- [ ] Advanced filtering by category, impact, or type
- [ ] Batch search capabilities
- [ ] Search result export functionality
- [ ] Integration with external vector databases (Pinecone, Weaviate)
- [ ] Multi-modal search (text + metadata) 