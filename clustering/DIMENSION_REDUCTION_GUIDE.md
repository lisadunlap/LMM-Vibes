# Dimension Reduction Guide for Better Semantic Clustering

## Problem: Poor Cluster Semantic Coherence

You've noticed that clusters contain very different properties that don't seem semantically related. This is often caused by aggressive dimension reduction that loses semantic structure.

## Root Causes

### 1. **Aggressive UMAP Settings**
The old implementation used problematic UMAP parameters:
- `min_dist=0.0` - Forces points to cluster based on local density rather than semantic similarity
- `n_neighbors=15` - Too low for preserving global structure
- Fixed 50 dimensions - May be too aggressive for semantic preservation

### 2. **One-Size-Fits-All Approach**
- Applied dimension reduction to all datasets regardless of size
- No consideration of embedding quality or dimensionality

## Solution: Improved Adaptive Dimension Reduction

### New Features

#### 1. **Adaptive Method Selection**
The system now intelligently chooses dimension reduction based on your data:

```python
# Small datasets (< 1000 points) - No reduction
# Medium datasets (1000-5000 points) - Conservative UMAP
# Large datasets (> 5000 points) - UMAP with optimized parameters
# Very high dimensional (> 300 dims) - PCA for variance preservation
```

#### 2. **Conservative UMAP Settings**
- `n_components=100` (vs old 50) - Preserves more information
- `n_neighbors=30` (vs old 15) - Better global structure preservation
- `min_dist=0.1` (vs old 0.0) - Maintains semantic relationships
- `metric='cosine'` - Better for semantic similarity

#### 3. **Multiple Methods**
- **Adaptive**: Auto-chooses best method for your data
- **UMAP**: Manual UMAP with conservative settings
- **PCA**: Linear dimension reduction preserving variance
- **None**: Skip dimension reduction entirely

## Usage Examples

### 1. **Test Different Settings**
```bash
python test_dimension_reduction.py --file your_data.jsonl
```

This will compare all methods and show you which gives the best semantic coherence.

### 2. **Skip Dimension Reduction**
```bash
python clustering/hierarchical_clustering.py \
    --file your_data.jsonl \
    --dim-reduction-method none
```

### 3. **Use Conservative UMAP**
```bash
python clustering/hierarchical_clustering.py \
    --file your_data.jsonl \
    --dim-reduction-method umap \
    --umap-n-components 100 \
    --umap-n-neighbors 30 \
    --umap-min-dist 0.1
```

### 4. **Use PCA for High-Dimensional Data**
```bash
python clustering/hierarchical_clustering.py \
    --file your_data.jsonl \
    --dim-reduction-method pca
```

## When to Use Each Method

### **No Dimension Reduction (`--dim-reduction-method none`)**
- ✅ **Best for**: Small datasets (< 1000 points)
- ✅ **Best for**: When you want maximum semantic preservation
- ✅ **Best for**: Debugging cluster quality issues
- ❌ **Avoid**: Very large datasets (will be slow)

### **Adaptive (Default)**
- ✅ **Best for**: Most use cases
- ✅ **Best for**: Automatic optimization
- ✅ **Best for**: Balanced performance and quality

### **UMAP Conservative**
- ✅ **Best for**: Large datasets where you want semantic preservation
- ✅ **Best for**: When adaptive chooses UMAP but you want to tune parameters
- ✅ **Best for**: Medium-sized datasets with complex structure

### **PCA**
- ✅ **Best for**: Very high-dimensional embeddings (> 300 dimensions)
- ✅ **Best for**: When you want linear, interpretable reduction
- ✅ **Best for**: Preserving maximum variance

## Troubleshooting Poor Clusters

### 1. **Clusters seem random or incoherent**
```bash
# Try no dimension reduction first
python clustering/hierarchical_clustering.py \
    --file your_data.jsonl \
    --dim-reduction-method none
```

### 2. **Too many outliers**
```bash
# Use more conservative UMAP settings
python clustering/hierarchical_clustering.py \
    --file your_data.jsonl \
    --dim-reduction-method umap \
    --umap-n-components 150 \
    --umap-n-neighbors 50 \
    --umap-min-dist 0.2
```

### 3. **Clusters too large and vague**
```bash
# Use more aggressive settings or PCA
python clustering/hierarchical_clustering.py \
    --file your_data.jsonl \
    --dim-reduction-method pca
```

## Performance Considerations

| Method | Speed | Semantic Preservation | Best For |
|--------|-------|----------------------|----------|
| None | Slow | Excellent | Small datasets, debugging |
| Adaptive | Fast | Good | Most use cases |
| UMAP | Medium | Good | Large datasets |
| PCA | Fast | Medium | High-dimensional data |

## Monitoring Quality

The system now provides detailed logging about dimension reduction:

```
Applying UMAP dimensionality reduction (preserving semantic coherence)...
  Original shape: (5000, 1536)
  Target components: 100
  Neighbors: 30, min_dist: 0.1
  Reduced to shape: (5000, 100)
```

## Migration from Old Settings

If you were using the old clustering with poor results:

1. **First, try no dimension reduction**:
   ```bash
   --dim-reduction-method none
   ```

2. **If that's too slow, use adaptive**:
   ```bash
   --dim-reduction-method adaptive
   ```

3. **For large datasets, use conservative UMAP**:
   ```bash
   --dim-reduction-method umap --umap-n-components 100
   ```

## Expected Improvements

With these changes, you should see:
- ✅ More semantically coherent clusters
- ✅ Better cluster summaries that make sense
- ✅ Fewer "random" groupings of unrelated properties
- ✅ More interpretable results
- ✅ Better preservation of semantic relationships

The key insight is that **semantic clustering benefits from preserving semantic structure**, which the old aggressive dimension reduction was destroying. 