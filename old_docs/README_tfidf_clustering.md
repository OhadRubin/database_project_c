# Ray TF-IDF Vectorization and Clustering

This module implements a distributed document clustering pipeline using [Ray](https://ray.io) for better scalability and performance. The pipeline combines TF-IDF vectorization, dimensionality reduction, and a custom K-Means clustering implementation to efficiently process large text datasets.

## Features

- **Distributed Processing**: Uses Ray for parallel processing across multiple machines/cores
- **Multi-Stage Clustering**: Hierarchical clustering in 2 stages for better granularity
- **Custom KMeans Implementation**:
  - Online learning support (update clusters without retraining)
  - Balanced clustering option (equal-sized clusters)
  - JAX acceleration for faster distance computations
- **Optimized Vectorization**: TF-IDF with number normalization and dimensionality reduction

## Implementation Details

### TF-IDF Vectorization

The implementation uses scikit-learn's TF-IDF vectorizer with the following enhancements:
- Number normalization (replacing all numeric tokens with a common token)
- Stop word filtering (including English stop words)
- SVD for dimensionality reduction
- Normalization for more stable clustering

### Custom KMeans

The custom KMeans implementation extends scikit-learn's KMeans with:

1. **Online Learning**: Update cluster centroids with new data batches without recomputing from scratch
2. **Balanced Clustering**: Ensures clusters have roughly equal numbers of points
3. **JAX Acceleration**: Uses JAX for faster distance computations when available

### Clustering Pipeline

The pipeline follows these steps:

1. **Stage 1**:
   - Sample dataset
   - Fit TF-IDF vectorizer and KMeans model
   - Apply to entire dataset

2. **Stage 2**:
   - For each Stage 1 cluster, train a separate TF-IDF and KMeans model
   - Apply models to documents within their respective Stage 1 cluster
   - Create a 2-level hierarchical clustering

## Usage

```python
from pyspark.sql import SparkSession
from src.ray_tfidf_vec import tfidf_minhash_ray

# Initialize Spark
spark = SparkSession.builder.appName("TextClustering").getOrCreate()

# Load your data
df = spark.read.parquet("your_documents.parquet")

# Run the clustering pipeline
output_path = tfidf_minhash_ray(
    spark=spark,
    df=df,
    column="text_column",  # Your text column name
    num_perm=128,         # MinHash parameters (optional)
    ngram_size=5,
    min_ngram_size=1,
    threshold=0.85
)

# Read the clustered data
clustered_df = spark.read.parquet(output_path)
```

## Configuration Options

The pipeline accepts a configuration object with the following parameters:

- `cluster_layout`: List of integers defining the number of clusters at each stage
- `max_docs`: Maximum number of documents to sample for training
- `stage1_train_cpus`: CPUs allocated to Stage 1 training
- `stage2_train_cpus`: CPUs allocated to each Stage 2 training task
- Various batch size parameters for different stages

## Performance Tips

- Increase `max_docs` for more accurate clustering but longer training time
- Adjust `tfidf_batch_size` based on available memory
- Enable JAX for faster distance calculations in K-means
- Set appropriate CPU allocation for different stages based on your hardware 