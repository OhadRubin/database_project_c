import ray
import ray.data
import pandas as pd
import numpy as np
import pickle
import json
from functools import partial
import time
import os
from typing import List, Dict, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as SklearnKMeans
import os
import numpy as np
import pandas as pd # Used for type hints, not core logic
from pyspark.sql import functions as F
import logging
import math
import time
from functools import reduce
from typing import Dict, Any, Iterator, List, Tuple, Optional, Set
# Import scikit-learn components
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
# Import Spark ML components

import socket
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from tqdm import tqdm

tfidf_logger = logging.getLogger('sklearn.feature_extraction.text')
import warnings
warnings.filterwarnings('ignore', message="Your stop_words may be inconsistent with your preprocessing.*", category=UserWarning)

import ray

# Try to import JAX for faster distance calculations
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    warnings.warn("JAX not found, using NumPy for distance calculations instead (slower).")


def jax_pairwise_distance(data1, data2):
    """Compute pairwise Euclidean distances between two sets of points.
    
    If JAX is available, computes distances using JAX for better performance.
    Otherwise, falls back to NumPy implementation.
    
    Args:
        data1: First set of points, shape (m, d)
        data2: Second set of points, shape (n, d)
        
    Returns:
        Distance matrix of shape (m, n)
    """
    if HAS_JAX:
        return _jax_pairwise_distance_impl(data1, data2)
    else:
        return _numpy_pairwise_distance_impl(data1, data2)


def _jax_pairwise_distance_impl(data1, data2):
    """JAX implementation of pairwise Euclidean distance calculation."""
    # Convert to jax arrays if they aren't already
    x1 = jnp.array(data1)
    x2 = jnp.array(data2)
    
    # Compute squared norms of each point
    x1_norm = jnp.sum(x1**2, axis=1)
    x2_norm = jnp.sum(x2**2, axis=1)
    
    # Compute the distance matrix using the formula:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.dot(y)
    x1_x2 = jnp.dot(x1, x2.T)
    dist_mat = x1_norm[:, jnp.newaxis] + x2_norm[jnp.newaxis, :] - 2.0 * x1_x2
    
    # To avoid numerical issues, enforce non-negative distances
    dist_mat = jnp.maximum(dist_mat, 0.0)
    
    # Return Euclidean distance (square root of squared distances)
    return jnp.sqrt(dist_mat)


def _numpy_pairwise_distance_impl(data1, data2):
    """NumPy implementation of pairwise Euclidean distance calculation."""
    # Ensure inputs are numpy arrays
    x1 = np.asarray(data1)
    x2 = np.asarray(data2)
    
    # Compute squared norms
    x1_norm = np.sum(x1**2, axis=1)
    x2_norm = np.sum(x2**2, axis=1)
    
    # Compute distances
    x1_x2 = np.dot(x1, x2.T)
    dist_mat = x1_norm[:, np.newaxis] + x2_norm[np.newaxis, :] - 2.0 * x1_x2
    
    # Enforce non-negative distances and take square root
    dist_mat = np.maximum(dist_mat, 0.0)
    return np.sqrt(dist_mat)


def number_normalizer(tokens):
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))
    


class KMeans:
    """K-Means implementation with support for online learning and balanced clusters.
    
    This implementation extends sklearn.cluster.KMeans with additional features:
    
    1. Online learning - Update cluster centers with new batches of data without retraining
    2. Balanced clustering - Ensure clusters have approximately equal number of points
    3. JAX acceleration - Use JAX for faster distance computations when available
    
    The standard scikit-learn KMeans implementation doesn't support these features, 
    particularly online learning which is important for large datasets that can't
    fit in memory at once.
    
    This implementation is designed to work with Ray's distributed processing pipeline
    for clustering documents using TF-IDF vectorization.
    
    Usage Examples:
    
    # Standard clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Balanced clustering
    kmeans = KMeans(n_clusters=5, balanced=True, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Online learning with batches
    kmeans = KMeans(n_clusters=5, random_state=42)
    for i, batch in enumerate(batches):
        if i == 0:
            kmeans.fit(batch)  # Initial fit
        else:
            kmeans.fit(batch, online=True, iter_k=i)  # Update with new batch
    
    # Prediction with new data
    labels = kmeans.predict(X_new)
    
    Args:
        n_clusters: Number of clusters
        balanced: If True, try to create equal-sized clusters
        random_state: Random seed for reproducibility
        use_jax: Whether to use JAX for distance calculations (faster, if available)
        **kwargs: Additional arguments passed to sklearn.cluster.KMeans
    """
    def __init__(self, n_clusters=8, balanced=False, random_state=None, use_jax=True, **kwargs):
        self.n_clusters = n_clusters
        self.balanced = balanced
        self.random_state = random_state
        self.use_jax = use_jax
        self.kmeans_kwargs = kwargs
        self.cluster_centers_ = None
        self.inertia_ = None
        self._is_fitted = False
        
        # Initialize the sklearn KMeans model
        self._sklearn_kmeans = SklearnKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X, iter_limit=None, online=False, iter_k=None, **kwargs):
        """Fit the K-means model to the data.
        
        Args:
            X: Training data
            iter_limit: Maximum number of iterations (for compatibility)
            online: If True, update existing centroids instead of recomputing
            iter_k: Batch number for online learning
            **kwargs: Additional arguments for sklearn KMeans
        
        Returns:
            Cluster assignments for the training data
        """
        X = np.asarray(X)
        
        # For the first batch or if not online, initialize centroids
        if not online or (online and iter_k == 0) or not self._is_fitted:
            self._sklearn_kmeans = SklearnKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.kmeans_kwargs
            )
            self._sklearn_kmeans.fit(X)
            self.cluster_centers_ = self._sklearn_kmeans.cluster_centers_
            self.inertia_ = self._sklearn_kmeans.inertia_
            self._is_fitted = True
            return self._get_labels(X)
        
        # For subsequent batches in online learning, compute distances using jax if available
        if self.use_jax:
            distances = jax_pairwise_distance(X, self.cluster_centers_)
            labels = np.argmin(distances, axis=1)
        else:
            labels = self._sklearn_kmeans.predict(X)
        
        # Update centroids based on new data
        for i in range(self.n_clusters):
            # Get points assigned to this cluster
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # Update centroid as weighted average of old and new centroids
                old_centroid = self.cluster_centers_[i]
                new_centroid = np.mean(cluster_points, axis=0)
                # Simple moving average: as iterations progress, give more weight to history
                weight = min(0.9, 1.0 / (iter_k + 1)) if iter_k is not None else 0.1
                updated_centroid = (1 - weight) * old_centroid + weight * new_centroid
                self.cluster_centers_[i] = updated_centroid
        
        # Update the sklearn model with the new centroids
        self._sklearn_kmeans.cluster_centers_ = self.cluster_centers_
        
        return self._get_labels(X)
    
    def _get_labels(self, X):
        """Get cluster assignments for X."""
        X = np.asarray(X)
        
        # Compute distances using JAX for better performance when available and enabled
        if self.use_jax:
            distances = jax_pairwise_distance(X, self.cluster_centers_)
        else:
            distances = self._sklearn_kmeans.transform(X)
            
        if self.balanced:
            # Implement balanced clustering (assign equal points to each cluster)
            # This is a simple greedy implementation - could be improved
            n_samples = X.shape[0]
            target_size = n_samples // self.n_clusters
            
            # Start with normal assignments
            labels = np.argmin(distances, axis=1)
            
            # Count points per cluster
            counts = np.bincount(labels, minlength=self.n_clusters)
            
            # Iteratively balance clusters
            for _ in range(3):  # Limit iterations for performance
                for i in range(self.n_clusters):
                    if counts[i] <= target_size:
                        continue
                    
                    # Find points that could be reassigned from this cluster
                    cluster_points = np.where(labels == i)[0]
                    
                    # Sort by how well they fit alternative clusters
                    point_distances = distances[cluster_points]
                    point_distances[:, i] = np.inf  # Exclude current cluster
                    alternative_clusters = np.argmin(point_distances, axis=1)
                    
                    # Sort points by how close they are to alternative
                    alt_distances = np.min(point_distances, axis=1)
                    sorted_indices = np.argsort(alt_distances)
                    
                    # Move points to alternative clusters until balance is achieved
                    for idx in sorted_indices:
                        if counts[i] <= target_size:
                            break
                        
                        point_idx = cluster_points[idx]
                        new_cluster = alternative_clusters[idx]
                        
                        # Only move if target cluster isn't already too full
                        if counts[new_cluster] < target_size:
                            labels[point_idx] = new_cluster
                            counts[i] -= 1
                            counts[new_cluster] += 1
            
            return labels
        else:
            # Use standard k-means assignments
            return np.argmin(distances, axis=1)
    
    def predict(self, X):
        """Predict the closest cluster for each sample in X."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        X = np.asarray(X)
        return self._get_labels(X)

    def fit_predict(self, X, **kwargs):
        """Fit and predict in one step."""
        self.fit(X, **kwargs)
        return self.predict(X)
    
    def transform(self, X):
        """Transform X to a cluster-distance space."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        X = np.asarray(X)
        
        if self.use_jax:
            # Return distances to all centroids using JAX
            return jax_pairwise_distance(X, self.cluster_centers_)
        else:
            # Fall back to sklearn's transform
            return self._sklearn_kmeans.transform(X)


def get_sklearn_feature_pipeline(n_components, random_seed):
    stop_words = list(ENGLISH_STOP_WORDS.union(["#NUMBER"]))
    vectorizer = Pipeline([('tfidf', NumberNormalizingVectorizer(stop_words=stop_words)),
                            ('svd', TruncatedSVD(n_components=n_components,random_state=random_seed)),
                            ('normalizer', Normalizer(copy=False))],
                            verbose=True)
    return vectorizer




def fit_kmeans(iterable, n_clusters, batch_size, **kwargs):
    kmeans = KMeans(n_clusters=n_clusters, balanced=True, **kwargs)
    with tqdm(dynamic_ncols=True,desc="fit_kmeans") as pbar:
        for i,batch in enumerate(iterable):
            pbar.update(batch.shape[0])
            kmeans.fit(batch, iter_limit=20, online=True, iter_k=i)
    return kmeans


# --- Constants ---
CLUSTER_A_COL = 'cluster_A'
CLUSTER_B_COL = 'cluster_B'
TEMP_ID_COL = 'temp_doc_id'



@ray.remote
def fit_models_remote(
    cfg: object,
    sample_data: pd.DataFrame,
    n_clusters: int,
    stage_label: str, # e.g., "Stage1", "Stage2_GroupX", etc.
    kmeans_train_batch_size_key: str, # e.g., "stage1_train_kmeans_bs"
) -> Tuple[object, object]:
    """Fits vectorizer and KMeans on sample data.
    
    This is a Ray remote function that fits the models on the provided data.
    
    Args:
        cfg: Configuration object
        sample_data: DataFrame containing training data
        n_clusters: Number of clusters for KMeans
        stage_label: Label for logging purposes
        kmeans_train_batch_size_key: Config key for batch size
        
    Returns:
        A tuple of (vectorizer, kmeans) model objects (not references).
        When called with .remote(), this function returns a single Ray ObjectRef 
        pointing to this tuple.
    """


    texts = sample_data["text"].tolist()
    print(f"[{stage_label}] Fitting vectorizer on {len(texts)} samples...")
    vectorizer = get_sklearn_feature_pipeline(n_components=128, random_seed=42)
    embeddings = vectorizer.fit_transform(texts)
    print(f"[{stage_label}] Vectorizer fitting done. Embedding shape: {embeddings.shape}")

    print(f"[{stage_label}] Fitting K-means with {n_clusters} clusters...")
    kmeans_batch_size_config = getattr(cfg, kmeans_train_batch_size_key)
        
    kmeans = fit_kmeans(embeddings, n_clusters, batch_size=kmeans_batch_size_config) # Pass computed BS if needed
    print(f"[{stage_label}] K-means fitting done.")

    return vectorizer, kmeans

def apply_models_batch(
    batch: pd.DataFrame,
    vectorizer_ref: ray.ObjectRef,
    kmeans_ref: ray.ObjectRef,
    # kmeans_batch_size: int, # Needed if using compile_nearest_cluster
    cluster_col_name: str
) -> pd.DataFrame:
    """Applies vectorizer (transform) and kmeans (predict) to a batch."""
    if batch.empty:
        return batch
    vectorizer = ray.get(vectorizer_ref) # Retrieve models from Object Store (Ray caches locally after first get)
    kmeans = ray.get(kmeans_ref)

    # if vectorizer is None or kmeans is None:
    #     assert False

    texts = batch["text"].tolist()
    # 1. Vectorize (Transform)
    embeddings = vectorizer.transform(texts)
    # 2. Predict Cluster
    batch[cluster_col_name] = kmeans.predict(embeddings)


    return batch

@ray.remote
def process_stage2_group(
    group_df: pd.DataFrame,
    cluster_a_id: int,
    cfg: object,
) -> Tuple[int, ray.ObjectRef]:
    """Samples, fits Stage 2 models for a group, returns cluster ID and model reference.
    
    Args:
        group_df: DataFrame containing data for this cluster group
        cluster_a_id: The cluster ID from stage 1
        cfg: Configuration object
        
    Returns:
        Tuple of (cluster_a_id, models_ref) where models_ref is a Ray ObjectRef 
        pointing to a tuple of (vectorizer, kmeans) models
    """
    n_clusters_b = cfg.cluster_layout[1]
    max_docs_sample = cfg.get("stage2_max_docs_sample", cfg.max_docs)
    stage_label = f"Stage2_A={cluster_a_id}"

    if group_df.empty:
        print(f"Warning [{stage_label}]: Empty group received.")
        return cluster_a_id, None

    sample_df = group_df.sample(n=min(len(group_df), max_docs_sample), random_state=42)

    # Use the generic fitting task
    models_ref = fit_models_remote.remote(
        cfg=cfg,
        sample_data=sample_df,
        n_clusters=n_clusters_b,
        stage_label=stage_label,
        kmeans_train_batch_size_key="stage2_train_kmeans_bs"
    )
    # Don't try to unpack the Ray object reference
    
    print(f"[{stage_label}] Model fitting tasks submitted.")
    # We return the cluster_id and the reference to the models
    return cluster_a_id, models_ref

def run_clustering_pipeline(ds, cfg: object):
    """Runs the full 2-stage clustering pipeline using Ray."""
    limit = cfg.get("ray_max_docs_limit", None)
    if limit:
         ds = ds.limit(limit)
         print(f"Dataset limited to {limit} documents.")
         
    # --- Stage 1: Train and Infer ---
    print("--- Stage 1 Starting ---")
    n_clusters_a = cfg.cluster_layout[0]
    
    # Sample for Stage 1 Training
    sample_fraction = min(1.0, cfg.max_docs / ds.count()) if ds.count() > 0 else 0.0
    print(f"Sampling {sample_fraction:.2%} ({cfg.max_docs} max) for Stage 1 training...")
    
    print("Stage 1 model fitting task submitted.")
    sample_ds = ds.random_sample(fraction=sample_fraction)
    # Collect sample - check memory constraints if max_docs is huge
    print(f"Collecting sample...")
    sample_df = sample_ds.to_pandas()
    print(f"Sample size: {len(sample_df)}")
    
    # Add detailed logging for resources
    print(f"Available Ray cluster resources before Stage 1 training:")
    resources_available = ray.available_resources()
    for resource, amount in resources_available.items():
        print(f"  - {resource}: {amount}")
    print(f"Requesting {cfg.stage1_train_cpus} CPUs for Stage 1 training task")
    print(f"Ray cluster status: {ray.cluster_resources()}")
    
    # Try to clear caches and force release resources
    print("Clearing Ray object store and releasing resources...")
    ray.runtime_context.get_runtime_context().free_plasma_memory()
    time.sleep(2)  # Give Ray time to release resources
    print(f"Available resources after cache clearing: {ray.available_resources()}")

    # Fit Stage 1 models remotely
    print(f"Attempting to allocate {cfg.stage1_train_cpus} CPUs for Stage 1 training task...")
    try:
        # Try with requested CPUs first
        models_s1_ref = fit_models_remote.options(
                num_cpus=cfg.stage1_train_cpus
        ).remote(
                cfg, sample_df, n_clusters_a, "Stage1", "stage1_train_kmeans_bs"
        )
        
        # Start a monitoring loop to check task status
        print("Monitoring task status...")
        start_monitor = time.time()
        while time.time() - start_monitor < 30:  # Wait up to 30 seconds
            try:
                # Try to get task status without blocking
                ready_refs, _ = ray.wait([models_s1_ref], timeout=1)
                if ready_refs:
                    print("Task started successfully!")
                    break
            except Exception as e:
                print(f"Error monitoring task: {e}")
            
            print(f"Task not started yet, available resources: {ray.available_resources().get('CPU', 0)} CPUs")
            time.sleep(2)
        else:
            # If we reached here, task didn't start in time
            print("Task didn't start in allocated time. Trying with fewer CPUs...")
            # Cancel the original task
            ray.cancel(models_s1_ref)
            # Try with fewer CPUs
            reduced_cpus = max(4, cfg.stage1_train_cpus // 2)
            print(f"Retrying with {reduced_cpus} CPUs instead...")
            models_s1_ref = fit_models_remote.options(
                    num_cpus=reduced_cpus
            ).remote(
                    cfg, sample_df, n_clusters_a, "Stage1", "stage1_train_kmeans_bs"
            )
    except Exception as e:
        print(f"Error allocating resources: {e}")
        print("Falling back to minimal resource allocation...")
        models_s1_ref = fit_models_remote.options(
                num_cpus=4  # Minimum viable CPUs
        ).remote(
                cfg, sample_df, n_clusters_a, "Stage1", "stage1_train_kmeans_bs"
        )
    
    # Get the actual models from the reference
    vectorizer_s1, kmeans_s1 = ray.get(models_s1_ref)
    
    # Put them back as separate references
    vectorizer_s1_ref = ray.put(vectorizer_s1)
    kmeans_s1_ref = ray.put(kmeans_s1)
    
    # Inference Stage 1
    print("Running Stage 1 inference...")
    map_s1_func = partial(apply_models_batch,
                          vectorizer_ref=vectorizer_s1_ref,
                          kmeans_ref=kmeans_s1_ref,
                          cluster_col_name=CLUSTER_A_COL)

    tagged_ds_A = ds.map_batches(
        map_s1_func,
        batch_format="pandas",
        batch_size=cfg.stage1_inf_batch_size,
    )
    
    # tagged_ds_A = tagged_ds_A.materialize()
    print("Stage 1 inference complete. Schema:", tagged_ds_A.schema())
    print("Sample row after Stage 1:", tagged_ds_A.take(1)) # Debug
    print("--- Stage 1 Done ---")


    # --- Stage 2: Train and Infer ---
    print("--- Stage 2 Starting ---")
    # Train Stage 2 models (one per Stage 1 cluster) using map_groups
    print("Training Stage 2 models (one per Stage 1 cluster)...")
    stage2_group_processor_partial = partial(process_stage2_group, cfg=cfg)
    
    # process_stage2_group returns (cluster_a_id, models_ref)
    stage2_model_results_ds = tagged_ds_A.groupby(CLUSTER_A_COL).map_groups(
        stage2_group_processor_partial,
        ray_remote_args={
             "num_cpus": cfg.stage2_train_cpus,
        },
        batch_format="pandas"
    )

    # Collect the model references (assume num stage 1 clusters is manageable)
    print("Collecting Stage 2 model references...")
    stage2_model_results = stage2_model_results_ds.take_all() # List of dicts
    
    # Build dictionary mapping cluster_A ID to model refs
    # Results look like: [{'cluster_A': 0, 'map_groups_output': models_ref}, ...]
    stage2_models_dict = {
        item[CLUSTER_A_COL]: item['map_groups_output']
        for item in stage2_model_results
    }
    stage2_models_dict_ref = ray.put(stage2_models_dict) # Put the whole dict in object store
    print(f"Stage 2 models references collected for {len(stage2_models_dict)} clusters.")
    # print("Stage 2 Model Dict Sample:", dict(list(stage2_models_dict.items())[:2])) # Debug
    
    print("Running Stage 2 inference...")
    def apply_stage2_batch(batch: pd.DataFrame, models_dict_ref) -> pd.DataFrame:
        models_dict = ray.get(models_dict_ref) # Get dict {cluster_id: models_ref}
        batch[CLUSTER_B_COL] = -1 # Initialize column
        
        # Process each cluster_A group within the batch
        for cluster_a_id, group in batch.groupby(CLUSTER_A_COL):
            models_ref = models_dict.get(cluster_a_id)
            
            # Skip if no models available for this cluster
            if models_ref is None:
                print(f"Warning: No models available for cluster_A={cluster_a_id}")
                continue
                
            # Get the actual models from the reference
            try:
                vectorizer, kmeans = ray.get(models_ref)
                
                # Skip if any model is None
                if vectorizer is None or kmeans is None:
                    print(f"Warning: Missing model components for cluster_A={cluster_a_id}")
                    continue
                    
                processed_group = apply_models_batch(
                    group.copy(), # Pass copy to avoid modifying original slice
                    vectorizer_ref=ray.put(vectorizer),
                    kmeans_ref=ray.put(kmeans),
                    cluster_col_name=CLUSTER_B_COL # Predict into the target col
                )
                batch.loc[group.index, CLUSTER_B_COL] = processed_group[CLUSTER_B_COL]
            except Exception as e:
                print(f"Error processing cluster_A={cluster_a_id}: {str(e)}")
                continue

        return batch

    tagged_ds_B = tagged_ds_A.map_batches(
        partial(apply_stage2_batch, models_dict_ref=stage2_models_dict_ref),
        batch_format="pandas",
        batch_size=cfg.stage2_inf_batch_size
    )
    
    final_ds = tagged_ds_B.materialize()
    print("Stage 2 inference complete. Schema:", final_ds.schema())
    print("Sample row after Stage 2:", final_ds.take(1)) # Debug
    print("--- Stage 2 Done ---")

    print("--- Writing Final Output ---")
    # Define output path based on config
    output_base_path = f"{cfg.base_dir}/ray_output_final_clustered" 
    print(f"Writing final partitioned data to: {output_base_path}")
    os.makedirs(output_base_path, exist_ok=True)

    # Write to parquet, ideally partitioned by the cluster assignments
    # This creates directories like: .../cluster_A=0/cluster_B=0/cluster_C=0/file.parquet
    # Note: Partitioning requires columns to exist. Handle potential missing CLUSTER_C_COL if groups failed.
    # It might be safer to write without partitioning first, or ensure default values.
    print(f"Final dataset successfully written to {output_base_path}")
    final_ds.write_parquet(
        output_base_path,
        # Ray automatically handles partitioning based on directory structure
        # partition_cols=[CLUSTER_A_COL, CLUSTER_B_COL, CLUSTER_C_COL] # Specify if needed explicitly
    )
    print("--- Pipeline Finished ---")


# if __name__ == "__main__":
    # --- Configuration ---
    # Set the path to your configuration file
    
# from config_dict import config_dict
from ml_collections import config_dict
import glob
def tfidf_minhash_ray(args):

    dummy_config = {
        "base_dir": "/tmp/ray_clustering_output",
        "cluster_layout": [5, 3, 2], # Smaller example layout
        "max_docs": 5000, # Sample size for training
        "stage1_train_kmeans_bs": 1024,
        "stage1_inf_kmeans_bs": 4096, # Needed if using JAX prediction
        "stage1_inf_batch_size": 1000, # Ray batch size for inference
        "stage1_train_cpus": 16, # Reduced from 70 to fit cluster capacity
        "stage2_train_kmeans_bs": 512,
        "stage2_inf_kmeans_bs": 2048, # Needed if using JAX prediction
        "stage2_inf_batch_size": 1000,
        "stage2_train_cpus": 16, # Reduced from 70 to fit cluster capacity
        "stage3_train_kmeans_bs": 256,
        "stage3_inf_kmeans_bs": 1024, # Needed if using JAX prediction
        "stage3_proc_cpus": 8, # Reduced from 30 to fit cluster capacity
        "stage3_min_group_size": 50, # Min size for Stage 3 processing
        "tfidf_batch_size": 500, # Default batch size if others not set
        "stage3_dedup": True,
        "similarity": args.threshold if args.threshold else 0.85,
        "num_perm": args.num_perm if args.num_perm else 128,
        "ngram_size": args.ngram_size if args.ngram_size else 5,
        "min_ngram_size": args.min_ngram_size if args.min_ngram_size else 1,
        "ray_max_docs_limit": 10000 # Limit total docs processed (for testing)
    }
    
    cfg = config_dict.ConfigDict(dummy_config)

    # --- Run Pipeline ---
    start_time = time.time()
    
    # Prepare the input data
    # If column name is not 'text', rename it
    # if args.column and args.column != 'text':
    #     df = df.withColumnRenamed(args.column, 'text')
    if args.limit_files is not None:
        input_file = glob.glob(args.input_file)[:args.limit_files]
    # Convert Spark DataFrame to Ray Dataset
    print(f"Converting Spark DataFrame to Ray Dataset...")
    # ray_df = ray.data.from_spark(df, parallelism=100)
    
    
    ray_df = ray.data.read_json(input_file,override_num_blocks=1000)
    
    
    print(f"Ray Dataset created with {ray_df.count()} rows")
    
    # Run the clustering pipeline
    run_clustering_pipeline(ray_df, cfg)
    
    end_time = time.time()
    print(f"Total pipeline execution time: {end_time - start_time:.2f} seconds")
    
    # Return the output path where results are stored
    return f"{cfg.base_dir}/ray_output_final_clustered"
    
def test_kmeans():
    """Test the KMeans implementation with a simple example."""
    print("Testing KMeans implementation...")
    
    # Generate sample data with clear clusters
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_clusters = 3
    
    # Create clear clusters
    centers = np.random.randn(n_clusters, n_features) * 5  # Well-separated centers
    
    # Generate points around centers
    X = np.vstack([
        centers[i] + np.random.randn(n_samples // n_clusters, n_features)
        for i in range(n_clusters)
    ])
    
    # Shuffle the data
    np.random.shuffle(X)
    
    # Test standard KMeans (without balanced clusters)
    print("\nTesting standard KMeans...")
    kmeans_standard = KMeans(n_clusters=n_clusters, balanced=False, random_state=42)
    labels_standard = kmeans_standard.fit_predict(X)
    
    # Count points in each cluster
    unique, counts = np.unique(labels_standard, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    print(f"Cluster distribution (standard): {cluster_counts}")
    
    # Test balanced KMeans
    print("\nTesting balanced KMeans...")
    kmeans_balanced = KMeans(n_clusters=n_clusters, balanced=True, random_state=42)
    labels_balanced = kmeans_balanced.fit_predict(X)
    
    # Count points in each cluster
    unique, counts = np.unique(labels_balanced, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    print(f"Cluster distribution (balanced): {cluster_counts}")
    
    # Test online learning
    print("\nTesting online learning...")
    kmeans_online = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Split data into batches
    batch_size = 200
    n_batches = n_samples // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = X[start_idx:end_idx]
        
        # Fit batch
        if i == 0:
            kmeans_online.fit(batch)
        else:
            kmeans_online.fit(batch, online=True, iter_k=i)
    
    # Predict on full dataset
    labels_online = kmeans_online.predict(X)
    
    # Count points in each cluster
    unique, counts = np.unique(labels_online, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    print(f"Cluster distribution (online): {cluster_counts}")
    
    print("\nKMeans testing complete!")
    
    return kmeans_standard, kmeans_balanced, kmeans_online

if __name__ == "__main__":
    # Run KMeans test if this file is executed directly
    test_kmeans()




    
    
    