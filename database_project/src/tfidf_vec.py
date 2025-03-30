import os
import numpy as np
import pandas as pd # Used for type hints, not core logic
from pyspark import SparkConf
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql import functions as F
import logging
import math
import time
from functools import reduce
from typing import Dict, Any, Iterator, List, Tuple, Optional, Set

# Import scikit-learn components
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

# Import Spark ML components
from pyspark.ml.linalg import Vectors as MLVectors, VectorUDT
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.storagelevel import StorageLevel




# --- Scikit-learn Pipeline Definition ---
class NumberNormalizingVectorizer(TfidfVectorizer):
    """Custom TF-IDF Vectorizer that replaces numbers with a token."""
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(self._number_normalizer(tokenize(doc)))
    def _number_normalizer(self, tokens: List[str]) -> Iterator[str]:
        for token in tokens:
            if any(char.isdigit() for char in token): yield "#NUMBER";
            else: yield token

def get_sklearn_feature_pipeline(n_components: int, random_seed: int) -> Pipeline:
    """Creates the TF-IDF -> SVD pipeline."""
    stop_words = set(ENGLISH_STOP_WORDS.union(["#NUMBER"]))
    pipeline = Pipeline([
        ('tfidf', NumberNormalizingVectorizer(
            stop_words=list(stop_words), max_features=10000, max_df=0.95, min_df=2
        )),
        ('svd', TruncatedSVD(n_components=n_components, random_state=random_seed))
    ])
    return pipeline

# --- TF-IDF Vectorization Function (using sklearn fit/transform) ---
# Globals for Lazy Initialization on Executors for the pipeline transform part
EXECUTOR_PIPELINE: Optional[Pipeline] = None

def _transform_partition_sklearn(
    rows_iter: Iterator[Row],
    bc_pipeline_broadcast: Any, # Spark Broadcast object
    column: str,
    batch_size: int
    ) -> Iterator[Tuple[int, List[float]]]:
    """Helper function run by mapPartitions to transform data using broadcasted sklearn pipeline."""
    global EXECUTOR_PIPELINE

    # Lazy init of model from broadcast *once per worker*
    if EXECUTOR_PIPELINE is None:
        pid = os.getpid()
        try:
            # print(f"EXECUTOR ({pid}): Initializing pipeline from broadcast {bc_pipeline_broadcast.id}") # Debug
            EXECUTOR_PIPELINE = bc_pipeline_broadcast.value
        except Exception as e:
            print(f"EXECUTOR ({pid}): ERROR initializing pipeline: {e}") # Use print as logger might not be setup
            raise RuntimeError("Executor pipeline initialization failed") from e

    # Process partition in batches
    batch_texts = []
    batch_ids = []
    for row in rows_iter:
        doc_id = row['__id__']
        text = row[column]
        if text: # Ensure text is not None or empty
            batch_texts.append(text)
            batch_ids.append(doc_id)

        if len(batch_texts) >= batch_size:
            if batch_texts:
                try:
                    features = EXECUTOR_PIPELINE.transform(batch_texts)
                    for i, doc_id_in_batch in enumerate(batch_ids):
                        yield (doc_id_in_batch, features[i].tolist())
                except Exception as e:
                     print(f"EXECUTOR ({os.getpid()}): ERROR transforming batch: {e}") # Log error
            batch_texts, batch_ids = [], [] # Reset batch

    # Process final partial batch
    if batch_texts:
        try:
            features = EXECUTOR_PIPELINE.transform(batch_texts)
            for i, doc_id_in_batch in enumerate(batch_ids):
                yield (doc_id_in_batch, features[i].tolist())
        except Exception as e:
            print(f"EXECUTOR ({os.getpid()}): ERROR transforming final batch: {e}")

def run_sklearn_vectorization(
    spark_session: SparkSession,
    df: DataFrame,
    column: str,
    n_components: int,
    random_seed: int = 42,
    max_sample_fit: int = 50000,
    map_partitions_batch_size: int = 100
    ) -> DataFrame:
    """Fits sklearn TF-IDF/SVD on sample, transforms full DataFrame."""
    print(f"Starting Sklearn Vectorization: Components={n_components}, SampleFit={max_sample_fit}, BatchSize={map_partitions_batch_size}")
    fit_start_time = time.time()

    # Ensure __id__ exists, cache input
    if "__id__" not in df.columns:
        df_with_id = df.withColumn("__id__", F.monotonically_increasing_id())
    else:
        df_with_id = df
    df_with_id.persist(StorageLevel.MEMORY_AND_DISK) # Persist input for sampling and joining later
    df_count = df_with_id.count()
    print(f"Input count: {df_count}")

    # Fit pipeline on sample
    sample_size = min(df_count, max_sample_fit)
    text_sample_rows = df_with_id.select(column).limit(sample_size).collect()
    text_sample = [row[column] for row in text_sample_rows if row and row[column]]
    if not text_sample: raise ValueError("No non-empty text found in sample for fitting.")
    print(f"Fitting sklearn pipeline on {len(text_sample)} documents...")
    pipeline = get_sklearn_feature_pipeline(n_components, random_seed)
    pipeline.fit(text_sample)
    print(f"Pipeline fitting took {time.time() - fit_start_time:.2f}s.")

    # Broadcast pipeline
    bc_pipeline = spark_session.sparkContext.broadcast(pipeline)
    print(f"Broadcasted fitted pipeline (ID: {bc_pipeline.id}).")

    # Define schema for output RDD
    from pyspark.sql.types import StructType, StructField, LongType, ArrayType, DoubleType
    vector_schema = StructType([
        StructField("__id__", LongType(), False),
        StructField("tfidf_features", ArrayType(DoubleType()), False)
    ])

    # Transform full dataset
    print("Applying distributed transformation...")
    transform_start_time = time.time()
    vectorized_rdd = df_with_id.select("__id__", column).rdd.mapPartitions(
        lambda rows_iter: _transform_partition_sklearn(rows_iter, bc_pipeline, column, map_partitions_batch_size)
    )
    vector_df = spark_session.createDataFrame(vectorized_rdd, schema=vector_schema)

    print(f"Distributed transformation took {time.time() - transform_start_time:.2f}s.")
    df_with_id.unpersist() # Unpersist input now
    return vector_df

# --- Duplicate Finding Function (Driver-Side - SCALES POORLY) ---
def find_duplicates_within_cluster_driver(
    cluster_df: DataFrame,
    vector_column: str,
    threshold: float,
    id_col: str = "__id__"
    ) -> Tuple[Set[int], int]:
    """Finds duplicates within a cluster by collecting vectors to the driver."""
    print(f"Executing find_duplicates_within_cluster_driver - collects data to driver!")
    collect_start_time = time.time()
    try:
        # Collect IDs and vectors for the current cluster
        id_vector_list = cluster_df.select(id_col, vector_column).collect()
        num_vectors = len(id_vector_list)
        print(f"Collected {num_vectors} vectors to driver (took {time.time() - collect_start_time:.2f}s).")
        if num_vectors <= 1: return set(), 0 # Base case: no duplicates possible
    except Exception as e:
        print(f"Failed to collect vectors for cluster: {e}")
        return set(), 0 # Return empty if collection fails

    id_to_vector = {row[id_col]: np.array(row[vector_column], dtype=np.float64) for row in id_vector_list}
    id_list = list(id_to_vector.keys())

    # Compute similarities efficiently using sklearn if available and dense
    # Otherwise, fallback to loop (as in original)
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        vectors_matrix = np.array([id_to_vector[id] for id in id_list])
        print("Calculating similarity matrix using sklearn...")
        sim_matrix = cosine_similarity(vectors_matrix)
        # Set diagonal to 0 and find pairs above threshold
        np.fill_diagonal(sim_matrix, 0)
        indices_row, indices_col = np.where(sim_matrix >= threshold)
        duplicate_pairs = set(tuple(sorted((id_list[r], id_list[c]))) for r, c in zip(indices_row, indices_col) if r < c)
        print(f"Similarity matrix calculation complete. Found {len(duplicate_pairs)} pairs.")
    except ImportError:
         print("sklearn not available for pairwise similarity, falling back to loop.")
         duplicate_pairs = set()
         for i in range(num_vectors):
             doc_id1 = id_list[i]; vector1 = id_to_vector[doc_id1]; norm1 = np.linalg.norm(vector1)
             if norm1 == 0: continue
             for j in range(i + 1, num_vectors):
                 doc_id2 = id_list[j]; vector2 = id_to_vector[doc_id2]; norm2 = np.linalg.norm(vector2)
                 if norm2 == 0: continue
                 similarity = np.dot(vector1, vector2) / (norm1 * norm2)
                 if similarity >= threshold: duplicate_pairs.add(tuple(sorted((doc_id1, doc_id2))))
         print(f"Pairwise loop complete. Found {len(duplicate_pairs)} pairs.")

    # Build connected components to find groups and identify IDs to remove
    adj = {doc_id: [] for doc_id in id_list}
    for id1, id2 in duplicate_pairs: adj[id1].append(id2); adj[id2].append(id1)

    visited = set()
    representatives = set()
    ids_to_remove = set()
    for doc_id in id_list:
        if doc_id not in visited:
            component = []
            q = [doc_id]; visited.add(doc_id); head = 0
            while head < len(q):
                 u = q[head]; head += 1; component.append(u)
                 for v in adj[u]:
                     if v not in visited: visited.add(v); q.append(v)
            if len(component) > 1:
                representative = min(component) # Keep the one with the smallest ID
                representatives.add(representative)
                for member_id in component:
                    if member_id != representative: ids_to_remove.add(member_id)

    num_duplicates_removed = len(ids_to_remove)
    print(f"Identified {len(representatives)} groups. Docs to remove: {num_duplicates_removed}.")
    return ids_to_remove, num_duplicates_removed

# --- Main Deduplication Function ---
def tfidf_minhash(
    spark_session: SparkSession,
    df: DataFrame,
    column: str,
    num_perm: int = None, # Unused
    ngram_size: int = None, # Unused
    min_ngram_size: int = None, # Unused
    threshold: float = 0.8,
    n_components: int = 128
    ) -> Tuple[DataFrame, int]:
    """
    Performs TF-IDF (sklearn) -> KMeans (Spark ML) -> Per-Cluster Similarity Check (Driver).
    """
    print(f"Starting TF-IDF + Clustering Deduplication: Threshold={threshold}, N_Components={n_components}")
    overall_start_time = time.time()
    original_cols = df.columns # Keep track of original columns

    # === Step 1: TF-IDF Vectorization (Sklearn Fit/Transform) ===
    vector_df = run_sklearn_vectorization(spark_session, df, column, n_components)
    vector_df.persist(StorageLevel.MEMORY_AND_DISK) # Cache vectors
    if vector_df.rdd.isEmpty():
        print("Vectorization resulted in an empty DataFrame. Returning original data.")
        return df, 0

    # === Step 2: Convert to Spark ML Vectors ===
    to_vector_udf = F.udf(lambda x: MLVectors.dense(x) if x else None, VectorUDT())
    vector_df_ml = vector_df.withColumn("features", to_vector_udf(F.col("tfidf_features")))
    vector_df_ml = vector_df_ml.filter(F.col("features").isNotNull()).select("__id__", "features", "tfidf_features")
    if vector_df_ml.rdd.isEmpty():
        print("No valid Spark ML vectors created. Returning original data.")
        vector_df.unpersist()
        return df, 0
    vector_df_ml.persist(StorageLevel.MEMORY_AND_DISK) # Cache ML vectors
    vector_df.unpersist() # Unpersist previous stage

    # === Step 3: Spark ML KMeans Clustering ===
    num_records = vector_df_ml.count()
    k = max(2, min(20, int(math.sqrt(num_records / 100)))) # Heuristic for K
    print(f"Running Spark ML KMeans with k={k} on {num_records} records.")
    kmeans = SparkKMeans(k=k, seed=42, featuresCol="features", predictionCol="prediction", maxIter=20)
    kmeans_start_time = time.time()
    kmeans_model = kmeans.fit(vector_df_ml)
    print(f"KMeans fitting took {time.time() - kmeans_start_time:.2f}s.")
    clustered_df = kmeans_model.transform(vector_df_ml)
    # Keep __id__, original tfidf_features (needed for similarity), and prediction
    clustered_df = clustered_df.select("__id__", "tfidf_features", "prediction")
    clustered_df.persist(StorageLevel.MEMORY_AND_DISK) # Cache clustered results
    vector_df_ml.unpersist() # Unpersist previous stage

    # === Step 4: Per-Cluster Duplicate Detection (Driver-Side) ===
    cluster_ids = [row.prediction for row in clustered_df.select("prediction").distinct().collect()]
    print(f"Found {len(cluster_ids)} distinct clusters. Processing each for duplicates...")
    all_ids_to_remove = set()
    total_duplicates_found = 0
    cluster_processing_start_time = time.time()

    for i, cluster_id in enumerate(cluster_ids):
        log.debug(f"--- Processing Cluster {cluster_id} ({i+1}/{len(cluster_ids)}) ---")
        # Select data for the current cluster
        # IMPORTANT: This filter + collect inside the loop can be inefficient if many clusters
        cluster_data_df = clustered_df.filter(F.col("prediction") == cluster_id).select("__id__", "tfidf_features")
        # Execute the driver-side similarity check
        ids_to_remove_in_cluster, duplicates_in_cluster = find_duplicates_within_cluster_driver(
            cluster_data_df, "tfidf_features", threshold=threshold, id_col="__id__"
        )
        if duplicates_in_cluster > 0:
            print(f"Cluster {cluster_id}: Found {duplicates_in_cluster} duplicates to remove.")
            all_ids_to_remove.update(ids_to_remove_in_cluster)
            total_duplicates_found += duplicates_in_cluster
        # No need to cache cluster_data_df as it's collected immediately

    print(f"Cluster processing took {time.time() - cluster_processing_start_time:.2f}s.")
    clustered_df.unpersist() # Unpersist clustered data

    # === Step 5: Filter Original DataFrame ===
    print(f"Total duplicates identified across all clusters: {total_duplicates_found}")
    print(f"Total unique documents to remove: {len(all_ids_to_remove)}")

    # Ensure original DF has __id__ if it wasn't added before vectorization
    if "__id__" not in df.columns:
         df_with_id = df.withColumn("__id__", F.monotonically_increasing_id())
         print("Original DataFrame did not have '__id__', adding it now for filtering.")
    else:
         df_with_id = df

    if all_ids_to_remove:
        final_df = df_with_id.filter(~F.col("__id__").isin(list(all_ids_to_remove)))
    else:
        print("No duplicates found matching threshold.")
        final_df = df_with_id # Return original data if no duplicates

    # Select only the original columns, dropping temporary __id__
    final_df = final_df.select(*original_cols)

    overall_end_time = time.time()
    print(f"TF-IDF + Clustering Deduplication completed in {overall_end_time - overall_start_time:.2f}s.")

    return final_df, total_duplicates_found


# === Example Usage ===
if __name__ == "__main__":

    # --- Create Simple Dummy Data ---
    BASE_DIR = "/tmp/spark_sklearn_dedup_v8_data"
    INPUT_DATA_PATTERN = os.path.join(BASE_DIR, "input", "*.csv")
    def create_simple_dummy_data(path_pattern: str, num_lines: int, prefix: str):
        base_dir = os.path.dirname(path_pattern.replace("*",""))
        file_path = os.path.join(base_dir, f"{prefix}_data.csv")
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"Creating dummy data: {file_path}")
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write("doc_id_orig,text\n") # Header
                for i in range(num_lines):
                     # Create near duplicates
                     text_content = f"The quick brown fox jumps over the lazy dog sentence number {i % 5}"
                     f.write(f"doc_{i},{text_content} with suffix {i%2}.\n")
                # Add exact duplicates
                f.write(f"doc_{num_lines},The quick brown fox jumps over the lazy dog sentence number {0 % 5} with suffix {0%2}.\n")
                f.write(f"doc_{num_lines+1},The quick brown fox jumps over the lazy dog sentence number {1 % 5} with suffix {1%2}.\n")
        else: print(f"Dummy data exists: {file_path}")
        return base_dir + "/*.csv"

    INPUT_DATA_PATTERN = create_simple_dummy_data(INPUT_DATA_PATTERN, 500, "documents")
    print(f"Using input data pattern: {INPUT_DATA_PATTERN}")

    try:
        # Load data
        input_df = spark.read.option("header", "true").csv(INPUT_DATA_PATTERN)
        input_df.persist()
        original_count = input_df.count()
        print(f"Loaded {original_count} records for deduplication.")
        input_df.show(5, truncate=False)

        # --- Run Deduplication ---
        deduplicated_df, duplicate_count = tfidf_minhash(
            spark_session=spark,
            df=input_df,
            column="text",
            threshold=0.95,  # High threshold for near-exact duplicates
            n_components=32  # Lower components for speed
        )

        final_count = deduplicated_df.count()
        print("\n--- Deduplication Results ---")
        print(f"Original record count: {original_count}")
        print(f"Deduplicated record count: {final_count}")
        print(f"Number of duplicates removed (estimate): {duplicate_count}") # Note this counts pairs leading to removal
        print(f"Actual records removed: {original_count - final_count}")
        print(f"Percentage removed: {((original_count - final_count) / original_count * 100) if original_count > 0 else 0:.2f}%")

        print("Sample of deduplicated data:")
        deduplicated_df.show(5, truncate=False)

        input_df.unpersist()

    except Exception as e:
        print(f"\n--- JOB FAILED: {e} ---", exc_info=True) # Log traceback
    finally:
        print("Stopping Spark session.")
        spark.stop()