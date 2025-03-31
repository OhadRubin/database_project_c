"""
<note> we have to use scikit-learn components within Spark because Spark does not offer sparse SVD. </note>
TF-IDF Vectorization Module for Spark

This module implements TF-IDF vectorization in Spark using scikit-learn. It follows three strategies: (1) train on sample data collected to one executor, (2) broadcast trained model to all executors, (3) apply model with mapPartitions to avoid reloading.
"""


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
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
# Import Spark ML components
from pyspark.ml.linalg import Vectors as MLVectors, VectorUDT
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.storagelevel import StorageLevel
import socket

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Get the logger for TfidfVectorizer
tfidf_logger = logging.getLogger('sklearn.feature_extraction.text')
# Ignore warnings about stop words inconsistency
import warnings
warnings.filterwarnings('ignore', message="Your stop_words may be inconsistent with your preprocessing.*", category=UserWarning)




def number_normalizer(tokens):
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))
    


def get_sklearn_feature_pipeline(n_components, random_seed):
    stop_words = list(ENGLISH_STOP_WORDS.union(["#NUMBER"]))
    vectorizer = Pipeline([('tfidf', NumberNormalizingVectorizer(stop_words=stop_words)),
                            ('svd', TruncatedSVD(n_components=n_components,random_state=random_seed)),
                            ('normalizer', Normalizer(copy=False))],
                            verbose=True)
    return vectorizer

# --- TF-IDF Vectorization Function (using sklearn fit/transform) ---
# Globals for Lazy Initialization on Executors for the pipeline transform part
EXECUTOR_PIPELINE: Optional[Pipeline] = None

import more_itertools
def _transform_partition_sklearn(
    rows_iter: Iterator[Row],
    bc_pipeline_broadcast: Any, # Spark Broadcast object
    column: str,
    batch_size: int
    ) -> Iterator[Tuple[int, List[float]]]:
    """Helper function run by mapPartitions to transform data using broadcasted sklearn pipeline."""
    global EXECUTOR_PIPELINE
    print(f"Executor node {socket.gethostname()} accessing broadcast data")
    # Lazy init of model from broadcast *once per worker*
    
    if EXECUTOR_PIPELINE is None:
        pid = os.getpid()
        try:
            # print(f"EXECUTOR ({pid}): Initializing pipeline from broadcast {bc_pipeline_broadcast.id}") # Debug
            EXECUTOR_PIPELINE = bc_pipeline_broadcast.value
        except Exception as e:
            print(f"EXECUTOR ({pid}): ERROR initializing pipeline: {e}") # Use print as logger might not be setup
            raise RuntimeError("Executor pipeline initialization failed") from e

    # Process partition in batches using more_itertools.chunked
    for batch in more_itertools.chunked(rows_iter, batch_size):
        batch_texts = [x[column] for x in batch]
        batch_ids = [x['__id__'] for x in batch]
                
        features = EXECUTOR_PIPELINE.transform(batch_texts)
        for i, doc_id_in_batch in enumerate(batch_ids):
            yield (doc_id_in_batch, features[i])
        

    # Process final partial batch

def train_sklearn_vectorization( df: DataFrame, column: str, n_components: int, random_seed: int = 42, max_sample_fit: int = 5000, ) -> DataFrame:
    """Fits sklearn TF-IDF/SVD on sample, transforms full DataFrame."""
    print(f"Starting Sklearn Vectorization Training: Components={n_components}, SampleFit={max_sample_fit}")
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
    return pipeline, df_with_id
    
def inference_sklearn_vectorization( spark: SparkSession, pipeline: Pipeline, df_with_id: DataFrame, column: str, map_partitions_batch_size ) -> DataFrame:
    

    # Broadcast pipeline
    bc_pipeline = spark.sparkContext.broadcast(pipeline)
    print(f"Broadcasted fitted pipeline.")
    

    # Define schema for output RDD
    from pyspark.sql.types import StructType, StructField, LongType, ArrayType, DoubleType
    vector_schema = StructType([
        StructField("__id__", LongType(), False),
        StructField("tfidf_features", ArrayType(DoubleType()), False)
    ])

    # Transform full dataset
    print("Applying distributed transformation...")
    
    print(f"Starting Sklearn Vectorization Inference: BatchSize={map_partitions_batch_size}")
    transform_start_time = time.time()
    # This code transforms the text data into TF-IDF vectors in a distributed manner:
    # 1. Selects only the document ID and text column from the DataFrame
    # 2. Converts to an RDD for parallel processing
    # 3. Uses mapPartitions to process data in batches within each partition
    # 4. Applies the broadcasted sklearn pipeline to transform text to vectors
    # 5. Returns (document_id, vector) pairs that will be converted to a DataFrame
    vectorized_rdd = df_with_id.select("__id__", column).rdd.mapPartitions(
        lambda rows_iter: _transform_partition_sklearn(rows_iter, bc_pipeline, column, map_partitions_batch_size)
    )
    vector_df = spark.createDataFrame(vectorized_rdd, schema=vector_schema)
    vector_df.show()
    print(f"Distributed transformation took {time.time() - transform_start_time:.2f}s.")
    df_with_id.unpersist() # Unpersist input now
    # vector_df.collect()
    return vector_df


# --- Main Deduplication Function ---
def tfidf_minhash(
    spark: SparkSession,
    df: DataFrame,
    column: str,
    num_perm: int = None, # Unused
    ngram_size: int = None, # Unused
    min_ngram_size: int = None, # Unused
    threshold: float = 0.8,
    n_components: int = 128,
    ) -> Tuple[DataFrame, int]:
    """
    Performs TF-IDF (sklearn) -> KMeans (Spark ML) -> Per-Cluster Similarity Check (Driver).
    """
    print(f"Starting TF-IDF + Clustering Deduplication: Threshold={threshold}, N_Components={n_components}")
    overall_start_time = time.time()
    original_cols = df.columns # Keep track of original columns
    df = df.repartition(100)

    # === Step 1: TF-IDF Vectorization (Sklearn Fit/Transform) ===
    pipeline, df_with_id = train_sklearn_vectorization(df, column, n_components)
    vector_df = inference_sklearn_vectorization(spark, pipeline, df_with_id, column, map_partitions_batch_size=1000)


    # # === Step 2: Convert to Spark ML Vectors ===
    to_vector_udf = F.udf(lambda x: MLVectors.dense(x) if x else None, VectorUDT())
    vector_df_ml = vector_df.withColumn("features", to_vector_udf(F.col("tfidf_features")))
    vector_df_ml = vector_df_ml.filter(F.col("features").isNotNull()).select("__id__", "features", "tfidf_features")
    # Drop the tfidf_features column as it's no longer needed
    vector_df_ml = vector_df_ml.drop("tfidf_features")
    
    # === Step 2.5: Join with original data ===
    print("Joining vector features with original data")
    join_start_time = time.time()
    # Join the ML vector features back with the original dataframe
    vector_df.unpersist() # Unpersist previous stage
    # This ensures we have both the features and all original columns
    joined_df = vector_df_ml.join(df_with_id, on="__id__", how="inner")
    print(f"Join operation completed in {time.time() - join_start_time:.2f}s.")
    print(f"Joined dataframe has {joined_df.count()} rows")
    joined_df.collect()
    joined_df.show()
    
    # if vector_df_ml.rdd.isEmpty():
    #     print("No valid Spark ML vectors created. Returning original data.")
    #     vector_df.unpersist()
    #     return df, 0
    # vector_df_ml.persist(StorageLevel.MEMORY_AND_DISK) # Cache ML vectors

    # # === Step 3: Spark ML KMeans Clustering ===
    # num_records = vector_df_ml.count()
    # k = max(2, min(20, int(math.sqrt(num_records / 100)))) # Heuristic for K
    # print(f"Running Spark ML KMeans with k={k} on {num_records} records.")
    kmeans = SparkKMeans(k=10, seed=42, featuresCol="features", predictionCol="prediction", maxIter=20)
    kmeans_start_time = time.time()
    kmeans_model = kmeans.fit(joined_df)
    print(f"KMeans fitting took {time.time() - kmeans_start_time:.2f}s.")
    clustered_df = kmeans_model.transform(vector_df_ml)
    clustered_df = clustered_df.drop("features")
    
    clustered_df.show()
    clustered_df.collect()
    
    # # Keep __id__, original tfidf_features (needed for similarity), and prediction
    # clustered_df = clustered_df.select("__id__", "tfidf_features", "prediction")
    # clustered_df.persist(StorageLevel.MEMORY_AND_DISK) # Cache clustered results
    # vector_df_ml.unpersist() # Unpersist previous stage

    # # === Step 4: Per-Cluster Duplicate Detection (Driver-Side) ===
    # cluster_ids = [row.prediction for row in clustered_df.select("prediction").distinct().collect()]
    # print(f"Found {len(cluster_ids)} distinct clusters. Processing each for duplicates...")
    # all_ids_to_remove = set()
    # total_duplicates_found = 0
    # cluster_processing_start_time = time.time()

    # for i, cluster_id in enumerate(cluster_ids):
    #     # log.debug(f"--- Processing Cluster {cluster_id} ({i+1}/{len(cluster_ids)}) ---")
    #     # Select data for the current cluster
    #     # IMPORTANT: This filter + collect inside the loop can be inefficient if many clusters
    #     cluster_data_df = clustered_df.filter(F.col("prediction") == cluster_id).select("__id__", "tfidf_features")
    #     # Execute the driver-side similarity check
    #     ids_to_remove_in_cluster, duplicates_in_cluster = find_duplicates_within_cluster_driver(
    #         cluster_data_df, "tfidf_features", threshold=threshold, id_col="__id__"
    #     )
    #     if duplicates_in_cluster > 0:
    #         print(f"Cluster {cluster_id}: Found {duplicates_in_cluster} duplicates to remove.")
    #         all_ids_to_remove.update(ids_to_remove_in_cluster)
    #         total_duplicates_found += duplicates_in_cluster
    #     # No need to cache cluster_data_df as it's collected immediately

    # print(f"Cluster processing took {time.time() - cluster_processing_start_time:.2f}s.")
    # clustered_df.unpersist() # Unpersist clustered data

    # # === Step 5: Filter Original DataFrame ===
    # print(f"Total duplicates identified across all clusters: {total_duplicates_found}")
    # print(f"Total unique documents to remove: {len(all_ids_to_remove)}")

    # # Ensure original DF has __id__ if it wasn't added before vectorization
    # if "__id__" not in df.columns:
    #      df_with_id = df.withColumn("__id__", F.monotonically_increasing_id())
    #      print("Original DataFrame did not have '__id__', adding it now for filtering.")
    # else:
    #      df_with_id = df

    # if all_ids_to_remove:
    #     final_df = df_with_id.filter(~F.col("__id__").isin(list(all_ids_to_remove)))
    # else:
    #     print("No duplicates found matching threshold.")
    #     final_df = df_with_id # Return original data if no duplicates

    # # Select only the original columns, dropping temporary __id__
    # final_df = final_df.select(*original_cols)

    # overall_end_time = time.time()
    # print(f"TF-IDF + Clustering Deduplication completed in {overall_end_time - overall_start_time:.2f}s.")

    return None, 0

