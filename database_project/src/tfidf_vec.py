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
    # print(f"Executor node {socket.gethostname()} accessing broadcast data")
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
            yield (doc_id_in_batch, features[i].tolist())
        

    # Process final partial batch

def train_sklearn_vectorization( df: DataFrame, column: str, n_components: int, random_seed: int = 42, max_sample_fit: int = 50000, ) -> DataFrame:
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
    return vector_df

def tfidf_cluster(spark: SparkSession, df: DataFrame, column: str, n_components: int, k: int):
    # === Step 1: TF-IDF Vectorization (Sklearn Fit/Transform) ===
    pipeline, df_with_id = train_sklearn_vectorization(df, column, n_components)
    vector_df = inference_sklearn_vectorization(spark, pipeline, df_with_id, column, map_partitions_batch_size=10000)


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
    vector_df = vector_df_ml.join(df_with_id, on="__id__", how="inner")
    print(f"Join operation completed in {time.time() - join_start_time:.2f}s.")
    print(f"Joined dataframe has {vector_df.count()} rows")
    
    # print(f"Running Spark ML KMeans with k={k} on {num_records} records.")
    kmeans = SparkKMeans(k=k, seed=42, featuresCol="features", predictionCol="prediction", maxIter=20)
    kmeans_start_time = time.time()
    sample_df = vector_df.limit(10000)
    kmeans_model = kmeans.fit(sample_df)
    print(f"KMeans fitting took {time.time() - kmeans_start_time:.2f}s.")
    kmeans_inf_time = time.time()
    # vector_df = vector_df.repartition(10000)
    clustered_df = kmeans_model.transform(vector_df)
    print(f"KMeans inference took {time.time() - kmeans_inf_time:.2f}s.")
    clustered_df = clustered_df.drop("features")
    return clustered_df


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
    k: int = 10,
    ) -> Tuple[DataFrame, int]:
    """
    Performs TF-IDF (sklearn) -> KMeans (Spark ML) -> Per-Cluster Similarity Check (Driver).
    """
    print(f"Starting TF-IDF + Clustering Deduplication: Threshold={threshold}, N_Components={n_components}")
    overall_start_time = time.time()
    original_cols = df.columns # Keep track of original columns
    df = df.repartition(100)

    clustered_df = tfidf_cluster(spark, df, column, n_components, k)
    
    
    clustered_df.show()
    
    
    # Sort the dataframe by prediction (cluster ID)
    print("Sorting dataframe by prediction (cluster ID)")
    sort_start_time = time.time()
    clustered_df = clustered_df.orderBy("prediction")
    print(f"Sorting completed in {time.time() - sort_start_time:.2f}s.")
    # clustered_df.collect()
    print(f"Overall deduplication took {time.time() - overall_start_time:.2f}s.")
    

    return clustered_df, 0

