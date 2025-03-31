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
import torch
from sklearn.cluster import KMeans


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

import socket

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from tqdm import tqdm

# Get the logger for TfidfVectorizer
tfidf_logger = logging.getLogger('sklearn.feature_extraction.text')
# Ignore warnings about stop words inconsistency
import warnings
warnings.filterwarnings('ignore', message="Your stop_words may be inconsistent with your preprocessing.*", category=UserWarning)

import ray


def number_normalizer(tokens):
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
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
CLUSTER_C_COL = 'cluster_C'
TEMP_ID_COL = 'temp_doc_id'



@ray.remote
def fit_models_remote(
    cfg: object,
    sample_data: pd.DataFrame,
    n_clusters: int,
    stage_label: str, # e.g., "Stage1", "Stage2_GroupX", etc.
    kmeans_train_batch_size_key: str, # e.g., "stage1_train_kmeans_bs"
) -> Tuple[object, object]:
    """Fits vectorizer and KMeans on sample data."""


    texts = sample_data["text"].tolist()
    print(f"[{stage_label}] Fitting vectorizer on {len(texts)} samples...")
    vectorizer = get_sklearn_feature_pipeline(n_components=128, random_seed=42)
    embeddings = vectorizer.fit_transform(texts)
    print(f"[{stage_label}] Vectorizer fitting done. Embedding shape: {embeddings.shape}")

    print(f"[{stage_label}] Fitting K-means with {n_clusters} clusters...")
    kmeans_batch_size_config = getattr(cfg, kmeans_train_batch_size_key)
        
    # kmeans = fit_kmeans(embeddings, n_clusters, batch_size=kmeans_batch_size_config) # Pass computed BS if needed
    kmeans = None
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
    # Convert embeddings if needed by predict (e.g., to tensor or dense numpy)
    batch[cluster_col_name] = np.zeros(embeddings.shape[0])


    return batch

@ray.remote
def process_stage2_group(
    group_df: pd.DataFrame,
    cluster_a_id: int,
    cfg: object,
) -> Tuple[int, Tuple[ray.ObjectRef, ray.ObjectRef]]:
    """Samples, fits Stage 2 models for a group, returns cluster ID and model refs."""
    n_clusters_b = cfg.cluster_layout[1]
    max_docs_sample = cfg.get("stage2_max_docs_sample", cfg.max_docs)
    stage_label = f"Stage2_A={cluster_a_id}"

    if group_df.empty:
        print(f"Warning [{stage_label}]: Empty group received.")
        return cluster_a_id, (None, None)

    sample_df = group_df.sample(n=min(len(group_df), max_docs_sample), random_state=42)

    # Use the generic fitting task
    vectorizer, kmeans = fit_models_remote.remote(
        cfg=cfg,
        sample_data=sample_df,
        n_clusters=n_clusters_b,
        stage_label=stage_label,
        kmeans_train_batch_size_key="stage2_train_kmeans_bs"
    )
    # We get back refs directly here
    vectorizer_ref, kmeans_ref = vectorizer, kmeans

    # Note: If fit_models_remote returned actual objects, we would ray.put them here.
    # Since it's already remote, it returns refs.
    print(f"[{stage_label}] Model fitting tasks submitted.")
    # We return the *refs* to the models, not the models themselves
    return cluster_a_id, (vectorizer_ref, kmeans_ref)

def run_clustering_pipeline(ds, cfg: object):
    """Runs the full 2-stage clustering pipeline using Ray."""
    
    
    


    # Optional: Limit size for testing/development
    limit = cfg.get("ray_max_docs_limit")
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

    # Fit Stage 1 models remotely
    vectorizer_s1_ref, kmeans_s1_ref = fit_models_remote.options(
            num_cpus=cfg.get("stage1_train_cpus", 4) # Configurable resources
            # num_gpus=cfg.get("stage1_train_gpus", 0)
    ).remote(
            cfg, sample_df, n_clusters_a, "Stage1", "stage1_train_kmeans_bs"
    )


    # Check if models were submitted successfully before proceeding
    if vectorizer_s1_ref is None or kmeans_s1_ref is None:
         print("Error: Stage 1 model fitting task failed to submit or returned None early. Aborting.")
         # Wait for potential remote error messages if needed? Or just exit.
         ray.shutdown()
         return
         
    # Inference Stage 1
    print("Running Stage 1 inference...")
    map_s1_func = partial(apply_models_batch,
                          vectorizer_ref=vectorizer_s1_ref,
                          kmeans_ref=kmeans_s1_ref,
                          # kmeans_batch_size=cfg.stage1_inf_kmeans_bs, # Only if using JAX predictor
                          cluster_col_name=CLUSTER_A_COL)

    tagged_ds_A = ds.map_batches(
        map_s1_func,
        batch_format="pandas",
        batch_size=cfg.get("stage1_inf_batch_size", cfg.tfidf_batch_size), # Configurable inference batch size
        # concurrency=cfg.get("stage1_inf_concurrency") # Control parallelism
    )
    
    # tagged_ds_A = tagged_ds_A.materialize()
    print("Stage 1 inference complete. Schema:", tagged_ds_A.schema())
    # print("Sample row after Stage 1:", tagged_ds_A.take(1)) # Debug
    print("--- Stage 1 Done ---")


    # --- Stage 2: Train and Infer ---
    print("--- Stage 2 Starting ---")
    # Train Stage 2 models (one per Stage 1 cluster) using map_groups
    print("Training Stage 2 models (one per Stage 1 cluster)...")
    stage2_group_processor_partial = partial(process_stage2_group, cfg=cfg)
    
    # process_stage2_group returns (cluster_a_id, (vec_ref, kmeans_ref))
    stage2_model_results_ds = tagged_ds_A.groupby(CLUSTER_A_COL).map_groups(
        stage2_group_processor_partial,
        ray_remote_args={ # Resources for each group processing task
             "num_cpus": cfg.get("stage2_train_cpus", 2),
        },
        batch_format="pandas" # Groups are passed as pandas DFs
    )

    # Collect the model references (assume num stage 1 clusters is manageable)
    print("Collecting Stage 2 model references...")
    stage2_model_results = stage2_model_results_ds.take_all() # List of dicts
    
    # Build dictionary mapping cluster_A ID to model refs
    # Results look like: [{'cluster_A': 0, 'map_groups_output': (v0_ref, k0_ref)}, ...]
    stage2_models_dict = {
        item[CLUSTER_A_COL]: item['map_groups_output']
        for item in stage2_model_results
    }
    stage2_models_dict_ref = ray.put(stage2_models_dict) # Put the whole dict in object store
    print(f"Stage 2 models references collected for {len(stage2_models_dict)} clusters.")
    # print("Stage 2 Model Dict Sample:", dict(list(stage2_models_dict.items())[:2])) # Debug
    
    print("Running Stage 2 inference...")
    # Define the batch mapping function for Stage 2
    def apply_stage2_batch(batch: pd.DataFrame, models_dict_ref) -> pd.DataFrame:
        models_dict = ray.get(models_dict_ref) # Get dict {cluster_id: (vec_ref, kmeans_ref)}
        batch[CLUSTER_B_COL] = -1 # Initialize column
        
        # Process each cluster_A group within the batch
        for cluster_a_id, group in batch.groupby(CLUSTER_A_COL):
            if cluster_a_id in models_dict:
                vec_ref, kmeans_ref = models_dict[cluster_a_id]
                if vec_ref is not None and kmeans_ref is not None:
                    # Apply the specific models for this group
                    processed_group = apply_models_batch(
                        group.copy(), # Pass copy to avoid modifying original slice
                        vectorizer_ref=vec_ref,
                        kmeans_ref=kmeans_ref,
                        # kmeans_batch_size=cfg.stage2_inf_kmeans_bs, # If needed
                        cluster_col_name=CLUSTER_B_COL # Predict into the target col
                    )
                    # Assign results back using index
                    batch.loc[group.index, CLUSTER_B_COL] = processed_group[CLUSTER_B_COL]
                else:
                     # Models failed training for this cluster_A
                     batch.loc[group.index, CLUSTER_B_COL] = -1 # Mark as failed/skipped
            else:
                # Should not happen if map_groups covered all clusters present
                print(f"Warning: cluster_A ID {cluster_a_id} not found in Stage 2 models dict.")
                batch.loc[group.index, CLUSTER_B_COL] = -4 # Mark as missing model error
        return batch

    tagged_ds_B = tagged_ds_A.map_batches(
        partial(apply_stage2_batch, models_dict_ref=stage2_models_dict_ref),
        batch_format="pandas",
        batch_size=cfg.get("stage2_inf_batch_size", cfg.tfidf_batch_size)
        # concurrency=cfg.get("stage2_inf_concurrency")
    )
    
    final_ds = tagged_ds_B.materialize()
    print("Stage 2 inference complete. Schema:", final_ds.schema())
    # print("Sample row after Stage 2:", tagged_ds_B.take(1)) # Debug
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

def tfidf_minhash_ray(spark, df, column, num_perm, ngram_size, min_ngram_size, threshold):

    dummy_config = {
        "base_dir": "/tmp/ray_clustering_output",
        "cluster_layout": [5, 3, 2], # Smaller example layout
        "max_docs": 5000, # Sample size for training
        "stage1_train_kmeans_bs": 1024,
        "stage1_inf_kmeans_bs": 4096, # Needed if using JAX prediction
        "stage1_inf_batch_size": 1000, # Ray batch size for inference
        "stage1_train_cpus": 70, # Resources for Stage 1 training task
        "stage2_train_kmeans_bs": 512,
        "stage2_inf_kmeans_bs": 2048, # Needed if using JAX prediction
        "stage2_inf_batch_size": 1000,
        "stage2_train_cpus": 2, # Resources per Stage 2 group task
        "stage3_train_kmeans_bs": 256,
        "stage3_inf_kmeans_bs": 1024, # Needed if using JAX prediction
        "stage3_proc_cpus": 2, # Resources per Stage 3 group task
        "stage3_min_group_size": 50, # Min size for Stage 3 processing
        "tfidf_batch_size": 500, # Default batch size if others not set
        "stage3_dedup": True,
        "similarity": 0.85,
        "num_perm": 128,
        "ngram_size": 5,
        "train1_ds_kwargs_load": { # Example using a common dataset
            "path": "ag_news",
            "split": "train" 
        },
        "ray_max_docs_limit": 10000 # Limit total docs processed (for testing)
    }
    cfg = config_dict.ConfigDict(dummy_config)

    # --- Run Pipeline ---
    start_time = time.time()
    df = ray.data.from_spark(df, parallelism=100)
    run_clustering_pipeline(df, cfg)
    end_time = time.time()
    print(f"Total pipeline execution time: {end_time - start_time:.2f} seconds")
    



    
    
    