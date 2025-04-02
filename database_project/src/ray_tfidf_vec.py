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
    

import torch


from flax.jax_utils import pad_shard_unpad    

def _nearest_cluster(data, clusters):
    data = jnp.expand_dims(data, axis=1)
    clusters = jnp.expand_dims(clusters, axis=0)
    dis = (data - clusters) ** 2.0
    dis = jnp.sum(dis, axis=-1)
    dis = jnp.squeeze(dis)
    return dis.argmin(axis=1)

def compile_nearest_cluster(kmeans, kmeans_batch_size):
    n_local_devices = jax.local_device_count()
    codebook = np.array(kmeans.cluster_centers)
    codebook = jax.device_put(codebook)
    
    def nearest_cluster_bound(element):
        return jax.pmap(_nearest_cluster,in_axes=(0, None))(element, codebook)
    
    nearest_cluster_padded = pad_shard_unpad(nearest_cluster_bound,
                                             static_return=False,static_argnums=())
    def nearest_cluster(batch):
        if isinstance(batch, torch.Tensor):
            batch = batch.numpy()
        else:
            batch = np.array(batch)
            
        batch_preds = nearest_cluster_padded(batch,
                                                        min_device_batch=kmeans_batch_size//n_local_devices)
        batch_preds = jax.device_get(batch_preds).reshape(-1).tolist()
        return batch_preds
    
    return nearest_cluster



def _jax_pairwise_distance(data1, data2):
    # Expand the data matrices to have an extra dimension.
    A = jnp.expand_dims(data1, axis=1)
    B = jnp.expand_dims(data2, axis=0)

    # Compute the squared pairwise distances.
    dis = (A - B) ** 2.0

    # Sum the squared pairwise distances over the feature dimension.
    dis = jnp.sum(dis, axis=-1)

    # Squeeze the output to have shape (N, M)[].
    dis = jnp.squeeze(dis)

    return dis

import torch

def reshape_for_jax(data1, data2):
    batch_size = data1.shape[0]
    n_clusters = data2.shape[0]
    data1 = data1.reshape([jax.local_device_count(),
                           batch_size//jax.local_device_count(),-1])
    return data1, data2, batch_size, n_clusters


def torch_pairwise_distance(data1, data2):
    A = torch.unsqueeze(data1, dim=1)
    B = torch.unsqueeze(data2, dim=0)
    dis = (A - B) ** 2.0
    dis = torch.sum(dis, dim=-1)
    dis = torch.squeeze(dis)
    return torch.argmin(dis, dim=1)


def jax_pairwise_distance(data1, data2):
    dist_func = jax.pmap(_jax_pairwise_distance,in_axes=(0, None))
    
    data1, data2, *shape = reshape_for_jax(data1, data2)
    dis = dist_func(data1, data2)
    dis = jax.device_get(dis).reshape(shape)
    return dis


def np_pairwise_distance(data1, data2):
    A = np.expand_dims(data1, axis=1)
    B = np.expand_dims(data2, axis=0)
    dis = (A - B) ** 2.0
    dis = np.sum(dis, axis=-1)
    dis = np.squeeze(dis)
    return dis.argmin(axis=1)

def auction_lap(job_and_worker_to_score, return_token_to_worker=True):
    """
    Solving the balanced linear assignment problem with auction algorithm.
    Arguments:
        - job_and_worker_to_score -> N x M euclidean distances between N data points and M cluster centers
    Returns:
        - assignment -> balanced assignment between jobs and workers
    """
    eps = (job_and_worker_to_score.max() - job_and_worker_to_score.min()) / 50
    eps.clamp_min_(1e-04)
    assert not torch.isnan(job_and_worker_to_score).any()
    if torch.isnan(job_and_worker_to_score).any():
        raise Exception("NaN distance")
    worker_and_job_to_score = job_and_worker_to_score.detach().transpose(0,1).contiguous()
    num_workers, num_jobs = worker_and_job_to_score.size()
    jobs_per_worker = num_jobs // num_workers
    value = torch.clone(worker_and_job_to_score)
    bids = torch.zeros((num_workers, num_jobs),
                        dtype=worker_and_job_to_score.dtype,
                        device=worker_and_job_to_score.device,
                        requires_grad=False)
    counter = 0
    index = None
    cost = torch.zeros((1,num_jobs,),
                        dtype=worker_and_job_to_score.dtype,
                        device=worker_and_job_to_score.device,
                        requires_grad=False)
    while True:
        top_values, top_index = value.topk(jobs_per_worker + 1, dim=1)
        # Each worker bids the difference in value between that job and the k+1th job
        bid_increments = top_values[:,:-1] - top_values[:,-1:]  + eps
        assert bid_increments.size() == (num_workers, jobs_per_worker)
        bids.zero_()
        bids.scatter_(dim=1, index=top_index[:,:-1], src=bid_increments)

        if counter < 100 and index is not None:
            # If we were successful on the last round, put in a minimal bid to retain
            # the job only if noone else bids. After N iterations, keep it anyway.
            bids.view(-1)[index] = eps
            # 
        if counter > 1000:
            bids.view(-1)[jobs_without_bidder] = eps
        # Find jobs that was a top choice for some worker
        jobs_with_bidder = (bids > 0).any(0).nonzero(as_tuple=False).squeeze(1)
        jobs_without_bidder = (bids == 0).all(0).nonzero(as_tuple=False).squeeze(1)

        # Find the highest bidding worker per job
        high_bids, high_bidders = bids[:, jobs_with_bidder].max(dim=0)
        if high_bidders.size(0) == num_jobs:
            # All jobs were bid for
            break
        
        # Make popular items more expensive
        cost[:, jobs_with_bidder] += high_bids
        value = worker_and_job_to_score - cost

        # # Hack to make sure that this item will be in the winning worker's top-k next time
        index = (high_bidders * num_jobs) + jobs_with_bidder
        value.view(-1)[index] = worker_and_job_to_score.view(-1)[index]
        counter += 1
    

    if return_token_to_worker:
        return high_bidders
    _, sorting = torch.sort(high_bidders)
    assignment = jobs_with_bidder[sorting]
    assert len(assignment.unique()) == num_jobs

    return assignment.view(-1)


class KMeans(object):
    def __init__(self, n_clusters=None, cluster_centers=None, device=torch.device('cpu'), balanced=False,use_jax=True):
        self.n_clusters = n_clusters
        self.cluster_centers = cluster_centers
        self.device = device
        self.balanced = balanced
        self.use_jax = use_jax
    
    @classmethod
    def load(cls, path_to_file):
        with open(path_to_file, 'rb') as f:
            saved = pickle.load(f)
        return cls(saved['n_clusters'], saved['cluster_centers'], torch.device('cpu'), saved['balanced'])
    
    def save(self, path_to_file):
        with open(path_to_file, 'wb+') as f :
            pickle.dump(self.__dict__, f)

    def initialize(self, X):
        num_samples = len(X)
        indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        initial_state = X[indices]
        return initial_state
    
    def fit(
            self,
            X,
            tol=1e-3,
            tqdm_flag=True,
            iter_limit=0,
            online=False,
            iter_k=None
    ):
        if tqdm_flag:
            print(f'running k-means on {self.device}..')
        X = X.float()
        X = X.to(self.device)

        # initialize
        if not online or (online and iter_k == 0):  # ToDo: make this less annoyingly weird
            self.cluster_centers = self.initialize(X)
            

        iteration = 0
        if tqdm_flag:
            tqdm_meter = tqdm(desc='[running kmeans]')
        done=False
        while True:
            if self.balanced:
                distance_matrix = jax_pairwise_distance(X.numpy(), self.cluster_centers.numpy())
                distance_matrix = torch.tensor(distance_matrix)
                cluster_assignments = auction_lap(-distance_matrix)
            else:
                if self.use_jax:
                    dis = jax_pairwise_distance(X.numpy(), self.cluster_centers.numpy())
                else:
                    dis = torch_pairwise_distance(X, self.cluster_centers)
                dis = torch.tensor(dis)
                cluster_assignments = torch.argmin(dis, dim=1)
            
            initial_state_pre = self.cluster_centers.clone()
            for index in range(self.n_clusters):
                selected = torch.nonzero(cluster_assignments == index).squeeze().to(self.device)

                selected = torch.index_select(X, 0, selected)

                # https://github.com/subhadarship/kmeans_pytorch/issues/16
                if selected.shape[0] == 0:
                    selected = X[torch.randint(len(X), (1,))]
                
                self.cluster_centers[index] = selected.mean(dim=0)

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((self.cluster_centers - initial_state_pre) ** 2, dim=1)
                ))

            # increment iteration
            iteration = iteration + 1

            # update tqdm meter
            if tqdm_flag:
                tqdm_meter.set_postfix(
                    iteration=f'{iteration}',
                    center_shift=f'{center_shift ** 2:0.6f}',
                    tol=f'{tol:0.6f}'
                )
                tqdm_meter.update()
            if center_shift ** 2 < tol:
                break
            if iter_limit != 0 and iteration >= iter_limit:
                break
        
        return cluster_assignments.cpu()


    def predict(
            self,X, return_distances=False):
        """
        predict using cluster centers
        :param X: (torch.tensor) matrix
        :param cluster_centers: (torch.tensor) cluster centers
        :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
        :param device: (torch.device) device [default: 'cpu']
        :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
        :return: (torch.tensor) cluster ids
        """


        #if X is a numpy array convert it into a torch tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        # convert to float
        X = X.float()
        # transfer to device
        if self.device != torch.device('cpu'):
            X = X.to(self.device)
        distance_matrix = jax_pairwise_distance(X, self.cluster_centers)
        distance_matrix = torch.tensor(distance_matrix)
        cluster_assignments = torch.argmin(distance_matrix, dim=1 if len(distance_matrix.shape) > 1 else 0)
        if len(distance_matrix.shape) == 1:
            cluster_assignments = cluster_assignments.unsqueeze(0)
        if return_distances:
            return cluster_assignments.cpu(), distance_matrix
        else:
            return cluster_assignments.cpu()


def get_sklearn_feature_pipeline(n_components, random_seed):
    stop_words = list(ENGLISH_STOP_WORDS.union(["#NUMBER"]))
    vectorizer = Pipeline([('tfidf', NumberNormalizingVectorizer(stop_words=stop_words)),
                            ('svd', TruncatedSVD(n_components=n_components,random_state=random_seed)),
                            ('normalizer', Normalizer(copy=False))],
                            verbose=True)
    return vectorizer



import ray.cloudpickle as cloudpickle
from torch.utils.data import DataLoader

def fit_kmeans(embeddings, n_clusters, batch_size, **kwargs):
    
    embeddings = DataLoader(embeddings, batch_size=batch_size, drop_last=True)
    
    kmeans = KMeans(n_clusters=n_clusters, balanced=True, **kwargs)
    with tqdm(dynamic_ncols=True,desc="fit_kmeans") as pbar:
        for i,batch in enumerate(embeddings):
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
    
    
    tagging_func = compile_nearest_cluster(kmeans, kmeans_batch_size=2048)

    # if vectorizer is None or kmeans is None:
    #     assert False

    texts = batch["text"].tolist()
    # 1. Vectorize (Transform)
    embeddings = vectorizer.transform(texts)
    # 2. Predict Cluster
    batch[cluster_col_name] = tagging_func(embeddings)


    return batch


def process_stage2_group(
    group_df: pd.DataFrame,
    cfg: object,
) -> Tuple[int, ray.ObjectRef]:
    if group_df.empty:
        assert False
    n_clusters_b = cfg.cluster_layout[1]
    max_docs_sample = cfg.get("stage2_max_docs_sample", cfg.max_docs)

    cluster_a_id = group_df[CLUSTER_A_COL].iloc[0]
    stage_label = f"Stage2_A={cluster_a_id}"
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
    result =  {"cluster_a_id":cluster_a_id, "models_ref": models_ref}
    result = serialize_objectref_dict(result)
    return pd.DataFrame([result])
def serialize_objectref_dict(objectref_dict):
    return {k: cloudpickle.dumps(v) for k, v in objectref_dict.items()}

def deserialize_objectref_dict(objectref_dict):
    return {k: cloudpickle.loads(v) for k, v in objectref_dict.items()}


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

    # Fit Stage 1 models remotely
    print(f"Attempting to allocate {cfg.stage1_train_cpus} CPUs for Stage 1 training task...")
    # Try with requested CPUs first
    models_s1_ref = fit_models_remote.options(
            num_cpus=cfg.stage1_train_cpus
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
    print("Stage 1 inference complete. Schema:", tagged_ds_A.schema(), "\nSample row after Stage 1:", tagged_ds_A.take(1), "\n--- Stage 1 Done ---\n--- Stage 2 Starting ---\nTraining Stage 2 models (one per Stage 1 cluster)...")
    
    

    stage2_model_results_ds = tagged_ds_A.groupby(CLUSTER_A_COL).map_groups(
        lambda group_df: process_stage2_group(group_df, cfg=cfg),
        num_cpus=cfg.stage2_train_cpus,
        batch_format="pandas"
    )

    # Collect the model references (assume num stage 1 clusters is manageable)
    print("Collecting Stage 2 model references...")
    stage2_model_results = stage2_model_results_ds.take_all() # List of dicts
    print("Finished collecting Stage 2 model references...")
    
    
    stage2_model_results = [deserialize_objectref_dict(item) for item in stage2_model_results]
        
    stage2_models_dict = {
        item['cluster_a_id']: item['models_ref']
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
        "stage1_train_cpus": 128, # Reduced from 70 to fit cluster capacity
        "stage2_train_kmeans_bs": 512,
        "stage2_inf_kmeans_bs": 2048, # Needed if using JAX prediction
        "stage2_inf_batch_size": 1000,
        "stage2_train_cpus": 128, # Reduced from 70 to fit cluster capacity
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




    
    
    