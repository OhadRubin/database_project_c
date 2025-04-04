import os
import time
# os.system("sudo kill -9 $(sudo lsof -w /dev/accel0 | awk 'NR>1{print $2}' |uniq)")
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
from sklearn.cluster import KMeans as SklearnKMeans, k_means
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


def get_sklearn_feature_pipeline(tfidf_cfg):
    n_components, random_seed = tfidf_cfg.train.n_components, tfidf_cfg.train.random_seed
    stop_words = list(ENGLISH_STOP_WORDS.union(["#NUMBER"]))
    vectorizer = Pipeline([('tfidf', NumberNormalizingVectorizer(stop_words=stop_words)),
                            ('svd', TruncatedSVD(n_components=n_components,random_state=random_seed)),
                            ('normalizer', Normalizer(copy=False))],
                            verbose=True)
    return vectorizer


import ray.cloudpickle as cloudpickle
from torch.utils.data import DataLoader

def serialize_objectref_dict(objectref_dict):
    return {k: cloudpickle.dumps(v) for k, v in objectref_dict.items()}

def deserialize_objectref_dict(objectref_dict):
    return {k: cloudpickle.loads(v) for k, v in objectref_dict.items()}

def fit_kmeans(embeddings, kmeans_cfg, **kwargs):    
    embeddings = DataLoader(embeddings, batch_size=kmeans_cfg.train.batch_size, drop_last=True)
    
    kmeans = KMeans(n_clusters=kmeans_cfg.n_clusters, balanced=True, **kwargs)
    with tqdm(dynamic_ncols=True,desc="fit_kmeans") as pbar:
        for i,batch in enumerate(embeddings):
            pbar.update(batch.shape[0])
            kmeans.fit(batch, iter_limit=kmeans_cfg.train.iter_limit, online=True, iter_k=i)
    return kmeans


@ray.remote
def fit_models_remote(
    cfg: object,
    ds: pd.DataFrame,
) -> Tuple[object, object]:
    sample_ds = ds.limit(cfg.max_docs)
    print(f"Collecting sample...")
    sample_df = sample_ds.to_pandas()
    print(f"Sample size: {len(sample_df)}")
    stage_label = cfg.cluster_col_name
    texts = sample_df["text"].tolist()
    print(f"[{stage_label}] Fitting vectorizer on {len(texts)} samples...")
    vectorizer = get_sklearn_feature_pipeline(cfg.tfidf)
    embeddings = vectorizer.fit_transform(texts)
    print(f"[{stage_label}] Vectorizer fitting done. Embedding shape: {embeddings.shape}")
    print(f"[{stage_label}] Fitting K-means with {cfg.kmeans.n_clusters} clusters...")
    kmeans = fit_kmeans(embeddings,  cfg.kmeans)
    print(f"[{stage_label}] K-means fitting done.")
    return vectorizer, kmeans




class TFIDFInferenceModel:
    def __init__(self, vectorizer_ref: ray.ObjectRef):
        vectorizer, _ =ray.get(vectorizer_ref)
        self.vectorizer = vectorizer
        
    def __call__(self, batch: pd.DataFrame):
        texts = batch["text"].tolist()
        embeddings = self.vectorizer.transform(texts)
        batch["embeddings"] = list(embeddings)
        return batch
    


class KMeansInferenceModel:
    def __init__(self,
        kmeans_ref: ray.ObjectRef,
        cfg: str
        ):
        _, kmeans = ray.get(kmeans_ref)
        self.kmeans = kmeans
        self.tagging_func = compile_nearest_cluster(self.kmeans, kmeans_batch_size=cfg.kmeans.inference.batch_size)
        self.cluster_col_name = cfg.cluster_col_name

    def __call__(self, batch: pd.DataFrame):
        embeddings = np.array([emb for emb in batch["embeddings"]])
        batch[self.cluster_col_name] = np.array(self.tagging_func(embeddings), dtype=np.int32)
        batch.drop(columns=["embeddings"], inplace=True)
        return batch


os.makedirs("/mnt/gcs_bucket/ray_clustering_output/ray_output_final_clustered", exist_ok=True)


def fit_predict(ds: ray.data.Dataset, cfg: object):
    print(f"--- {cfg.pretty_name} Starting ---")
    



    
    # ray.remote
    models_s1_ref = fit_models_remote.options(
            num_cpus=cfg.tfidf.train.num_cpus,
            resources={"TPU-v4-8-head": 1},
    ).remote(
            cfg, ds
    )
    
    # vectorizer_s1, kmeans_s1 = ray.get(models_s1_ref)
    
    # vectorizer_s1_ref = ray.put(vectorizer_s1)
    # kmeans_s1_ref = ray.put(kmeans_s1)

    emb_tagged_ds_A = ds.map_batches(
        TFIDFInferenceModel,
        batch_format="pandas",
        batch_size=cfg.tfidf.inference.batch_size,
        num_cpus=cfg.tfidf.inference.num_cpus,
        concurrency=cfg.tfidf.inference.concurrency,
        fn_constructor_kwargs={"vectorizer_ref": models_s1_ref},
    )
    tagged_ds_A = emb_tagged_ds_A.map_batches(
        KMeansInferenceModel,
        batch_format="pandas",
        batch_size=cfg.kmeans.inference.batch_size,
        resources={"TPU-v4-8-head": 1},
        num_cpus=cfg.kmeans.inference.num_cpus,
        concurrency=cfg.kmeans.inference.concurrency,
        fn_constructor_kwargs={"kmeans_ref": models_s1_ref,
                               "cfg": cfg},
    )
    

    return tagged_ds_A

def stage1(ds: ray.data.Dataset, cfg: object):
    start_time = time.time()
    tagged_ds_A = fit_predict(ds, cfg).materialize()

    end_time = time.time()
    print(f"{cfg.pretty_name} complete. Time taken: {end_time - start_time:.2f} seconds")
    print(f"Schema:", tagged_ds_A.schema())
    print(f"Sample row after {cfg.pretty_name}:", tagged_ds_A.take(1))
    return tagged_ds_A
    
    


def apply_models_batch(
    batch: pd.DataFrame,
    vectorizer_ref: ray.ObjectRef,
    kmeans_ref: ray.ObjectRef,
    cluster_col_name: str
) -> pd.DataFrame:
    """Applies vectorizer (transform) and kmeans (predict) to a batch."""
    if batch.empty:
        return batch
    vectorizer = ray.get(vectorizer_ref) # Retrieve models from Object Store (Ray caches locally after first get)
    kmeans = ray.get(kmeans_ref)
    
    
    tagging_func = compile_nearest_cluster(kmeans, kmeans_batch_size=8192)
    texts = batch["text"].tolist()
    # 1. Vectorize (Transform)
    embeddings = vectorizer.transform(texts)
    # 2. Predict Cluster
    batch[cluster_col_name] = tagging_func(embeddings)


    return batch


def fit_stage2(
    group_df: pd.DataFrame,
    cfg: object,
) -> Tuple[int, ray.ObjectRef]:
    if group_df.empty:
        assert False
    max_docs_sample = cfg.max_docs
    

    cluster_a_id = group_df[cfg.partition_cols[0]].iloc[0]
    stage_label = f"Stage2_A={cluster_a_id}"
    sample_df = group_df.sample(n=min(len(group_df), max_docs_sample), random_state=42)

    # Use the generic fitting task
    models_ref = fit_models_remote.remote(
        cfg=cfg,
        sample_data=sample_df,
    )
    print(f"[{stage_label}] Model fitting tasks submitted.")
    result =  {"cluster_a_id":cluster_a_id, "models_ref": models_ref}
    result = serialize_objectref_dict(result)
    return pd.DataFrame([result])




def _apply_stage2_batch(batch: pd.DataFrame, stage2_models_dict_ref:object, cfg: object) -> pd.DataFrame:
    models_dict = ray.get(stage2_models_dict_ref)
    cluster_col_name = cfg.partition_cols[1]
    batch[cluster_col_name] = -1 # Initialize column
    
    # Process each cluster_A group within the batch
    for cluster_a_id, group in batch.groupby(cfg.partition_cols[0]):
        models_ref = models_dict.get(cluster_a_id)
        vectorizer, kmeans = ray.get(models_ref)
        
        # Skip if any model is None
        if vectorizer is None or kmeans is None:
            print(f"Warning: Missing model components for cluster_A={cluster_a_id}")
            continue
            
        processed_group = apply_models_batch(
            group.copy(), # Pass copy to avoid modifying original slice
            vectorizer_ref=ray.put(vectorizer),
            kmeans_ref=ray.put(kmeans),
            cluster_col_name=cluster_col_name
        )
        batch.loc[group.index, cluster_col_name] = processed_group[cluster_col_name]


    return batch

def stage2(tagged_ds_A: ray.data.Dataset, cfg: object):
    print("\n--- Stage 2 Starting ---\nTraining Stage 2 models (one per Stage 1 cluster)...")
    
    def _fit_stage2(group_df: pd.DataFrame):
        return fit_stage2(group_df, cfg=cfg)
    

    stage2_model_results_ds = tagged_ds_A.groupby(cfg.partition_cols[0]).map_groups(
        _fit_stage2,
        num_cpus=cfg.tfidf.train.num_cpus,
        batch_format="pandas",
        resources={"TPU-v4-8-head": 1},
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
    
    print("Running Stage 2 inference...")
    def apply_stage2_batch(batch: pd.DataFrame):
        return _apply_stage2_batch(batch, stage2_models_dict_ref, cfg)
    
    tagged_ds_B = tagged_ds_A.map_batches(
        apply_stage2_batch,
        batch_format="pandas",
        batch_size=cfg.stage2_inf_batch_size,
        resources={"TPU-v4-8-head": 1},
        concurrency=10,
        num_cpus=210,
    )
    
    final_ds = tagged_ds_B.sort(cfg.partition_cols[:2])
    print("Stage 2 inference complete. Schema:", final_ds.schema())
    print("Sample row after Stage 2:", final_ds.take(1)) # Debug
    print("--- Stage 2 Done ---")
    return final_ds

@ray.remote
def fit_predict_remote(ds: ray.data.Dataset, cfg: object):
    return fit_predict(ds.materialize(), cfg)
    
    
    
from ray.util.queue import Queue, Empty



def new_stage2(ds: ray.data.Dataset, cfg: object):
    stage1_clusters = cfg.cluster_spec[0]
    stage1_cluster_col_name = cfg.partition_cols[0]
    
    og_ds = ds
    ds_ref_list = []
    for cluster_id in range(stage1_clusters):
        ds = og_ds.filter(expr=f"{stage1_cluster_col_name} == {cluster_id}")
        new_ds = fit_predict_remote.remote(ds, cfg)
        new_ds = ray.get(new_ds).materialize()
        ds_ref_list.append(new_ds)
        
    # ds_list = ray.get(ds_ref_list)
    ds_list = ds_ref_list
    final_ds = ds_list[0]
    final_ds = final_ds.union(*ds_list[1:])



    return final_ds.materialize()

from ml_collections import config_dict
import glob
import yaml



def read_config(path):
    with open(path) as f:
        config_data = yaml.safe_load(f)
        cfg = config_dict.ConfigDict(config_data)
    return cfg


def fake_stage1(ds, cfg):
    # Assign random cluster IDs to each element
    import numpy as np
    
    # Get the number of clusters from the configuration
    n_clusters = cfg.kmeans.n_clusters
    
    # Create a copy of the dataset with random cluster assignments
    def assign_random_cluster(batch):
        batch_size = len(batch)
        batch[cfg.cluster_col_name] = np.random.randint(0, n_clusters, size=batch_size)
        return batch
    
    # Apply the random assignment to each batch
    ds = ds.map_batches(
        assign_random_cluster,
        batch_format="pandas",
        batch_size=2048
    ).materialize()
    
    return ds
    


def run_clustering_pipeline(ds, cfg: object):
    output_base_path = f"{cfg.base_dir}/ray_output_final_clustered" 
    os.makedirs(output_base_path, exist_ok=True)
    

        
    partition_cols = [x["cluster_col_name"] for x in cfg.stages_list]
    cluster_spec = [x["kmeans"]["n_clusters"] for x in cfg.stages_list]
    
    limit = cfg.get("ray_max_docs_limit", None)
    if limit:
         ds = ds.limit(limit)
         print(f"Dataset limited to {limit} documents.")
    
    
    base_cfg = config_dict.ConfigDict(cfg.base_stage)
    base_cfg.cluster_spec = cluster_spec
    base_cfg.partition_cols = partition_cols
    for stage, func in zip(cfg.stages_list,[
        # stage1, 
        fake_stage1,
        new_stage2
        ]):
        stage_cfg = base_cfg.copy_and_resolve_references()
        stage_cfg.update(stage)
        print(stage_cfg)
        ds = func(ds, stage_cfg)
    
    final_ds = ds.materialize()
    
    print(f"Final dataset successfully written to {output_base_path}")
    final_ds.write_parquet(
        output_base_path,
        partition_cols=partition_cols,
    )
    print("--- Pipeline Finished ---")

    



def tfidf_minhash_ray(args):
    cfg = read_config("database_project/src/configs/base.yml")
    cfg.args = args

    start_time = time.time()
    

    if args.limit_files is not None:
        input_file = glob.glob(args.input_file)[:args.limit_files]    
    ray_df = ray.data.read_json(input_file, override_num_blocks=cfg.num_blocks)
    
    
    print(f"Ray Dataset created with {ray_df.count()} rows")
    
    # Run the clustering pipeline
    run_clustering_pipeline(ray_df, cfg)
    
    end_time = time.time()
    print(f"Total pipeline execution time: {end_time - start_time:.2f} seconds")
    
    return f"{cfg.base_dir}/ray_output_final_clustered"
    



    
    
    