# /database_project/src/ray_tfidf_vec.py
import os
import time
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
import os
import numpy as np
import pandas as pd # Used for type hints, not core logic
import logging
import math
import time
from functools import reduce
from typing import Dict, Any, Iterator, List, Tuple, Optional, Set
# Import scikit-learn components
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

import socket
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from tqdm import tqdm

tfidf_logger = logging.getLogger('sklearn.feature_extraction.text')
import warnings
warnings.filterwarnings('ignore', message="Your stop_words may be inconsistent with your preprocessing.*", category=UserWarning)
import torch
import ray



def number_normalizer(tokens):
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))



def compile_nearest_cluster(kmeans, kmeans_batch_size):
    from flax.jax_utils import pad_shard_unpad
    import jax
    import jax.numpy as jnp
    def _nearest_cluster(data, clusters):
        data = jnp.expand_dims(data, axis=1)
        clusters = jnp.expand_dims(clusters, axis=0)
        dis = (data - clusters) ** 2.0
        dis = jnp.sum(dis, axis=-1)
        dis = jnp.squeeze(dis)
        return dis.argmin(axis=1)
    n_local_devices = jax.local_device_count()
    codebook = np.array(kmeans.cluster_centers)
    codebook = jax.device_put(codebook)
    nearest_cluster_p = jax.pmap(_nearest_cluster, in_axes=(0, None))
    def nearest_cluster_bound(element):
        return nearest_cluster_p(element, codebook)

    nearest_cluster_padded = pad_shard_unpad(nearest_cluster_bound,
                                             static_return=False,static_argnums=())
    def nearest_cluster(batch):
        batch_preds = nearest_cluster_padded(batch,
                                                        min_device_batch=kmeans_batch_size//n_local_devices)
        batch_preds = jax.device_get(batch_preds).reshape(-1).tolist()
        return batch_preds

    return nearest_cluster




import torch




def torch_pairwise_distance(data1, data2):
    A = torch.unsqueeze(data1, dim=1)
    B = torch.unsqueeze(data2, dim=0)
    dis = (A - B) ** 2.0
    dis = torch.sum(dis, dim=-1)
    dis = torch.squeeze(dis)
    return torch.argmin(dis, dim=1)





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
            iter_k=None,
            jax_pairwise_distance=None
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


def create_jax_pairwise_distance():
    import jax
    import jax.numpy as jnp
    def reshape_for_jax(data1, data2):

        batch_size = data1.shape[0]
        n_clusters = data2.shape[0]
        data1 = data1.reshape([jax.local_device_count(),
                            batch_size//jax.local_device_count(),-1])
        return data1, data2, batch_size, n_clusters

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

    dist_func = jax.pmap(_jax_pairwise_distance,in_axes=(0, None))
    def jax_pairwise_distance(data1, data2):


        data1, data2, *shape = reshape_for_jax(data1, data2)
        dis = dist_func(data1, data2)
        dis = jax.device_get(dis).reshape(shape)
        return dis
    return jax_pairwise_distance

def fit_kmeans(embeddings, kmeans_cfg, **kwargs):

    jax_pairwise_distance = create_jax_pairwise_distance()

    embeddings = DataLoader(embeddings, batch_size=kmeans_cfg.train.batch_size, drop_last=True)

    kmeans = KMeans(n_clusters=kmeans_cfg.n_clusters, balanced=True, **kwargs)
    with tqdm(dynamic_ncols=True,desc="fit_kmeans") as pbar:
        for i,batch in enumerate(embeddings):
            pbar.update(batch.shape[0])
            kmeans.fit(batch, iter_limit=kmeans_cfg.train.iter_limit, online=True, iter_k=i, jax_pairwise_distance=jax_pairwise_distance)
    return kmeans



def _fit_models_remote(
    cfg: object,
    ds: ray.data.Dataset, # Changed type hint
) -> Tuple[object, object, float]: # Return train_time
    start_time = time.time()
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
    train_time = time.time() - start_time
    print(f"[{stage_label}] Model training took {train_time:.2f} seconds.")
    return vectorizer, kmeans, train_time

@ray.remote(num_returns=3) # Expecting three return values now
def fit_models_remote(cfg, ds):
    return _fit_models_remote(cfg, ds)


class TFIDFInferenceModel:
    def __init__(self, vectorizer_ref: ray.ObjectRef):
        self.vectorizer = ray.get(vectorizer_ref)

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
        self.kmeans = ray.get(kmeans_ref)
        self.tagging_func = compile_nearest_cluster(self.kmeans, kmeans_batch_size=cfg.kmeans.inference.batch_size)
        self.cluster_col_name = cfg.cluster_col_name

    def __call__(self, batch: pd.DataFrame):
        embeddings = np.array([emb for emb in batch["embeddings"]])
        batch[self.cluster_col_name] = np.array(self.tagging_func(embeddings), dtype=np.int32)
        batch.drop(columns=["embeddings"], inplace=True)
        return batch


os.makedirs("/mnt/gcs_bucket/ray_clustering_output/ray_output_final_clustered", exist_ok=True)


def fit_predict(ds: ray.data.Dataset, cfg: object) -> Tuple[ray.data.Dataset, float, float]:
    """Fits models and predicts clusters, returning dataset and timings."""
    print(f"--- {cfg.pretty_name} Fitting Models ---")
    fit_start_time = time.time()
    vectorizer_ref, kmeans_ref, train_time_ref = fit_models_remote.options(
            num_cpus=cfg.tfidf.train.num_cpus,
            resources={"TPU-v4-8-head": 1},
    ).remote(
            cfg, ds
    )
    # Wait for training to finish and get the actual time
    train_time = ray.get(train_time_ref)
    fit_end_time = time.time()
    print(f"Models fitted in {fit_end_time - fit_start_time:.2f}s (Reported Train Time: {train_time:.2f}s).")

    print(f"--- {cfg.pretty_name} Starting Inference ---")
    inference_start_time = time.time()

    # Combine models ref for inference actors
    # models_ref = (vectorizer_ref, kmeans_ref, train_time_ref) # Pass tuple

    emb_tagged_ds_A = ds.map_batches(
        TFIDFInferenceModel,
        batch_format="pandas",
        batch_size=cfg.tfidf.inference.batch_size,
        num_cpus=cfg.tfidf.inference.num_cpus,
        concurrency=cfg.tfidf.inference.concurrency,
        fn_constructor_kwargs={"vectorizer_ref": vectorizer_ref}, # Pass tuple
    )

    tagged_ds_A = emb_tagged_ds_A.map_batches(
        KMeansInferenceModel,
        batch_format="pandas",
        batch_size=cfg.kmeans.inference.batch_size,
        resources={"TPU-v4-8-head": 1},
        num_cpus=cfg.kmeans.inference.num_cpus,
        concurrency=cfg.kmeans.inference.concurrency,
        fn_constructor_kwargs={"kmeans_ref": kmeans_ref, # Pass tuple
                               "cfg": cfg},
    ).materialize() # Materialize after inference

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time
    print(f"Inference completed in {inference_time:.2f} seconds.")

    return tagged_ds_A, train_time, inference_time

def stage1(ds: ray.data.Dataset, cfg: object) -> Tuple[ray.data.Dataset, float, float, float]:
    """Runs stage 1 clustering and returns dataset and timings."""
    start_time = time.time()
    tagged_ds_A, train_time, inference_time = fit_predict(ds, cfg)
    end_time = time.time()
    print(f"{cfg.pretty_name} complete. Total Time: {end_time - start_time:.2f}s (Train: {train_time:.2f}s, Infer: {inference_time:.2f}s)")
    # Stage 1 doesn't have a separate "stage2_time", so return 0
    metrics = {"inference_time": inference_time, "train_time": train_time, "total_time": end_time - start_time, "stage": cfg.name}
    return tagged_ds_A, metrics

from ray_minhash import dedup
@ray.remote(num_returns=2)
def dedup_remote(ds: ray.data.Dataset, cfg: object):
    deduplicated_dataset, duplicate_count = dedup(ds, cfg)
    return deduplicated_dataset.materialize(), duplicate_count

@ray.remote(num_returns=3) # Now returns dataset, train_time, inference_time
def fit_predict_remote(ds: ray.data.Dataset, cfg):
    tagged_ds, train_time, inference_time = fit_predict(ds.materialize(), cfg)
    return tagged_ds.materialize(), train_time, inference_time

def stage2(ds: ray.data.Dataset, cfg: object) -> Tuple[ray.data.Dataset, float, float, float, int, str]:
    """Runs stage 2 clustering and optional deduplication, returns dataset, timings, dupe count, and distribution."""
    stage2_start_time = time.time()
    stage1_clusters = cfg.cluster_spec[0]
    stage1_cluster_col_name = cfg.partition_cols[0]

    og_ds = ds
    stage1_datasets = [og_ds.filter(expr=f"{stage1_cluster_col_name} == {cluster_id}")
                       for cluster_id in range(stage1_clusters)]

    processed_refs = [] # List of tuples: (final_ds_ref, train_time_ref, infer_time_ref, dupe_count_ref)
    for ds_cluster_data in stage1_datasets:
        # Stage 2 clustering always happens
        s2_clustered_ds_ref, s2_train_time_ref, s2_infer_time_ref = fit_predict_remote.remote(ds_cluster_data, cfg)

        # Conditional deduplication
        if cfg.should_dedup:
            # Call dedup_remote which now returns two refs
            final_ds_ref, dupe_count_ref = dedup_remote.remote(s2_clustered_ds_ref, cfg)
            processed_refs.append((final_ds_ref, s2_train_time_ref, s2_infer_time_ref, dupe_count_ref))
        else:
            # If not deduping, store the clustered dataset ref and a 0 placeholder for count ref
            processed_refs.append((s2_clustered_ds_ref, s2_train_time_ref, s2_infer_time_ref, ray.put(0)))

        time.sleep(20) # Consider if this sleep is still necessary/optimal - Removed for now

    # Retrieve all results
    dataset_results = ray.get([ref_pair[0] for ref_pair in processed_refs]) # Get datasets
    train_time_results = ray.get([ref_pair[1] for ref_pair in processed_refs]) # Get train times
    infer_time_results = ray.get([ref_pair[2] for ref_pair in processed_refs]) # Get inference times
    count_results = ray.get([ref_pair[3] for ref_pair in processed_refs]) # Get counts

    # Aggregate results
    ds_list = dataset_results # List of datasets
    total_cluster_duplicates = sum(count_results) # Sum the counts
    total_train_time = np.mean(train_time_results) # Sum train times (might be misleading if parallel)
    total_inference_time = np.mean(infer_time_results) # Sum inference times (might be misleading if parallel)

    # Union datasets (ensure ds_list is not empty)
    if not ds_list:
        raise ValueError("No datasets returned from stage 2 processing.")

    final_ds = ds_list[0]
    if len(ds_list) > 1:
        final_ds = final_ds.union(*ds_list[1:])

    final_ds = final_ds.sort(cfg.partition_cols[:2]).materialize() # Materialize after union and sort


    stage2_end_time = time.time()
    stage2_time = stage2_end_time - stage2_start_time
    print(f"{cfg.pretty_name} complete. Total Time: {stage2_time:.2f}s (Agg Train: {total_train_time:.2f}s, Agg Infer: {total_inference_time:.2f}s)")
    print(f"Stage 2 Duplicates Found (if enabled): {total_cluster_duplicates}")
    
    metrics = {"inference_time": total_inference_time, "train_time": total_train_time, 
               "total_time": stage2_time, "stage": cfg.name,
               }
    if cfg.should_dedup:
        metrics["n_duplicates"] = total_cluster_duplicates

    return final_ds, metrics




import glob

from ml_collections import config_dict
import yaml



def read_config(path):
    with open(path) as f:
        config_data = yaml.safe_load(f)
        cfg = config_dict.ConfigDict(config_data)
    return cfg


def fake_stage1(ds, cfg):
    import numpy as np
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

    # Fake timings for compatibility
    fake_train_time = 1.0
    fake_inference_time = 1.0
    fake_stage2_time = 0.0

    return ds, fake_train_time, fake_inference_time, fake_stage2_time



def run_cl_step_for_workflow(ds, cfg: object) -> Tuple[ray.data.Dataset, int, float, float, float, str]:
    """
    Runs the full multi-stage clustering workflow.

    Returns:
        Tuple containing:
        - final_ds: The final clustered (and potentially deduplicated) Ray Dataset.
        - workflow_duplicate_count: Total duplicates found (relevant for CL->ND).
        - total_train_time: Aggregated training time across stages.
        - total_inference_time: Aggregated inference time across stages.
        - total_stage2_time: Time specifically spent in the stage2 logic.
    """
    output_base_path = cfg.args.output # Use the output path from args
    os.makedirs(output_base_path, exist_ok=True)
    print(f"Clustering output will be directed towards: {output_base_path}") # Verify path

    ds = ds.repartition(cfg.num_blocks) # Use num_blocks from config
    

    partition_cols = [x["cluster_col_name"] for x in cfg.stages_list]
    cluster_spec = [x["kmeans"]["n_clusters"] for x in cfg.stages_list]

    limit = cfg.get("ray_max_docs_limit", None)
    if limit:
         ds = ds.limit(limit)
         print(f"Dataset limited to {limit} documents.")


    base_cfg = config_dict.ConfigDict(cfg.base_stage)
    base_cfg.cluster_spec = cluster_spec
    base_cfg.partition_cols = partition_cols
    base_cfg.args = cfg.args # Pass args down


    workflow_duplicate_count = 0 # Initialize count
    # --- Execute Stages ---
    stage_functions = [stage1, stage2] # Define stage functions
    metric_list = []
    for i, stage_config_data in enumerate(cfg.stages_list):
        func = stage_functions[i]
        stage_cfg = base_cfg.copy_and_resolve_references()
        stage_cfg.update(stage_config_data)
        stage_cfg.should_dedup = cfg.base_stage.should_dedup # Ensure dedup flag propagates

        print(f"\n--- Running Stage {i+1} ({stage_cfg.name}) ---")
        print(stage_cfg)
        ds, metrics = func(ds, stage_cfg)
        if "n_duplicates" in metrics:
            workflow_duplicate_count = metrics["n_duplicates"]
        metric_list.append(metrics)



    final_ds = ds.materialize()



    return final_ds, workflow_duplicate_count, metric_list