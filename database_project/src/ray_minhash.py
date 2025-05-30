# Adapted from https://github.com/modelscope/data-juicer
import ray
import sys
import logging
import time
import sys
import os
import os
import time
from typing import List, Optional, Union
import re
import numpy as np
import pyarrow as pa
import ray
import regex
# from loguru import logger
from pydantic import Field, PositiveInt
from typing_extensions import Annotated

import hashlib
import struct
from collections import defaultdict
from typing import Optional
import scipy.integrate as integrate


# python3.10 -m pip install ray==2.43.0 numpy~=1.0




MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
    """
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from
    datasketch.

    :param threshold: float. The threshold for similarity
    :param num_perm: int. The number of permutations
    :param false_positive_weight: float. The weight of false positive
    :param false_negative_weight: float. The weight of false negative
    :return: Tuple[int, int]. The optimal `b` and `r` parameters. The number of
        bands, and the number of rows per band respectively
    """

    def false_positive_probability(th: float, band: int, rows: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s**float(rows))**float(band)

        a, _ = integrate.quad(proba, 0.0, th)
        return a

    def false_negative_probability(th: float, band: int, rows: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s**float(rows))**float(band))

        a, _ = integrate.quad(proba, th, 1.0)
        return a

    # object: minimize the weighted FP and FN ratio
    min_error = float('inf')
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


BATCH_SIZE = 1000


@ray.remote
class IdGenerator:

    def __init__(self, start_id=0):
        self.next_id = start_id

    @ray.method(num_returns=2)
    def get_next_id(self, count):
        current_id = self.next_id
        self.next_id += count
        return (current_id, self.next_id)


@ray.remote(scheduling_strategy='SPREAD')
class EdgeBuffer:

    def __init__(self):
        self.edge_dict = {}

    def clear(self):
        self.edge_dict = {}

    def set_edges(self, edge_dict):
        self.edge_dict = edge_dict

    def get_edges(self, key):
        return self.edge_dict.pop(key, [])


@ray.remote(scheduling_strategy='SPREAD')
class BTSUnionFind:
    """
    A distributed implementation of Union-Find with load balancing.

    The original paper on BTS Union-Find is available at:
    https://ieeexplore.ieee.org/document/10598116
    """

    def __init__(
        self,
        union_threshold,
        parallel_num,
        parallel_id,
        remote_edge_buffers,
        max_pending_edge_buffer_task,
        num_edge_buffer_task_returns,
    ):
        self.union_threshold = union_threshold
        self.parallel_num = parallel_num
        self.parallel_id = parallel_id
        self.hash_table = {}
        self.parent = {}
        self.old_parent = {}
        self.remote_edge_buffers = remote_edge_buffers
        self.edge_buffer = []
        self.edge_list_dict = {}
        self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
        self.num_edge_buffer_task_returns = num_edge_buffer_task_returns

    def add_key_value_pairs(self, pairs):
        for key, value in pairs:
            if key not in self.hash_table:
                self.hash_table[key] = []
            self.hash_table[key].append(value)
            if len(self.hash_table[key]) > self.union_threshold:
                self.hash_table[key] = [self.union_list(self.hash_table[key])]

    def flush_key_value_pairs(self):
        for value in self.hash_table.values():
            if len(value) > 1:
                self.union_list(value)
        del self.hash_table

    def balanced_union_find(self):
        for x, y in self.edge_buffer:
            self.union(x, y)
        self.edge_buffer = []
        result_refs = []
        for remote_edge_buffer in self.remote_edge_buffers:
            if len(result_refs) > self.max_pending_edge_buffer_task:
                ready_refs, result_refs = ray.wait(
                    result_refs, num_returns=self.num_edge_buffer_task_returns)
                edge_list = ray.get(ready_refs)
                for edges in edge_list:
                    for x, y in edges:
                        self.union(x, y)
                del ready_refs
            result_refs.append(
                remote_edge_buffer.get_edges.remote(self.parallel_id))
        edge_list = ray.get(result_refs)
        for edges in edge_list:
            for x, y in edges:
                self.union(x, y)
        del edge_list, result_refs
        self.rebalancing()
        return self.old_parent != self.parent

    def distribute_edge(self, u, v):
        hash_u = u // BATCH_SIZE % self.parallel_num
        hash_v = v // BATCH_SIZE % self.parallel_num
        if hash_u not in self.edge_list_dict:
            self.edge_list_dict[hash_u] = []
        self.edge_list_dict[hash_u].append((u, v))
        if hash_u != hash_v:
            if hash_v not in self.edge_list_dict:
                self.edge_list_dict[hash_v] = []
            self.edge_list_dict[hash_v].append((u, v))

    def set_edge_buffer(self):
        if self.parallel_id in self.edge_list_dict:
            self.edge_buffer = self.edge_list_dict[self.parallel_id]
            del self.edge_list_dict[self.parallel_id]
        else:
            self.edge_buffer = []
        ray.get(self.remote_edge_buffers[self.parallel_id].set_edges.remote(
            self.edge_list_dict))
        self.edge_list_dict = {}

    def edge_redistribution(self):
        self.flush_key_value_pairs()
        self.rebalancing()
        self.edge_list_dict = {}
        for u, v in self.parent.items():
            self.distribute_edge(u, v)
        self.parent = {}
        self.set_edge_buffer()

    def communication(self):
        self.edge_list_dict = {}
        del_list = []
        for u, v in self.parent.items():
            hash_u = u // BATCH_SIZE % self.parallel_num
            if self.parent[u] != self.old_parent.get(u, u) or (
                    hash_u != self.parallel_id and v not in self.parent):
                self.distribute_edge(u, v)
            if hash_u != self.parallel_id:
                del_list.append(u)
        self.old_parent = self.parent.copy()
        for u in del_list:
            del self.parent[u]
        self.set_edge_buffer()

    def find(self, x):
        if x not in self.parent:
            return x
        else:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return
        if px > py:
            px, py = py, px
        self.parent[py] = px

    def union_list(self, x_list):
        px_list = [self.find(x) for x in x_list]
        p = min(px_list)
        for px in px_list:
            if p != px:
                self.parent[px] = p
        return p

    def rebalancing(self):
        new_px_dict = {}
        for x in self.parent:
            hash_x = x // BATCH_SIZE % self.parallel_num
            px = self.find(x)
            key = (px, hash_x)
            if key not in new_px_dict:
                new_px_dict[key] = x
            else:
                new_px_dict[key] = min(new_px_dict[key], x)
        px_set = set(px for px, _ in new_px_dict)
        for px in px_set:
            hash_px = px // BATCH_SIZE % self.parallel_num
            key = (px, hash_px)
            if key not in new_px_dict:
                new_px_dict[key] = px
            else:
                new_px_dict[key] = min(new_px_dict[key], px)

        for x in self.parent:
            hash_x = x // BATCH_SIZE % self.parallel_num
            px = self.find(x)
            key = (px, hash_x)
            if x == new_px_dict[key]:
                continue
            self.parent[x] = new_px_dict[key]

    def squeeze(self, mode="filter"):
        dup_keys = {
            x
            for x in self.parent
            if x // BATCH_SIZE % self.parallel_num == self.parallel_id
        }
        if mode == "filter":
            self.parent = dup_keys
        else:
            self.parent = {x: self.parent[x] for x in dup_keys}
        self.old_parent = {}
        self.edge_buffer = []
        ray.get(self.remote_edge_buffers[self.parallel_id].clear.remote())

    def dup_idx(self, queries):
        return [idx for uid, idx in queries if uid in self.parent]
    
    def get_root_ids(self, queries):
        """
        For each queried UID, find its root in the Union-Find structure.

        Args:
            queries: A list of tuples, where each tuple is (uid, original_batch_index).

        Returns:
            A list of tuples, where each tuple is (original_batch_index, root_id).
            The root_id is the representative ID for the set the uid belongs to.
        """
        root_id_results = []
        for uid, original_index in queries:
            try:
                root_id = self.find(uid)  # Find the root ID using path compression
            except Exception as e:
                # Log error if find fails unexpectedly
                # If find fails, assume the item is its own root
                print(f"Error finding root for UID {uid}: {e}")
                root_id = uid  # Fallback: assume it's its own root
            root_id_results.append((original_index, root_id))
        return root_id_results

from typing import Set, Iterable
from itertools import tee

NON_ALPHA = re.compile("[^A-Za-z_0-9]")

def ngrams(sequence: List[str], n: int, min_ngram_size: int = 5) -> Iterable:
    """
    Code taken from NLTK, without padding.

    Parameters
    ----------
    sequence : list
        The sequence of items to be converted into n-grams.
    n : int
        The order of the n-grams to be extracted.
    min_ngram_size : int
        The minimum number of items in the sequence to generate n-grams.

    Returns
    -------
    Iterable
        The n-grams generated from the sequence.

    Examples
    --------
    >>> list(ngrams(['a', 'b', 'c', 'd'], 2))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    >>> list(ngrams(['a', 'b', 'c', 'd'], 3))
    [('a', 'b', 'c'), ('b', 'c', 'd')]
    """
    if len(sequence) < min_ngram_size:
        return []

    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)

def tokenize(content: str, ngram_size: int, min_ngram_size: int) -> Set[str]:
    tokens = {
        " ".join(t).encode("utf-8")
        for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)
    }
    return tokens

from functools import partial

class RayBTSMinhashDeduplicator:
    """
    A MinhashLSH deduplicator based on RAY.
    """

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = 'EMPTY'
    _batched_op = True

    def __init__(
        self,
        ngram_size: PositiveInt = 5,
        min_ngram_size: PositiveInt = 5,
        num_permutations: PositiveInt = 256,
        jaccard_threshold: Annotated[float, Field(ge=0, le=1)] = 0.7,
        num_bands: Optional[PositiveInt] = None,
        num_rows_per_band: Optional[PositiveInt] = None,
        union_find_parallel_num: Union[int, str] = 'auto',
        union_threshold: Optional[int] = 256,
        max_pending_edge_buffer_task: Optional[int] = 20,
        num_edge_buffer_task_returns: Optional[int] = 10,
        max_pending_filter_tasks: Optional[int] = 20,
        num_filter_task_returns: Optional[int] = 10,
        merge_batch_size: Optional[int] = 1000,
        hashing_batch_size: Optional[int] = 10000,
        **kwargs,
    ):
        """
        Initialization method.

        :param tokenization: tokenization method for sample texts. It
            should be one of [space, punctuation, character,
            sentencepiece]. For English-like languages, we recommend
            to use 'space', for Chinese-like languages, we recommend
            to use 'character', and for multiple languages, we recommend
            to use 'sentencepiece'. If using 'sentencepiece', please
            provided the model path in the 'tokenizer_model' field.
        :param min_ngram_size: window size of shingling
        :param lowercase: whether to convert text to lower case first
        :param ignore_pattern: whether to ignore sub-strings with
            specific pattern when computing minhash
        :param num_permutations: number of permutations in minhash
            computing
        :param jaccard_threshold: the min jaccard similarity threshold
            in near-duplicate detection. When the jaccard similarity of
            two sample texts is >= this threshold, they are regarded as
            similar samples and this op will only keep one of them after
            deduplication
        :param num_bands: number of bands in LSH. Default it's None, and
            it will be determined by an optimal params computation
            algorithm by minimize the weighted sum of probs of False
            Positives and False Negatives
        :param num_rows_per_band: number of rows in each band in LSH.
            Default it's None, and it will be determined by an optimal
            params computation algorithm
        :param tokenizer_model: path for the sentencepiece model, used for
            sentencepiece tokenization.
        :param union_find_parallel_num: number of parallel workers for
            union-find algorithm. Default it's 'auto', and it will be
            determined by half of the number of CPUs.
        :param union_threshold: threshold for minhash values group to
            perform union-find algorithm. Default it's 256.
        :param max_pending_edge_buffer_task: max number of pending edge buffer
            ray tasks. Default it's 20.
        :param num_edge_buffer_task_returns: number of edge buffer tasks for
            `ray.wait` to return. Default it's 10.
        :param max_pending_filter_tasks: max number of pending filter ray
            tasks. Default it's 20.
        :param num_filter_task_returns: number of filter tasks for `ray.wait`
            to return. Default it's 10.
        :param merge_batch_size: batch size for BTS operations. Default
            it's 1000.
        :param tmp_file_name: the temporary folder name for deduplication.
        """
        self.text_key = kwargs.get('text_key', 'text')
        # self.work_dir = kwargs.get('work_dir', None)
        self.batch_size = kwargs.get('batch_size', 1000)
        self.hashing_batch_size = hashing_batch_size
        self.min_ngram_size = min_ngram_size


            
        self.tokenization_func = partial(tokenize, ngram_size=ngram_size, min_ngram_size=min_ngram_size)

        # about deduplication
        self.num_permutation = num_permutations
        self.jaccard_threshold = jaccard_threshold
        self.num_bands = num_bands
        self.num_rows_per_band = num_rows_per_band

        # initialize deduplication parameters
        # check number of bands and rows
        if self.num_bands is None or self.num_rows_per_band is None:
            self.num_bands, self.num_rows_per_band = optimal_param(
                self.jaccard_threshold,
                self.num_permutation,
            )

        # compute hash ranges and create hash tables
        self.hash_ranges = [(i * self.num_rows_per_band,
                             (i + 1) * self.num_rows_per_band)
                            for i in range(self.num_bands)]

        # generate permutations
        gen = np.random.RandomState(seed=42)
        self.perm_a, self.perm_b = np.array(
            [(
                gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            ) for _ in range(self.num_permutation)],
            dtype=np.uint64,
        ).T

        if union_find_parallel_num == 'auto':
            union_find_parallel_num = int(ray.cluster_resources().get('CPU') /
                                          2)
        else:
            union_find_parallel_num = int(union_find_parallel_num)

        self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
        self.num_edge_buffer_task_returns = num_edge_buffer_task_returns
        self.max_pending_filter_tasks = max_pending_filter_tasks
        self.num_filter_task_returns = num_filter_task_returns
        self.merge_batch_size = min(merge_batch_size, union_find_parallel_num)

        print(f'union_find_parallel_num = {union_find_parallel_num}')
        self.union_find_parallel_num = union_find_parallel_num
        self.union_threshold = union_threshold
        self.remote_edge_buffers = [
            EdgeBuffer.remote() for _ in range(self.union_find_parallel_num)
        ]
        self.union_find_list = [
            BTSUnionFind.remote(
                self.union_threshold,
                self.union_find_parallel_num,
                i,
                self.remote_edge_buffers,  # TODO: fix this
                self.max_pending_edge_buffer_task,
                self.num_edge_buffer_task_returns,
            ) for i in range(self.union_find_parallel_num)
        ]

        empty_hash_value = np.full((self.num_rows_per_band, ),
                                   MAX_HASH,
                                   dtype=np.uint32)
        self.empty_hash_value = b'\x00\x00\x00\x00' \
            + empty_hash_value.tobytes()
        self.empty_hash_table_id = int(MAX_HASH % self.union_find_parallel_num)

    def calc_minhash(self, text_list: pa.Array, uid_list: List) -> pa.Table:
        pairs = {}

        for text, uid in zip(text_list, uid_list):
            text = text.as_py()

            tokens = self.tokenization_func(text)

            if len(tokens) > 0:
                hv = np.array([sha1_hash32(token) for token in tokens],
                              dtype=np.uint64)
                phv = ((hv[:, None] * self.perm_a[None, :] + self.perm_b) %
                       MERSENNE_PRIME).astype(np.uint32)
                hash_values = phv.min(axis=0)
                for i, (start, end) in enumerate(self.hash_ranges):
                    hash_value = i.to_bytes(4, 'big') \
                        + hash_values[start:end].tobytes()
                    hash_table_id = hash_values[start] \
                        % self.union_find_parallel_num
                    if hash_table_id not in pairs:
                        pairs[hash_table_id] = []
                    pairs[hash_table_id].append((hash_value, uid))
            else:
                if self.empty_hash_table_id not in pairs:
                    pairs[self.empty_hash_table_id] = []
                pairs[self.empty_hash_table_id].append(
                    (self.empty_hash_value, uid))
        result_refs = []
        for i, p in pairs.items():
            if len(result_refs) > self.max_pending_filter_tasks:
                ready_refs, result_refs = ray.wait(
                    result_refs, num_returns=self.num_filter_task_returns)
                ray.get(ready_refs)
            result_refs.append(
                self.union_find_list[i].add_key_value_pairs.remote(p))
        ray.get(result_refs)

    def merge_op_batch(self, object_refs):
        results = []
        while object_refs:
            ready_refs, object_refs = ray.wait(object_refs,
                                               num_returns=min(
                                                   self.merge_batch_size,
                                                   len(object_refs)))
            results.extend(ray.get(ready_refs))
        return results

    def merge(self, mode="filter"):
        self.merge_op_batch([
            union_find.edge_redistribution.remote()
            for union_find in self.union_find_list
        ])
        while any(
                self.merge_op_batch([
                    union_find.balanced_union_find.remote()
                    for union_find in self.union_find_list
                ])):
            self.merge_op_batch([
                union_find.communication.remote()
                for union_find in self.union_find_list
            ])
        self.merge_op_batch([
            union_find.squeeze.remote(mode) for union_find in self.union_find_list
        ])

    def filter_with_union_find(self, samples: pa.Table) -> pa.Table:
        query_dict = {}
        for idx, uid in enumerate(samples["uid"]):
            uid = uid.as_py()
            hash_id = uid // BATCH_SIZE % self.union_find_parallel_num
            if hash_id not in query_dict:
                query_dict[hash_id] = []
            query_dict[hash_id].append((uid, idx))
        mask = np.ones(len(samples), dtype=np.bool_)
        result_refs = []
        for hash_id, query in query_dict.items():
            if len(result_refs) > self.max_pending_filter_tasks:
                ready_refs, result_refs = ray.wait(
                    result_refs, num_returns=self.num_filter_task_returns)
                results = ray.get(ready_refs)
                for result in results:
                    mask[result] = False
                del ready_refs
            result_refs.append(
                self.union_find_list[hash_id].dup_idx.remote(query))
        results = ray.get(result_refs)
        for result in results:
            mask[result] = False
        del query_dict, results
        columns_to_keep = [
            name for name in samples.column_names if name != "uid"
        ]
        return samples.select(columns_to_keep).filter(mask)
    
    def tag_with_union_find(self, samples: pa.Table) -> pa.Table:
        """
        Queries the distributed Union-Find structure to get the root ID for each sample
        and appends it as the 'duplicate_set_id' column.
        
        Unlike filter_with_union_find, this doesn't remove any rows, it just tags them.
        """
        num_samples = len(samples)
        if num_samples == 0:
            # Handle empty batches: return the batch with an empty duplicate_set_id column
            empty_duplicate_ids = pa.array([], type=pa.int64())
            return samples.append_column("duplicate_set_id", empty_duplicate_ids)

        # Prepare query dictionaries to send to the appropriate actors
        query_dict = {}
        for idx, uid in enumerate(samples["uid"]):
            uid = uid.as_py()
            # Use the same hash function as in filter_with_union_find
            hash_id = uid // BATCH_SIZE % self.union_find_parallel_num
            if hash_id not in query_dict:
                query_dict[hash_id] = []
            query_dict[hash_id].append((uid, idx))
        
        # Initialize array for storing root IDs
        root_ids_array = np.full(num_samples, -1, dtype=np.int64)  # Initialize with placeholder
        
        # Query the union_find actors for root IDs
        result_refs = []
        for hash_id, query in query_dict.items():
            if len(result_refs) > self.max_pending_filter_tasks:
                ready_refs, result_refs = ray.wait(
                    result_refs, num_returns=self.num_filter_task_returns)
                try:
                    results = ray.get(ready_refs)
                    for result_list in results:
                        if result_list:  # Check if the result list is not empty
                            for original_index, root_id in result_list:
                                if 0 <= original_index < num_samples:  # Bounds check
                                    root_ids_array[original_index] = root_id
                                else:
                                    print(f"Received out-of-bounds index {original_index} for batch size {num_samples}")
                except Exception as e:
                    print(f"Error getting results from get_root_ids: {e}")
                finally:
                    del ready_refs  # Memory management
            
            # Call the new remote method 'get_root_ids'
            result_refs.append(self.union_find_list[hash_id].get_root_ids.remote(query))
        
        # Process any remaining refs
        if result_refs:
            try:
                results = ray.get(result_refs)
                for result_list in results:
                    if result_list:  # Check if the result list is not empty
                        for original_index, root_id in result_list:
                            if 0 <= original_index < num_samples:  # Bounds check
                                root_ids_array[original_index] = root_id
                            else:
                                print(f"Received out-of-bounds index {original_index} for batch size {num_samples}")
            except Exception as e:
                print(f"Error getting remaining results from get_root_ids: {e}")
            finally:
                del result_refs  # Memory management
        
        # Sanity check and fallback for unassigned entries
        unassigned_indices = np.where(root_ids_array == -1)[0]
        if len(unassigned_indices) > 0:
            unassigned_count = len(unassigned_indices)
            print(f"Found {unassigned_count} samples missing a root_id assignment in a batch of size {num_samples}.")
            # Assign self-UID as root for unassigned entries
            uid_column = samples["uid"]  # Re-access column if needed
            for i in unassigned_indices:
                uid_scalar = uid_column[i]
                if uid_scalar.is_valid:
                    root_ids_array[i] = uid_scalar.as_py()
                else:
                    print(f"Cannot assign self-UID for missing root_id at index {i} because original UID is invalid.")
                    root_ids_array[i] = -2  # Example sentinel value
        
        # Append the new column
        try:
            tagged_table = samples.append_column("duplicate_set_id", pa.array(root_ids_array, type=pa.int64()))
            return tagged_table
        except Exception as e:
            print(f"Error appending duplicate_set_id column: {e}")
            # Fallback: return original table
            return samples
        
        
    def run(self, dataset, mode="filter", **kwargs):
        start_time = time.time()
        id_generator = IdGenerator.remote()

        def minhash_with_uid(table: pa.Table) -> pa.Table:
            num_rows = len(table)
            min_id, max_id = ray.get(id_generator.get_next_id.remote(num_rows))
            uid_list = range(min_id, max_id)
            self.calc_minhash(table[self.text_key], uid_list)
            new_table = table.append_column("uid",
                                            pa.array(list(uid_list)))
            return new_table
            
        dataset = dataset.map_batches(
            minhash_with_uid,
            batch_format='pyarrow',
            zero_copy_batch=True,
            num_cpus=1,
        ).materialize()

        end_time = time.time()
        print(f'MinHash time = {end_time - start_time}')
        

        start_time = time.time()
        self.merge(mode)
        end_time = time.time()
        print(f'merge time = {end_time - start_time}')
        
        if mode == "filter":
            result = dataset.map_batches(
                self.filter_with_union_find,
                batch_format='pyarrow',
                zero_copy_batch=True,
            )
        else:
            result = dataset.map_batches(
                self.tag_with_union_find,
                batch_format='pyarrow',
                zero_copy_batch=True,
            )
        return result


def jaccard(set_a, set_b):
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

# Count false positives per duplicate set
def _analyze_duplicate_set(group_df, threshold, ngram_size, min_ngram_size):
    if len(group_df) <= 1:  # Skip singleton groups
        return {"false_positive_rate": np.array([-1]), "false_positive_count": np.array([0]), "total_pairs": np.array([0])}
    else:
        def tok(text):
            return tokenize(text, ngram_size=ngram_size, min_ngram_size=min_ngram_size)
        # Calculate Jaccard similarity for all pairs in this group
        texts = group_df["text"].tolist()
        
        tokenized = [tok(text) for text in texts]
        
        false_positive_count = 0
        total_pairs = 0
        
        for i in range(len(tokenized)):
            for j in range(i+1, len(tokenized)):
                if i == j:
                    continue
                similarity = jaccard(tokenized[i], tokenized[j])
                if similarity < threshold:  # This pair is a false positive
                    false_positive_count += 1
                total_pairs += 1
        
        false_positive_rate = false_positive_count / total_pairs if total_pairs > 0 else 0
        
        return {"false_positive_rate": np.array([float(false_positive_rate)]), "false_positive_count": np.array([false_positive_count]), "total_pairs": np.array([total_pairs])}
        


def analyze(intermediate_ray_ds, args):
    def analyze_duplicate_set(group_df):
        return _analyze_duplicate_set(group_df, args.threshold, args.ngram_size, args.min_ngram_size)
    # Map each group to its false positive rate
    metrics = intermediate_ray_ds.groupby("duplicate_set_id").map_groups(
        analyze_duplicate_set, 
        batch_format="pandas",
        num_cpus=1
    )

    metrics = metrics.filter(lambda x: x["false_positive_rate"] >= 0)
    false_positive_rate =  metrics.mean("false_positive_rate")
    false_positive_count = metrics.sum("false_positive_count")
    total_pairs = metrics.sum("total_pairs")
    return {"false_positive_rate": float(false_positive_rate), "false_positive_count": float(false_positive_count), "total_pairs": float(total_pairs)}


def dedup(ray_df, cfg):
    cfg.args.union_find_parallel_num = 10
    cfg.args.union_threshold = 256
    return run_nd_step_for_workflow(ray_df, cfg.args)


def run_nd_step_for_workflow(ray_df, args):
    
    print(f"minhash_lsh called with args: {args}")


    
    original_count = ray_df.count()
    print(f"Original record count: {original_count}")
    
    import time
    start_time = time.time()
    deduplicator = RayBTSMinhashDeduplicator(
        text_key=args.column,
        ngram_size=args.ngram_size,
        min_ngram_size=args.min_ngram_size,
        num_permutations=args.num_perm,
        jaccard_threshold=args.threshold,
        union_find_parallel_num=args.union_find_parallel_num,
        union_threshold=args.union_threshold,
        max_pending_edge_buffer_task=args.max_pending_edge_buffer_task,
        num_edge_buffer_task_returns=args.num_edge_buffer_task_returns,
        max_pending_filter_tasks=args.max_pending_filter_tasks,
        num_filter_task_returns=args.num_filter_task_returns,
        merge_batch_size=args.merge_batch_size,
    )
    mode = args.dedup_mode
    if mode=="filter":
        deduplicated_dataset = deduplicator.run(ray_df, mode="filter").materialize()
        total_time = time.time() - start_time
        print(f"Total time taken: {total_time:.2f} seconds")
        execution_time = time.time() - start_time
        print(f"Total execution time: {execution_time:.2f} seconds")
        unique_count = deduplicated_dataset.count()
        duplicate_count = original_count - unique_count
        print(f"Duplicate count: {duplicate_count}")
        result_dataset = deduplicated_dataset
        metrics = {"duplicate_count": duplicate_count,
                   "execution_time": execution_time}
    elif mode=="tag":
        # Use tag mode to add duplicate_set_id column
        print("Running deduplication in tag mode to add duplicate_set_id column")
        result_dataset = deduplicator.run(ray_df, mode="tag").materialize()
        execution_time = time.time() - start_time
        print(f"Finished tag mode, it took {execution_time:.2f} seconds")
        metrics_ = analyze(result_dataset, args)
        # Count the number of records in each duplicate set
        grouped = result_dataset.groupby("duplicate_set_id").count().materialize()
        
        # Count singletons (sets with only 1 record)
        singleton_count = grouped.filter(lambda row: row["count()"] == 1).count()
        
        # Count duplicate sets (sets with > 1 record)
        duplicate_sets_count = grouped.filter(lambda row: row["count()"] > 1).count()
        
        # Calculate final count (if we were to deduplicate)
        final_count = singleton_count + duplicate_sets_count
        duplicate_count = original_count - final_count
        
        metrics = {"duplicate_count": duplicate_count,
                   "execution_time": execution_time,
                   **metrics_}

    else:
        assert False, "Mode not supported"
    
    return result_dataset, metrics



def main():

    ray.init()
    
    # Set more detailed logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    import pandas as pd
    import pyarrow as pa
    from ray.data import from_pandas
    
    # Sample data with duplicate and near-duplicate texts
    sample_data = pd.DataFrame({
        "id": range(3),
        "text": [
            "This is a sample document for testing Minhash LSH",
            "This is a sample document for testing Minhash LSH",
            "The quick brown fox jumps over the lazy dog"  # Exact duplicate
        ]
    })
    
    # Print sample data for verification
    print("Sample data:")
    for i, text in enumerate(sample_data["text"]):
        print(f"{i}: {text}")
    
    # Create Ray dataset
    dataset = from_pandas(sample_data)
    # os.makedirs('./ray_minhash_work_dir', exist_ok=True)
    deduplicator = RayBTSMinhashDeduplicator(
        # work_dir='./ray_minhash_work_dir',
        tokenization='space',
        min_ngram_size=5,
        lowercase=True,
        ignore_pattern=None,
        num_permutations=256,
        jaccard_threshold=0.7,
        num_bands=None,
        num_rows_per_band=None,
        union_find_parallel_num=400,
        union_threshold=256,
        max_pending_edge_buffer_task=20,
        num_edge_buffer_task_returns=10,
        max_pending_filter_tasks=20,
        num_filter_task_returns=10,
        merge_batch_size=1000,
    )
    deduplicated_dataset = deduplicator.run(dataset)
    print(deduplicated_dataset.materialize().take_all())




if __name__ == "__main__":
    main()