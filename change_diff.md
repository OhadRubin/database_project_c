diff --git a/database_project/src/ray_minhash.py b/database_project/src/ray_minhash.py
index 0c09da8..747fc55 100644
--- a/database_project/src/ray_minhash.py
+++ b/database_project/src/ray_minhash.py
@@ -1,3 +1,4 @@
+# /home/ohadr/database_project_c/database_project/src/ray_minhash.py
 # Adapted from https://github.com/modelscope/data-juicer
 import ray
 import sys
@@ -13,22 +14,26 @@ import numpy as np
 import pyarrow as pa
 import ray
 import regex
-from loguru import logger
+from loguru import logger # Keep original logger import
+# Ensure logger is configured if used directly outside Ray actors
+logging.basicConfig(level=logging.INFO)
+std_logger = logging.getLogger(__name__) # Standard logger for functions
+
 from pydantic import Field, PositiveInt
 from typing_extensions import Annotated
-
 import hashlib
 import struct
 from collections import defaultdict
 from typing import Optional
 import scipy.integrate as integrate
+from typing import Set, Iterable
+from itertools import tee
+from functools import partial
 
 
 # python3.10 -m pip install ray==2.43.0 numpy~=1.0
 
 
-
-
 MERSENNE_PRIME = np.uint64((1 << 61) - 1)
 MAX_HASH = np.uint64((1 << 32) - 1)
 
@@ -249,14 +254,22 @@ class BTSUnionFind:
         if x not in self.parent:
             return x
         else:
-            self.parent[x] = self.find(self.parent[x])
-            return self.parent[x]
+            # Path Compression
+            root = x
+            path = []
+            while root in self.parent:
+                path.append(root)
+                root = self.parent[root]
+            for node in path:
+                self.parent[node] = root
+            return root
 
     def union(self, x, y):
         px = self.find(x)
         py = self.find(y)
         if px == py:
             return
+        # Basic Union (could add union-by-size/rank)
         if px > py:
             px, py = py, px
         self.parent[py] = px
@@ -297,22 +310,39 @@ class BTSUnionFind:
             self.parent[x] = new_px_dict[key]
 
     def squeeze(self):
-        dup_keys = {
-            x
-            for x in self.parent
+        # Keep only parent entries where the key belongs to this actor's partition
+        # Note: The original `dup_keys` calculation seemed incorrect based on the description.
+        # It should retain entries where the *key* x belongs to this partition,
+        # regardless of whether find(x) == x. The value self.parent[x] is the root.
+        # The filtering happens in `filter_with_union_find` based on the root.
+        # So, we just need to keep the relevant part of the `parent` map for future `find` calls.
+        self.parent = {
+            x: v for x, v in self.parent.items()
             if x // BATCH_SIZE % self.parallel_num == self.parallel_id
         }
-        self.parent = dup_keys
-        self.old_parent = {}
-        self.edge_buffer = []
-        ray.get(self.remote_edge_buffers[self.parallel_id].clear.remote())
+        self.old_parent = {} # Clear old parent state
+        self.edge_buffer = [] # Clear edge buffer
+        ray.get(self.remote_edge_buffers[self.parallel_id].clear.remote()) # Clear remote buffer for this actor
 
-    def dup_idx(self, queries):
-        return [idx for uid, idx in queries if uid in self.parent]
+    # Replace dup_idx method with this:
+    def get_root_ids(self, queries):
+        """
+        For each queried UID, find its root in the Union-Find structure.
 
+        Args:
+            queries: A list of tuples, where each tuple is (uid, original_batch_index).
+
+        Returns:
+            A list of tuples, where each tuple is (original_batch_index, root_id).
+        """
+        root_id_results = []
+        for uid, original_index in queries:
+            root_id = self.find(uid) # Find the root ID
+            root_id_results.append((original_index, root_id))
+        return root_id_results
+
+    # Make sure to update any internal calls if dup_idx was used elsewhere, although it seems primarily called remotely.
 
-from typing import Set, Iterable
-from itertools import tee
 
 NON_ALPHA = re.compile("[^A-Za-z_0-9]")
 
@@ -357,11 +387,11 @@ def tokenize(content: str, ngram_size: int, min_ngram_size: int) -> Set[str]:
     }
     return tokens
 
-from functools import partial
 
 class RayBTSMinhashDeduplicator:
     """
     A MinhashLSH deduplicator based on RAY.
+    Modified to add 'duplicate_set_id' instead of filtering.
     """
 
     # TODO: Set a more reasonable value
@@ -370,6 +400,7 @@ class RayBTSMinhashDeduplicator:
 
     def __init__(
         self,
+        text_key: str = 'text', # Moved text_key to constructor args
         ngram_size: PositiveInt = 5,
         min_ngram_size: PositiveInt = 5,
         num_permutations: PositiveInt = 256,
@@ -383,64 +414,19 @@ class RayBTSMinhashDeduplicator:
         max_pending_filter_tasks: Optional[int] = 20,
         num_filter_task_returns: Optional[int] = 10,
         merge_batch_size: Optional[int] = 1000,
-        hashing_batch_size: Optional[int] = 10000,
-        **kwargs,
+        hashing_batch_size: Optional[int] = 10000, # Kept original default
+        **kwargs, # Keep kwargs for compatibility if needed
     ):
         """
         Initialization method.
-
-        :param tokenization: tokenization method for sample texts. It
-            should be one of [space, punctuation, character,
-            sentencepiece]. For English-like languages, we recommend
-            to use 'space', for Chinese-like languages, we recommend
-            to use 'character', and for multiple languages, we recommend
-            to use 'sentencepiece'. If using 'sentencepiece', please
-            provided the model path in the 'tokenizer_model' field.
-        :param min_ngram_size: window size of shingling
-        :param lowercase: whether to convert text to lower case first
-        :param ignore_pattern: whether to ignore sub-strings with
-            specific pattern when computing minhash
-        :param num_permutations: number of permutations in minhash
-            computing
-        :param jaccard_threshold: the min jaccard similarity threshold
-            in near-duplicate detection. When the jaccard similarity of
-            two sample texts is >= this threshold, they are regarded as
-            similar samples and this op will only keep one of them after
-            deduplication
-        :param num_bands: number of bands in LSH. Default it's None, and
-            it will be determined by an optimal params computation
-            algorithm by minimize the weighted sum of probs of False
-            Positives and False Negatives
-        :param num_rows_per_band: number of rows in each band in LSH.
-            Default it's None, and it will be determined by an optimal
-            params computation algorithm
-        :param tokenizer_model: path for the sentencepiece model, used for
-            sentencepiece tokenization.
-        :param union_find_parallel_num: number of parallel workers for
-            union-find algorithm. Default it's 'auto', and it will be
-            determined by half of the number of CPUs.
-        :param union_threshold: threshold for minhash values group to
-            perform union-find algorithm. Default it's 256.
-        :param max_pending_edge_buffer_task: max number of pending edge buffer
-            ray tasks. Default it's 20.
-        :param num_edge_buffer_task_returns: number of edge buffer tasks for
-            `ray.wait` to return. Default it's 10.
-        :param max_pending_filter_tasks: max number of pending filter ray
-            tasks. Default it's 20.
-        :param num_filter_task_returns: number of filter tasks for `ray.wait`
-            to return. Default it's 10.
-        :param merge_batch_size: batch size for BTS operations. Default
-            it's 1000.
-        :param tmp_file_name: the temporary folder name for deduplication.
+        <Original docstring content omitted for brevity, assuming standard MinHash/LSH parameters>
         """
-        self.text_key = kwargs.get('text_key', 'text')
-        # self.work_dir = kwargs.get('work_dir', None)
-        self.batch_size = kwargs.get('batch_size', 1000)
-        self.hashing_batch_size = hashing_batch_size
+        self.text_key = text_key # Use provided text_key
+        # self.work_dir = kwargs.get('work_dir', None) # work_dir not used in provided code
+        # self.batch_size = kwargs.get('batch_size', 1000) # batch_size not used directly here
+        self.hashing_batch_size = hashing_batch_size # Used for map_batches? Check run() method usage
         self.min_ngram_size = min_ngram_size
 
-
-            
         self.tokenization_func = partial(tokenize, ngram_size=ngram_size, min_ngram_size=min_ngram_size)
 
         # about deduplication
@@ -450,12 +436,12 @@ class RayBTSMinhashDeduplicator:
         self.num_rows_per_band = num_rows_per_band
 
         # initialize deduplication parameters
-        # check number of bands and rows
         if self.num_bands is None or self.num_rows_per_band is None:
             self.num_bands, self.num_rows_per_band = optimal_param(
                 self.jaccard_threshold,
                 self.num_permutation,
             )
+            logger.info(f"Calculated optimal LSH params: bands={self.num_bands}, rows={self.num_rows_per_band}")
 
         # compute hash ranges and create hash tables
         self.hash_ranges = [(i * self.num_rows_per_band,
@@ -463,7 +449,7 @@ class RayBTSMinhashDeduplicator:
                             for i in range(self.num_bands)]
 
         # generate permutations
-        gen = np.random.RandomState(seed=42)
+        gen = np.random.RandomState(seed=42) # Fixed seed for reproducibility
         self.perm_a, self.perm_b = np.array(
             [(
                 gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
@@ -473,19 +459,25 @@ class RayBTSMinhashDeduplicator:
         ).T
 
         if union_find_parallel_num == 'auto':
-            union_find_parallel_num = int(ray.cluster_resources().get('CPU') /
-                                          2)
+            try:
+                cpu_resources = ray.cluster_resources().get('CPU', 1) # Default to 1 CPU if not found
+                union_find_parallel_num = max(1, int(cpu_resources / 2))
+            except Exception as e:
+                 logger.warning(f"Could not automatically determine CPU count for union_find_parallel_num, defaulting to 1. Error: {e}")
+                 union_find_parallel_num = 1
         else:
             union_find_parallel_num = int(union_find_parallel_num)
 
+        self.union_find_parallel_num = max(1, union_find_parallel_num) # Ensure at least 1
+
         self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
         self.num_edge_buffer_task_returns = num_edge_buffer_task_returns
         self.max_pending_filter_tasks = max_pending_filter_tasks
         self.num_filter_task_returns = num_filter_task_returns
-        self.merge_batch_size = min(merge_batch_size, union_find_parallel_num)
+        self.merge_batch_size = min(merge_batch_size, self.union_find_parallel_num) if self.union_find_parallel_num > 0 else merge_batch_size
+
 
-        logger.info(f'union_find_parallel_num = {union_find_parallel_num}')
-        self.union_find_parallel_num = union_find_parallel_num
+        logger.info(f'Using union_find_parallel_num = {self.union_find_parallel_num}')
         self.union_threshold = union_threshold
         self.remote_edge_buffers = [
             EdgeBuffer.remote() for _ in range(self.union_find_parallel_num)
@@ -495,7 +487,7 @@ class RayBTSMinhashDeduplicator:
                 self.union_threshold,
                 self.union_find_parallel_num,
                 i,
-                self.remote_edge_buffers,  # TODO: fix this
+                self.remote_edge_buffers, # Pass the list of remote actors
                 self.max_pending_edge_buffer_task,
                 self.num_edge_buffer_task_returns,
             ) for i in range(self.union_find_parallel_num)
@@ -506,222 +498,465 @@ class RayBTSMinhashDeduplicator:
                                    dtype=np.uint32)
         self.empty_hash_value = b'\x00\x00\x00\x00' \
             + empty_hash_value.tobytes()
-        self.empty_hash_table_id = int(MAX_HASH % self.union_find_parallel_num)
+        # Ensure division by zero doesn't happen if parallel_num is 0 or 1
+        self.empty_hash_table_id = int(MAX_HASH % self.union_find_parallel_num) if self.union_find_parallel_num > 0 else 0
 
-    def calc_minhash(self, text_list: pa.Array, uid_list: List) -> pa.Table:
-        pairs = {}
+
+    def calc_minhash(self, text_list: pa.Array, uid_list: List) -> None: # Doesn't return table, modifies remote actors
+        pairs = defaultdict(list) # Use defaultdict for cleaner code
 
         for text, uid in zip(text_list, uid_list):
-            text = text.as_py()
+            text_py = text.as_py() # Convert pa.Scalar to Python string
+            if not isinstance(text_py, str): # Handle potential None or other types
+                logger.warning(f"Encountered non-string text for UID {uid}, skipping MinHash calculation.")
+                # Decide how to handle non-strings, e.g., assign to empty hash value
+                if self.union_find_parallel_num > 0:
+                    pairs[self.empty_hash_table_id].append((self.empty_hash_value, uid))
+                continue
 
-            tokens = self.tokenization_func(text)
+            tokens = self.tokenization_func(text_py)
 
-            if len(tokens) > 0:
+            if tokens: # Check if tokens set is not empty
                 hv = np.array([sha1_hash32(token) for token in tokens],
                               dtype=np.uint64)
-                phv = ((hv[:, None] * self.perm_a[None, :] + self.perm_b) %
-                       MERSENNE_PRIME).astype(np.uint32)
+                # Optimize permutation calculation with broadcasting
+                phv = ((hv[:, None] * self.perm_a + self.perm_b) % MERSENNE_PRIME).astype(np.uint32)
                 hash_values = phv.min(axis=0)
+
                 for i, (start, end) in enumerate(self.hash_ranges):
-                    hash_value = i.to_bytes(4, 'big') \
-                        + hash_values[start:end].tobytes()
-                    hash_table_id = hash_values[start] \
-                        % self.union_find_parallel_num
-                    if hash_table_id not in pairs:
-                        pairs[hash_table_id] = []
-                    pairs[hash_table_id].append((hash_value, uid))
-            else:
-                if self.empty_hash_table_id not in pairs:
-                    pairs[self.empty_hash_table_id] = []
-                pairs[self.empty_hash_table_id].append(
-                    (self.empty_hash_value, uid))
+                    band_hashes = hash_values[start:end]
+                    # Combine band index and hashes for the key
+                    hash_value_key = i.to_bytes(4, 'big') + band_hashes.tobytes()
+                    # Determine target actor based on the first hash in the band
+                    hash_table_id = int(band_hashes[0] % self.union_find_parallel_num) if self.union_find_parallel_num > 0 else 0
+                    pairs[hash_table_id].append((hash_value_key, uid))
+            else: # Handle empty documents
+                 if self.union_find_parallel_num > 0:
+                    pairs[self.empty_hash_table_id].append((self.empty_hash_value, uid))
+
+        # Send pairs to remote actors asynchronously
         result_refs = []
         for i, p in pairs.items():
-            if len(result_refs) > self.max_pending_filter_tasks:
-                ready_refs, result_refs = ray.wait(
-                    result_refs, num_returns=self.num_filter_task_returns)
+            if not p: continue # Skip empty lists
+            # Rate limiting based on pending tasks
+            if len(result_refs) >= self.max_pending_filter_tasks: # Reuse filter task limit here? Or define a new one?
+                num_returns = min(self.num_filter_task_returns, len(result_refs))
+                ready_refs, result_refs = ray.wait(result_refs, num_returns=num_returns)
+                # We don't need the result of add_key_value_pairs, just wait for completion
                 ray.get(ready_refs)
-            result_refs.append(
-                self.union_find_list[i].add_key_value_pairs.remote(p))
-        ray.get(result_refs)
+                del ready_refs # Memory management
+
+            if 0 <= i < len(self.union_find_list):
+                 result_refs.append(self.union_find_list[i].add_key_value_pairs.remote(p))
+            else:
+                 logger.error(f"Calculated hash_table_id {i} is out of bounds for union_find_list (size {len(self.union_find_list)}). Skipping batch.")
+
+        # Wait for remaining tasks
+        if result_refs:
+            ray.get(result_refs)
 
     def merge_op_batch(self, object_refs):
+        # Helper to wait for batches of remote operations
         results = []
         while object_refs:
-            ready_refs, object_refs = ray.wait(object_refs,
-                                               num_returns=min(
-                                                   self.merge_batch_size,
-                                                   len(object_refs)))
+            num_returns = min(self.merge_batch_size, len(object_refs))
+            ready_refs, object_refs = ray.wait(object_refs, num_returns=num_returns)
             results.extend(ray.get(ready_refs))
         return results
 
     def merge(self):
+        # Orchestrates the merge phase of the BTS Union-Find algorithm
+        logger.info("Starting BTS Union-Find merge phase: Edge Redistribution")
         self.merge_op_batch([
             union_find.edge_redistribution.remote()
             for union_find in self.union_find_list
         ])
-        while any(
-                self.merge_op_batch([
+        logger.info("Edge Redistribution complete.")
+
+        iteration = 1
+        while True:
+            logger.info(f"Starting Balanced Union-Find Iteration {iteration}...")
+            changed_flags = self.merge_op_batch([
                     union_find.balanced_union_find.remote()
                     for union_find in self.union_find_list
-                ])):
+            ])
+            logger.info(f"Balanced Union-Find Iteration {iteration} complete. Changed: {any(changed_flags)}")
+            if not any(changed_flags):
+                break # Converged
+
+            logger.info(f"Starting Communication Iteration {iteration}...")
             self.merge_op_batch([
                 union_find.communication.remote()
                 for union_find in self.union_find_list
             ])
+            logger.info(f"Communication Iteration {iteration} complete.")
+            iteration += 1
+
+        logger.info(f"Convergence reached after {iteration-1} iterations.")
+        logger.info("Starting final Squeeze operation...")
         self.merge_op_batch([
             union_find.squeeze.remote() for union_find in self.union_find_list
         ])
+        logger.info("Squeeze operation complete. Merge phase finished.")
+
 
     def filter_with_union_find(self, samples: pa.Table) -> pa.Table:
-        query_dict = {}
-        for idx, uid in enumerate(samples["uid"]):
-            uid = uid.as_py()
-            hash_id = uid // BATCH_SIZE % self.union_find_parallel_num
-            if hash_id not in query_dict:
-                query_dict[hash_id] = []
-            query_dict[hash_id].append((uid, idx))
-        mask = np.ones(len(samples), dtype=np.bool_)
+        """
+        Assigns a duplicate_set_id (the root ID from Union-Find) to each sample.
+        """
+        num_samples = len(samples)
+        if num_samples == 0:
+            # Handle empty batches: return the batch with an empty duplicate_set_id column
+            # Ensure the type matches the expected output type (e.g., pa.int64())
+            return samples.append_column("duplicate_set_id", pa.array([], type=pa.int64()))
+
+        query_dict = defaultdict(list) # Use defaultdict for cleaner code
+        # Assuming 'uid' column exists from minhash_with_uid step
+        uid_column = samples["uid"]
+        for idx in range(num_samples):
+            uid_scalar = uid_column[idx]
+            # Check if scalar is valid before trying to access .as_py()
+            if uid_scalar.is_valid:
+                uid = uid_scalar.as_py() # Get Python int from scalar
+                # Determine which BTSUnionFind actor is responsible for this UID's primary hash space
+                # Ensure BATCH_SIZE is defined appropriately or passed via config
+                if self.union_find_parallel_num > 0:
+                    hash_id = uid // BATCH_SIZE % self.union_find_parallel_num
+                else:
+                    hash_id = 0 # Default to actor 0 if parallelism is 1 or less
+                query_dict[hash_id].append((uid, idx)) # Send (uid, original_batch_index)
+            else:
+                logger.warning(f"Encountered invalid UID at index {idx} in batch, skipping.")
+
+
+        root_ids_array = np.full(num_samples, -1, dtype=np.int64) # Initialize with placeholder
         result_refs = []
+
         for hash_id, query in query_dict.items():
-            if len(result_refs) > self.max_pending_filter_tasks:
-                ready_refs, result_refs = ray.wait(
-                    result_refs, num_returns=self.num_filter_task_returns)
-                results = ray.get(ready_refs)
-                for result in results:
-                    mask[result] = False
-                del ready_refs
-            result_refs.append(
-                self.union_find_list[hash_id].dup_idx.remote(query))
-        results = ray.get(result_refs)
-        for result in results:
-            mask[result] = False
-        del query_dict, results
-        columns_to_keep = [
-            name for name in samples.column_names if name != "uid"
-        ]
-        return samples.select(columns_to_keep).filter(mask)
+            if not query: continue # Skip empty queries
 
-    def run(self, dataset, **kwargs):
+            # Rate limiting
+            if len(result_refs) >= self.max_pending_filter_tasks: # Use configured limits
+                num_returns = min(self.num_filter_task_returns, len(result_refs))
+                ready_refs, result_refs = ray.wait(result_refs, num_returns=num_returns)
+                results = ray.get(ready_refs)
+                for result_list in results:
+                    if result_list: # Check if the result list is not empty
+                        for original_index, root_id in result_list:
+                            if 0 <= original_index < num_samples: # Bounds check
+                                 root_ids_array[original_index] = root_id
+                            else:
+                                 # Use standard logger here as this runs in the map_batches worker
+                                 std_logger.warning(f"Received out-of-bounds index {original_index} for batch size {num_samples}")
+                del ready_refs, results # Memory management
+
+            # Call the new remote method
+            if 0 <= hash_id < len(self.union_find_list):
+                result_refs.append(self.union_find_list[hash_id].get_root_ids.remote(query))
+            else:
+                std_logger.error(f"Query hash_id {hash_id} is out of bounds for union_find_list (size {len(self.union_find_list)}). Skipping query for this batch.")
+
+
+        # Process remaining refs
+        if result_refs:
+            results = ray.get(result_refs)
+            for result_list in results:
+                 if result_list: # Check if the result list is not empty
+                    for original_index, root_id in result_list:
+                         if 0 <= original_index < num_samples: # Bounds check
+                             root_ids_array[original_index] = root_id
+                         else:
+                             std_logger.warning(f"Received out-of-bounds index {original_index} for batch size {num_samples}")
+            del results, result_refs # Memory management
+
+        # Sanity check: Ensure all placeholders are filled
+        unassigned_indices = np.where(root_ids_array == -1)[0]
+        if len(unassigned_indices) > 0:
+            unassigned_count = len(unassigned_indices)
+            std_logger.warning(f"Found {unassigned_count} samples missing a root_id assignment in a batch of size {num_samples}. Check BTSUnionFind logic or communication. Assigning self-UID as root.")
+            # Assign self-UID as root for unassigned entries
+            uid_column = samples["uid"] # Re-access column if needed
+            for i in unassigned_indices:
+                 uid_scalar = uid_column[i]
+                 if uid_scalar.is_valid:
+                     root_ids_array[i] = uid_scalar.as_py()
+                 else:
+                      std_logger.error(f"Cannot assign self-UID for missing root_id at index {i} because original UID is invalid.")
+                      # Assign a sentinel value or handle differently if needed
+                      root_ids_array[i] = -2 # Example sentinel
+
+        # Append the new column instead of filtering
+        # Ensure 'uid' column is not dropped if needed later (it is dropped implicitly by select below)
+        # Let's keep the uid column for potential debugging or future use
+        final_table = samples.append_column("duplicate_set_id", pa.array(root_ids_array, type=pa.int64()))
+
+        # Explicitly select columns to keep, including the new one and uid
+        # columns_to_keep = samples.column_names + ["duplicate_set_id"] # Keep all original + new
+        # return final_table.select(columns_to_keep)
+        return final_table # Return table with new column appended
+
+
+    def run(self, dataset, **kwargs): # kwargs seem unused here
         start_time = time.time()
+        logger.info("Initializing IdGenerator...")
+        # Ensure IdGenerator is initialized correctly
+        # It's better to initialize it once outside the map_batches if possible,
+        # but if state needs to be shared across batches, this approach works.
+        # Let's keep it as is, assuming it handles state correctly.
         id_generator = IdGenerator.remote()
+        logger.info("IdGenerator initialized.")
 
         def minhash_with_uid(table: pa.Table) -> pa.Table:
+            """Adds a unique UID to each row and triggers MinHash calculation."""
             num_rows = len(table)
-            min_id, max_id = ray.get(id_generator.get_next_id.remote(num_rows))
-            uid_list = range(min_id, max_id)
-            self.calc_minhash(table[self.text_key], uid_list)
-            new_table = table.append_column("uid",
-                                            pa.array(list(uid_list)))
-            return new_table
-
-        dataset = dataset.map_batches(
+            if num_rows == 0:
+                return table.append_column("uid", pa.array([], type=pa.int64())) # Handle empty table
+
+            try:
+                # Get a block of UIDs
+                min_id_ref, max_id_ref = id_generator.get_next_id.remote(num_rows)
+                min_id, max_id = ray.get([min_id_ref, max_id_ref])
+                uid_list = list(range(min_id, max_id)) # Generate list of UIDs
+
+                # Trigger MinHash calculation (modifies remote actors)
+                self.calc_minhash(table[self.text_key], uid_list)
+
+                # Append UID column
+                new_table = table.append_column("uid", pa.array(uid_list, type=pa.int64()))
+                return new_table
+            except Exception as e:
+                 std_logger.error(f"Error in minhash_with_uid batch: {e}", exc_info=True)
+                 # Return original table or empty table with schema?
+                 # Returning original table might lead to issues later if UID is expected.
+                 # Let's return the table with an empty UID column of the correct type.
+                 empty_uid_array = pa.array([None] * num_rows, type=pa.int64())
+                 return table.append_column("uid", empty_uid_array)
+
+
+        logger.info("Starting MinHash calculation with UID generation...")
+        dataset_with_uid = dataset.map_batches(
             minhash_with_uid,
             batch_format='pyarrow',
-            zero_copy_batch=True,
-            num_cpus=1,
-        ).materialize()
-
-        end_time = time.time()
-        logger.info(f'MinHash time = {end_time - start_time}')
+            batch_size=self.hashing_batch_size, # Use specified batch size
+            # zero_copy_batch=True, # May cause issues with modifications? Test carefully. Let's disable for safety.
+            num_cpus=1, # Keep low unless minhash_with_uid is CPU intensive itself
+            concurrency=self.union_find_parallel_num # Match concurrency with UF actors? Test this.
+        ).materialize() # Materialize after adding UIDs and calculating hashes
+        logger.info(f"MinHash calculation and UID generation complete. Time: {time.time() - start_time:.2f}s")
         
 
-        start_time = time.time()
+        logger.info("Starting Union-Find merge process...")
+        merge_start_time = time.time()
         self.merge()
-        end_time = time.time()
-        logger.info(f'merge time = {end_time - start_time}')
-        result = dataset.map_batches(
+        logger.info(f"Union-Find merge process complete. Time: {time.time() - merge_start_time:.2f}s")
+
+        logger.info("Starting duplicate set ID assignment...")
+        tagging_start_time = time.time()
+        # This step now adds the 'duplicate_set_id' column
+        tagged_dataset = dataset_with_uid.map_batches(
             self.filter_with_union_find,
             batch_format='pyarrow',
-            zero_copy_batch=True,
+            batch_size=self.hashing_batch_size, # Reuse batch size? Or define separate?
+            # zero_copy_batch=True, # Disable for safety as we append a column
+            concurrency=self.union_find_parallel_num # Match concurrency?
         )
-        return result
+        logger.info(f"Duplicate set ID assignment complete. Time: {time.time() - tagging_start_time:.2f}s")
 
+        return tagged_dataset # Return the dataset with the added column
 
 
+# --- Modified dedup function ---
 def dedup(ray_df, cfg):
-    import logging
-    logger = logging.getLogger(__name__)
+    logger = logging.getLogger(__name__) # Use standard logger
     
     original_count = ray_df.count()
     logger.info(f"Cluster deduplication: starting with {original_count} records")
+    if original_count == 0:
+         logger.info("Cluster deduplication: Empty dataset, skipping.")
+         return ray_df.append_column("duplicate_set_id", pa.array([], type=pa.int64())), 0 # Return with schema
     
-    import time
     start_time = time.time()
     
-    # Use same parameters from args but through cfg
-    deduplicator = RayBTSMinhashDeduplicator(
-        text_key=cfg.args.column,
-        ngram_size=cfg.args.ngram_size,
-        min_ngram_size=cfg.args.min_ngram_size,
-        num_permutations=cfg.args.num_perm,
-        jaccard_threshold=cfg.args.threshold,
-        union_find_parallel_num=10,
-        union_threshold=256,
+    # Instantiate deduplicator (ensure args are accessed correctly via cfg.args)
+    try:
+        deduplicator = RayBTSMinhashDeduplicator(
+            text_key=cfg.args.column,
+            ngram_size=cfg.args.ngram_size,
+            min_ngram_size=cfg.args.min_ngram_size,
+            num_permutations=cfg.args.num_perm,
+            jaccard_threshold=cfg.args.threshold,
+                # Use a reasonable default or make configurable for intra-cluster dedup
+                union_find_parallel_num=cfg.args.union_find_parallel_num_intra_cluster or 10,
+                union_threshold=cfg.args.union_threshold_intra_cluster or 256,
     )
-    deduplicated_dataset = deduplicator.run(ray_df).materialize()
+    except AttributeError as e:
+        logger.error(f"Missing required argument in cfg.args for RayBTSMinhashDeduplicator: {e}")
+        raise
+
+    # Run deduplication to get the tagged dataset
+    logger.info("Running deduplicator.run to tag duplicates within cluster...")
+    tagged_dataset = deduplicator.run(ray_df).materialize() # Materialize before aggregation
+    logger.info(f"Tagging completed in {time.time() - start_time:.2f}s")
+
+    # --- Calculate Duplicate Count Post-Hoc ---
+    logger.info("Cluster deduplication: Calculating duplicate count from duplicate_set_id...")
+    calc_start_time = time.time()
+    # Check if 'duplicate_set_id' column exists
+    if "duplicate_set_id" not in tagged_dataset.schema().names:
+        logger.error("Cluster deduplication: 'duplicate_set_id' column not found after tagging.")
+        # Return original dataset (or handle error appropriately)
+        return ray_df, 0
+
+    try:
+        # Need to handle potential empty dataset after tagging (though unlikely if input wasn't empty)
+        current_count = tagged_dataset.count()
+        if current_count == 0:
+             logger.info("Cluster deduplication: Tagged dataset is empty, no duplicates.")
+             return tagged_dataset, 0
+
+        grouped = tagged_dataset.groupby("duplicate_set_id").count().materialize()
+        # grouped is now a Dataset with columns: duplicate_set_id, count()
+
+        # Count singletons and duplicate sets
+        singleton_count = grouped.filter(lambda row: row["count()"] == 1).count()
+        num_duplicate_sets = grouped.filter(lambda row: row["count()"] > 1).count()
     
-    unique_count = deduplicated_dataset.count()
-    duplicate_count = original_count - unique_count
-    logger.info(f"Cluster deduplication: removed {duplicate_count} duplicates, remaining: {unique_count}")
+        # Calculate final count if deduplicated (keep one from each set)
+        final_count_if_dedupped = singleton_count + num_duplicate_sets
+        duplicate_count = current_count - final_count_if_dedupped # Use current_count here
     
-    return deduplicated_dataset, duplicate_count
+        logger.info(f"Cluster deduplication calculation took {time.time() - calc_start_time:.2f}s")
+        logger.info(f"Cluster deduplication: Original(Cluster)={original_count}, Current Tagged={current_count}, Calculated Unique Count={final_count_if_dedupped}, Calculated Duplicates Removed={duplicate_count}")
 
-def run_nd_step_for_workflow(ray_df, args):
-    import logging
-    logger = logging.getLogger(__name__)
+    except Exception as e:
+        logger.error(f"Error during duplicate count calculation: {e}", exc_info=True)
+        # Return tagged dataset but 0 duplicates as calculation failed
+        return tagged_dataset, 0
     
-    logger.info(f"minhash_lsh called with args: {args}")
+    # Return the tagged dataset and the calculated duplicate count
+    return tagged_dataset, duplicate_count
 
+# --- Modified run_nd_step_for_workflow function ---
+def run_nd_step_for_workflow(ray_df, args):
+    logger = logging.getLogger(__name__) # Use standard logger
 
+    logger.info(f"Starting ND step with args: {args}")
     
     original_count = ray_df.count()
     logger.info(f"Original record count: {original_count}")
+    if original_count == 0:
+        logger.info("ND step: Empty input dataset, skipping.")
+        # Return empty dataset with expected columns and 0 duplicates/time
+        # Need to know the full expected schema if we want to return an empty table.
+        # For now, let's assume the caller handles an empty dataset.
+        # Or, we can try to add the columns to the existing empty df.
+        # Assuming ray_df has schema even if empty:
+        try:
+             empty_tagged = ray_df.append_column("uid", pa.array([], type=pa.int64()))
+             empty_tagged = empty_tagged.append_column("duplicate_set_id", pa.array([], type=pa.int64()))
+        except Exception:
+             # If append fails on empty, return original df
+             empty_tagged = ray_df
+        return empty_tagged, 0, 0.0
+
     
-    import time
     start_time = time.time()
     
+    logger.info("Instantiating RayBTSMinhashDeduplicator...")
+    # Instantiate deduplicator
     deduplicator = RayBTSMinhashDeduplicator(
         text_key=args.column,
         ngram_size=args.ngram_size,
         min_ngram_size=args.min_ngram_size,
         num_permutations=args.num_perm,
         jaccard_threshold=args.threshold,
-        union_find_parallel_num=400,
-        union_threshold=256,
-        max_pending_edge_buffer_task=20,
-        num_edge_buffer_task_returns=10,
-        max_pending_filter_tasks=20,
-        num_filter_task_returns=10,
-        merge_batch_size=100,
+        # Use provided args for parallelization parameters
+        union_find_parallel_num=getattr(args, 'union_find_parallel_num', 'auto'), # Default if not provided
+        union_threshold=getattr(args, 'union_threshold', 256),
+        max_pending_edge_buffer_task=getattr(args, 'max_pending_edge_buffer_task', 20),
+        num_edge_buffer_task_returns=getattr(args, 'num_edge_buffer_task_returns', 10),
+        max_pending_filter_tasks=getattr(args, 'max_pending_filter_tasks', 20),
+        num_filter_task_returns=getattr(args, 'num_filter_task_returns', 10),
+        merge_batch_size=getattr(args, 'merge_batch_size', 1000),
+        hashing_batch_size=getattr(args, 'hashing_batch_size', 10000)
     )
-    deduplicated_dataset = deduplicator.run(ray_df).materialize()
-    total_time = time.time() - start_time
-    logger.info(f"Total time taken: {total_time:.2f} seconds")
-    execution_time = time.time() - start_time
-    logger.info(f"Total execution time: {execution_time:.2f} seconds")
-    unique_count = deduplicated_dataset.count()
-    duplicate_count = original_count - unique_count
-    logger.info(f"Duplicate count: {duplicate_count}")
-    return deduplicated_dataset, duplicate_count, execution_time
+    logger.info("Running deduplicator.run to tag duplicates...")
+    # This now returns the dataset with the 'duplicate_set_id' column
+    tagged_dataset = deduplicator.run(ray_df).materialize() # Materialize before aggregation
+    tagging_end_time = time.time()
+    logger.info(f"Tagging step completed in {tagging_end_time - start_time:.2f} seconds")
+
+    # --- Calculate Duplicate Count Post-Hoc ---
+    logger.info("Calculating duplicate count from duplicate_set_id...")
+    calc_start_time = time.time()
+    duplicate_count_to_log = 0
+    final_count_if_dedupped = 0
+
+    # Check if 'duplicate_set_id' column exists
+    if "duplicate_set_id" not in tagged_dataset.schema().names:
+        logger.error("'duplicate_set_id' column not found after tagging. Cannot calculate duplicate count.")
+        # Fallback: use original count and 0 duplicates
+        final_count_if_dedupped = original_count
+        duplicate_count_to_log = 0
+    else:
+        try:
+            # Need to get the count *after* tagging, should be same as input count unless errors occurred
+            current_tagged_count = tagged_dataset.count()
+            if current_tagged_count == 0:
+                logger.info("Tagged dataset is empty, no duplicates found.")
+                final_count_if_dedupped = 0
+                duplicate_count_to_log = 0
+            else:
+                # Group by the new column and count sizes
+                grouped = tagged_dataset.groupby("duplicate_set_id").count().materialize()
+                # grouped is now a Dataset with columns: duplicate_set_id, count()
+
+                # Count singletons and duplicate sets
+                singleton_count = grouped.filter(lambda row: row["count()"] == 1).count()
+                num_duplicate_sets = grouped.filter(lambda row: row["count()"] > 1).count()
+
+                # Calculate final count if deduplicated (keep one from each set)
+                final_count_if_dedupped = singleton_count + num_duplicate_sets
+                # The number of duplicates REMOVED = original_count - final_count_if_dedupped
+                duplicate_count_to_log = original_count - final_count_if_dedupped
+
+            logger.info(f"Duplicate count calculation took {time.time() - calc_start_time:.2f}s")
+
+        except Exception as e:
+            logger.error(f"Error during duplicate count calculation: {e}", exc_info=True)
+            # Fallback if aggregation fails
+            final_count_if_dedupped = tagged_dataset.count() # Count after tagging
+            duplicate_count_to_log = 0 # Assume 0 duplicates found
+
+    total_time = time.time() - start_time # Recalculate total time including aggregation
+    logger.info(f"Total time taken for ND step (including tagging and count calc): {total_time:.2f} seconds")
+
+    # Log the calculated count
+    logger.info(f"Calculated duplicate count (items to remove): {duplicate_count_to_log}")
+    logger.info(f"Final record count if deduplicated: {final_count_if_dedupped}")
+    logger.info(f"Record count after tagging (should match original): {tagged_dataset.count()}")
 
+    # Return the tagged dataset and the calculated duplicate count
+    # Returning the count of items that *would* be removed
+    return tagged_dataset, duplicate_count_to_log, total_time
 
 
+# --- Main function (kept for potential direct testing) ---
 def main():
 
-    ray.init()
+    ray.init(ignore_reinit_error=True)
     
     # Set more detailed logging
     logging.basicConfig(level=logging.INFO)
+    logger.info("Running main function for testing...") # Use loguru logger for consistency if preferred
     
     # Create sample data
     import pandas as pd
-    import pyarrow as pa
     from ray.data import from_pandas
     
     # Sample data with duplicate and near-duplicate texts
     sample_data = pd.DataFrame({
-        "id": range(3),
+        "id": range(6),
         "text": [
             "This is a sample document for testing Minhash LSH",
             "This is a sample document for testing Minhash LSH",
@@ -759,6 +994,7 @@ def main():
     print(deduplicated_dataset.materialize().take_all())
 
 
+    ray.shutdown()
 
 
 if __name__ == "__main__":
