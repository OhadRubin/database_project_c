Okay, let's dive into the small details of implementing the `duplicate_set_id`. The goal is to modify the deduplication logic (primarily within `ray_minhash.py`) so that instead of removing duplicates, it adds a column identifying which duplicate set each document belongs to.

**1. Modifying `BTSUnionFind` Actor:**

*   **Current Method:** `dup_idx(self, queries)`
    *   Identifies indices (`idx`) corresponding to UIDs (`uid`) that are *duplicates* (i.e., `uid in self.parent`, meaning `find(uid) != uid`).
    *   Returns a list of these indices.
*   **New Method:** Rename `dup_idx` to `get_root_ids(self, queries)` for clarity.
    *   **Input:** Same `queries` (list of `(uid, original_batch_index)` tuples).
    *   **Logic:**
        *   Initialize an empty list: `root_id_results = []`.
        *   Iterate through `uid, original_index` in `queries`.
        *   Call `root_id = self.find(uid)`. This works correctly for both duplicates (returns the root) and non-duplicates (returns the `uid` itself).
        *   Append the tuple `(original_index, root_id)` to `root_id_results`.
    *   **Return:** The `root_id_results` list.

```python
# Inside class BTSUnionFind in ray_minhash.py

# Replace dup_idx method with this:
def get_root_ids(self, queries):
    """
    For each queried UID, find its root in the Union-Find structure.

    Args:
        queries: A list of tuples, where each tuple is (uid, original_batch_index).

    Returns:
        A list of tuples, where each tuple is (original_batch_index, root_id).
    """
    root_id_results = []
    for uid, original_index in queries:
        root_id = self.find(uid) # Find the root ID
        root_id_results.append((original_index, root_id))
    return root_id_results

# Make sure to update any internal calls if dup_idx was used elsewhere, although it seems primarily called remotely.
```

**2. Modifying `RayBTSMinhashDeduplicator.filter_with_union_find`:**

*   **Current Logic:**
    *   Builds `query_dict` mapping `hash_id` (actor ID) to list of `(uid, idx)`.
    *   Calls `union_find_list[hash_id].dup_idx.remote(query)`.
    *   Collects results (lists of indices to remove).
    *   Builds a boolean `mask` to filter the batch.
    *   Returns `samples.select(columns_to_keep).filter(mask)`.
*   **New Logic:**
    *   Keep `query_dict` logic.
    *   Initialize `root_ids_array = np.full(len(samples), -1, dtype=np.int64)` (or use the actual UIDs as defaults if easily accessible, but -1 is clear). *Using `np.int64` assumes UIDs fit; adjust if necessary.*
    *   Call `self.union_find_list[hash_id].get_root_ids.remote(query)`.
    *   Collect results using `ray.wait` and `ray.get`. Each result `res` will be a list like `[(idx1, root1), (idx2, root2), ...]`.
    *   Process results: Iterate through `res` in `results`, and for each `original_index, root_id` pair in `res`, update the array: `root_ids_array[original_index] = root_id`.
    *   Verify all entries in `root_ids_array` were updated (i.e., no -1 left, unless an error occurred).
    *   Instead of filtering, append the new column: `return samples.append_column("duplicate_set_id", pa.array(root_ids_array))`

```python
# Inside class RayBTSMinhashDeduplicator in ray_minhash.py

def filter_with_union_find(self, samples: pa.Table) -> pa.Table:
    """
    Assigns a duplicate_set_id (the root ID from Union-Find) to each sample.
    """
    num_samples = len(samples)
    if num_samples == 0:
        # Handle empty batches: return the batch with an empty duplicate_set_id column
        return samples.append_column("duplicate_set_id", pa.array([], type=pa.int64()))

    query_dict = {}
    # Assuming 'uid' column exists from minhash_with_uid step
    for idx, uid_scalar in enumerate(samples["uid"]):
        uid = uid_scalar.as_py() # Get Python int from scalar
        # Determine which BTSUnionFind actor is responsible for this UID's primary hash space
        # Ensure BATCH_SIZE is defined appropriately or passed via config
        hash_id = uid // BATCH_SIZE % self.union_find_parallel_num
        if hash_id not in query_dict:
            query_dict[hash_id] = []
        query_dict[hash_id].append((uid, idx)) # Send (uid, original_batch_index)

    root_ids_array = np.full(num_samples, -1, dtype=np.int64) # Initialize with placeholder
    result_refs = []

    for hash_id, query in query_dict.items():
        if not query: continue # Skip empty queries
        if len(result_refs) > self.max_pending_filter_tasks: # Use configured limits
            num_returns = min(self.num_filter_task_returns, len(result_refs))
            ready_refs, result_refs = ray.wait(result_refs, num_returns=num_returns)
            results = ray.get(ready_refs)
            for result_list in results:
                for original_index, root_id in result_list:
                    if 0 <= original_index < num_samples: # Bounds check
                         root_ids_array[original_index] = root_id
                    else:
                         logger.warning(f"Received out-of-bounds index {original_index} for batch size {num_samples}")
            del ready_refs, results # Memory management

        # Call the new remote method
        result_refs.append(self.union_find_list[hash_id].get_root_ids.remote(query))

    # Process remaining refs
    results = ray.get(result_refs)
    for result_list in results:
        for original_index, root_id in result_list:
             if 0 <= original_index < num_samples: # Bounds check
                 root_ids_array[original_index] = root_id
             else:
                 logger.warning(f"Received out-of-bounds index {original_index} for batch size {num_samples}")
    del results, result_refs # Memory management

    # Sanity check: Ensure all placeholders are filled
    if np.any(root_ids_array == -1):
        unassigned_count = np.sum(root_ids_array == -1)
        logger.warning(f"Found {unassigned_count} samples missing a root_id assignment in a batch of size {num_samples}. Check BTSUnionFind logic or communication.")
        # Optional: Assign self-UID or handle error
        # for i in range(num_samples):
        #    if root_ids_array[i] == -1:
        #        root_ids_array[i] = samples["uid"][i].as_py() # Assign self as root

    # Append the new column instead of filtering
    # Ensure 'uid' column is not dropped if needed later
    final_table = samples.append_column("duplicate_set_id", pa.array(root_ids_array, type=pa.int64()))
    return final_table
```

**3. Calculating Duplicate Count:**

*   This now requires an extra step *after* the dataset has been processed by `filter_with_union_find`.
*   **Where to put this calculation?**
    *   Option A: Inside `RayBTSMinhashDeduplicator.run`: Modify it to perform the aggregation after the `map_batches` call to `filter_with_union_find`. This keeps the deduplication logic contained.
    *   Option B: In the calling functions (`run_nd_step_for_workflow`, `dedup`): Perform the aggregation there after receiving the tagged dataset. This makes the `RayBTSMinhashDeduplicator` purely responsible for tagging. (Slightly cleaner separation of concerns).

*   **Let's choose Option B for clarity.**

```python
# Inside run_nd_step_for_workflow in ray_minhash.py (and similarly in dedup)

def run_nd_step_for_workflow(ray_df, args):
    # ... (setup logger, get original_count, start_time) ...
    logger.info("Instantiating RayBTSMinhashDeduplicator...")
    deduplicator = RayBTSMinhashDeduplicator(
        text_key=args.column,
        # ... other params ...
    )
    logger.info("Running deduplicator.run to tag duplicates...")
    # This now returns the dataset with the 'duplicate_set_id' column
    tagged_dataset = deduplicator.run(ray_df) # Assume run calls map_batches(filter_with_union_find)

    # --- Calculate Duplicate Count Post-Hoc ---
    logger.info("Calculating duplicate count from duplicate_set_id...")
    calc_start_time = time.time()
    # Group by the new column and count sizes
    grouped = tagged_dataset.groupby("duplicate_set_id").count()
    # grouped is now a Dataset with columns: duplicate_set_id, count()

    # Count how many records belong to non-singleton sets
    # Method 1: Filter groups, then sum counts
    # duplicate_sets = grouped.filter(lambda row: row["count()"] > 1)
    # total_records_in_duplicates = duplicate_sets.sum(on="count()") # Might be 0 if no duplicates

    # Method 2: Easier - Count singletons and subtract from total
    original_count = tagged_dataset.count() # Get count *after* tagging, should be same as input count
    singleton_count = grouped.filter(lambda row: row["count()"] == 1).count()
    records_in_duplicate_sets = original_count - singleton_count

    # The number of duplicates REMOVED would be (records_in_duplicate_sets - number_of_duplicate_sets)
    # Let's report the number of *records* that are part of a duplicate set
    duplicate_record_count = records_in_duplicate_sets # Renaming for clarity
    # OR maybe more standard: number of items *removed* = original_count - final_count_if_dedupped
    # Final count if dedupped = singleton_count + number_of_duplicate_sets
    num_duplicate_sets = grouped.filter(lambda row: row["count()"] > 1).count()
    final_count_if_dedupped = singleton_count + num_duplicate_sets
    duplicate_count_to_log = original_count - final_count_if_dedupped


    logger.info(f"Duplicate count calculation took {time.time() - calc_start_time:.2f}s")
    # --- End Calculation ---

    total_time = time.time() - start_time # Recalculate total time
    logger.info(f"Total time taken (including tagging and count calc): {total_time:.2f} seconds")

    # Log the calculated count
    logger.info(f"Calculated duplicate count (items to remove): {duplicate_count_to_log}")
    logger.info(f"Number of non-singleton records: {duplicate_record_count}")
    logger.info(f"Final record count if deduplicated: {final_count_if_dedupped}")


    # Return the tagged dataset and the calculated duplicate count
    return tagged_dataset, duplicate_count_to_log, total_time # Returning the count of items that *would* be removed

# --- Similar modification needed in the `dedup` function ---
def dedup(ray_df, cfg):
    # ... (setup logger, original_count, start_time) ...
    deduplicator = RayBTSMinhashDeduplicator(...)
    tagged_dataset = deduplicator.run(ray_df).materialize() # Materialize maybe useful before aggregation

    # --- Calculate Duplicate Count Post-Hoc ---
    logger.info("Cluster deduplication: Calculating duplicate count from duplicate_set_id...")
    calc_start_time = time.time()
    original_count = tagged_dataset.count()
    if original_count == 0:
         logger.info("Cluster deduplication: Empty dataset, no duplicates.")
         return tagged_dataset, 0

    grouped = tagged_dataset.groupby("duplicate_set_id").count()
    singleton_count = grouped.filter(lambda row: row["count()"] == 1).count()
    num_duplicate_sets = grouped.filter(lambda row: row["count()"] > 1).count()
    final_count_if_dedupped = singleton_count + num_duplicate_sets
    duplicate_count = original_count - final_count_if_dedupped
    logger.info(f"Cluster deduplication calculation took {time.time() - calc_start_time:.2f}s")
    # --- End Calculation ---

    unique_count = final_count_if_dedupped # This is the count if we *had* deduplicated
    logger.info(f"Cluster deduplication: Original={original_count}, Calculated Unique Count={unique_count}, Calculated Duplicates Removed={duplicate_count}")

    # Return the tagged dataset and the calculated duplicate count
    return tagged_dataset, duplicate_count
```


in more details:

```
class BTSUnionFind:
    ...
    def get_root_ids(self, queries):
        """
        For each queried UID, find its root in the Union-Find structure.

        Args:
            queries: A list of tuples, where each tuple is (uid, original_batch_index).

        Returns:
            A list of tuples, where each tuple is (original_batch_index, root_id).
        """
        root_id_results = []
        for uid, original_index in queries:
            root_id = self.find(uid) # Find the root ID
            root_id_results.append((original_index, root_id))
        return root_id_results
    
    
class RayBTSMinhashDeduplicator:
    ...

    def merge_op_batch(self, object_refs):
        # Helper to wait for batches of remote operations
        results = []
        while object_refs:
            ready_refs, object_refs = ray.wait(object_refs,
                                               num_returns=min(
                                                   self.merge_batch_size,
                                                   len(object_refs)))
            results.extend(ray.get(ready_refs))
        return results

    def merge(self):
        self.merge_op_batch([
            union_find.edge_redistribution.remote()
            for union_find in self.union_find_list
        ])
        logger.info("Edge Redistribution complete.")

        iteration = 1
        while True:
            logger.info(f"Starting Balanced Union-Find Iteration {iteration}...")
            changed_flags = self.merge_op_batch([
                union_find.balanced_union_find.remote()
                for union_find in self.union_find_list
            ])
            logger.info(f"Balanced Union-Find Iteration {iteration} complete. Changed: {any(changed_flags)}")
            if not any(changed_flags):
                break # Converged

            logger.info(f"Starting Communication Iteration {iteration}...")
            self.merge_op_batch([
                union_find.communication.remote()
                for union_find in self.union_find_list
            ])
            logger.info(f"Communication Iteration {iteration} complete.")
            iteration += 1

        logger.info(f"Convergence reached after {iteration-1} iterations.")
        logger.info("Starting final Squeeze operation...")
        self.merge_op_batch([
            union_find.squeeze.remote() for union_find in self.union_find_list
        ])
        logger.info("Squeeze operation complete. Merge phase finished.")


    def filter_with_union_find(self, samples: pa.Table) -> pa.Table:
        """
        Assigns a duplicate_set_id (the root ID from Union-Find) to each sample.
        """
        num_samples = len(samples)
        if num_samples == 0:
            # Handle empty batches: return the batch with an empty duplicate_set_id column
            # Ensure the type matches the expected output type (e.g., pa.int64())
            return samples.append_column("duplicate_set_id", pa.array([], type=pa.int64()))

        query_dict = defaultdict(list) # Use defaultdict for cleaner code
        # Assuming 'uid' column exists from minhash_with_uid step
        uid_column = samples["uid"]
        for idx in range(num_samples):
            uid_scalar = uid_column[idx]
            # Check if scalar is valid before trying to access .as_py()
            if uid_scalar.is_valid:
                uid = uid_scalar.as_py() # Get Python int from scalar
                # Determine which BTSUnionFind actor is responsible for this UID's primary hash space
                # Ensure BATCH_SIZE is defined appropriately or passed via config
                if self.union_find_parallel_num > 0:
                    hash_id = uid // BATCH_SIZE % self.union_find_parallel_num
                else:
                    hash_id = 0 # Default to actor 0 if parallelism is 1 or less
                query_dict[hash_id].append((uid, idx)) # Send (uid, original_batch_index)
            else:
                logger.warning(f"Encountered invalid UID at index {idx} in batch, skipping.")


        root_ids_array = np.full(num_samples, -1, dtype=np.int64) # Initialize with placeholder
        result_refs = []

        for hash_id, query in query_dict.items():
            if not query: continue # Skip empty queries

            # Rate limiting
            if len(result_refs) >= self.max_pending_filter_tasks: # Use configured limits
                num_returns = min(self.num_filter_task_returns, len(result_refs))
                ready_refs, result_refs = ray.wait(result_refs, num_returns=num_returns)
                results = ray.get(ready_refs)
                for result_list in results:
                    if result_list: # Check if the result list is not empty
                        for original_index, root_id in result_list:
                            if 0 <= original_index < num_samples: # Bounds check
                                 root_ids_array[original_index] = root_id
                            else:
                                 # Use standard logger here as this runs in the map_batches worker
                                 std_logger.warning(f"Received out-of-bounds index {original_index} for batch size {num_samples}")
                del ready_refs, results # Memory management

            # Call the new remote method
            if 0 <= hash_id < len(self.union_find_list):
                result_refs.append(self.union_find_list[hash_id].get_root_ids.remote(query))
            else:
                std_logger.error(f"Query hash_id {hash_id} is out of bounds for union_find_list (size {len(self.union_find_list)}). Skipping query for this batch.")


        # Process remaining refs
        if result_refs:
            results = ray.get(result_refs)
            for result_list in results:
                 if result_list: # Check if the result list is not empty
                    for original_index, root_id in result_list:
                         if 0 <= original_index < num_samples: # Bounds check
                             root_ids_array[original_index] = root_id
                         else:
                             std_logger.warning(f"Received out-of-bounds index {original_index} for batch size {num_samples}")
            del results, result_refs # Memory management

        # Sanity check: Ensure all placeholders are filled
        unassigned_indices = np.where(root_ids_array == -1)[0]
        if len(unassigned_indices) > 0:
            unassigned_count = len(unassigned_indices)
            std_logger.warning(f"Found {unassigned_count} samples missing a root_id assignment in a batch of size {num_samples}. Check BTSUnionFind logic or communication. Assigning self-UID as root.")
            # Assign self-UID as root for unassigned entries
            uid_column = samples["uid"] # Re-access column if needed
            for i in unassigned_indices:
                 uid_scalar = uid_column[i]
                 if uid_scalar.is_valid:
                     root_ids_array[i] = uid_scalar.as_py()
                 else:
                      std_logger.error(f"Cannot assign self-UID for missing root_id at index {i} because original UID is invalid.")
                      # Assign a sentinel value or handle differently if needed
                      root_ids_array[i] = -2 # Example sentinel

        # Append the new column instead of filtering
        # Ensure 'uid' column is not dropped if needed later (it is dropped implicitly by select below)
        # Let's keep the uid column for potential debugging or future use
        final_table = samples.append_column("duplicate_set_id", pa.array(root_ids_array, type=pa.int64()))

        # Explicitly select columns to keep, including the new one and uid
        # columns_to_keep = samples.column_names + ["duplicate_set_id"] # Keep all original + new
        # return final_table.select(columns_to_keep)
        return final_table # Return table with new column appended


    def run(self, dataset, **kwargs): # kwargs seem unused here
        start_time = time.time()
        logger.info("Initializing IdGenerator...")
        # Ensure IdGenerator is initialized correctly
        # It's better to initialize it once outside the map_batches if possible,
        # but if state needs to be shared across batches, this approach works.
        # Let's keep it as is, assuming it handles state correctly.
        id_generator = IdGenerator.remote()
        logger.info("IdGenerator initialized.")

        def minhash_with_uid(table: pa.Table) -> pa.Table:
            """Adds a unique UID to each row and triggers MinHash calculation."""
            num_rows = len(table)
            if num_rows == 0:
                return table.append_column("uid", pa.array([], type=pa.int64())) # Handle empty table

            try:
                # Get a block of UIDs
                min_id_ref, max_id_ref = id_generator.get_next_id.remote(num_rows)
                min_id, max_id = ray.get([min_id_ref, max_id_ref])
                uid_list = list(range(min_id, max_id)) # Generate list of UIDs

                # Trigger MinHash calculation (modifies remote actors)
                self.calc_minhash(table[self.text_key], uid_list)

                # Append UID column
                new_table = table.append_column("uid", pa.array(uid_list, type=pa.int64()))
                return new_table
            except Exception as e:
                 std_logger.error(f"Error in minhash_with_uid batch: {e}", exc_info=True)
                 # Return original table or empty table with schema?
                 # Returning original table might lead to issues later if UID is expected.
                 # Let's return the table with an empty UID column of the correct type.
                 empty_uid_array = pa.array([None] * num_rows, type=pa.int64())
                 return table.append_column("uid", empty_uid_array)


        logger.info("Starting MinHash calculation with UID generation...")
        dataset_with_uid = dataset.map_batches(
            minhash_with_uid,
            batch_format='pyarrow',
            batch_size=self.hashing_batch_size, # Use specified batch size
            # zero_copy_batch=True, # May cause issues with modifications? Test carefully. Let's disable for safety.
            num_cpus=1, # Keep low unless minhash_with_uid is CPU intensive itself
            concurrency=self.union_find_parallel_num # Match concurrency with UF actors? Test this.
        ).materialize() # Materialize after adding UIDs and calculating hashes
        logger.info(f"MinHash calculation and UID generation complete. Time: {time.time() - start_time:.2f}s")


        logger.info("Starting Union-Find merge process...")
        merge_start_time = time.time()
        self.merge()
        logger.info(f"Union-Find merge process complete. Time: {time.time() - merge_start_time:.2f}s")

        logger.info("Starting duplicate set ID assignment...")
        tagging_start_time = time.time()
        # This step now adds the 'duplicate_set_id' column
        tagged_dataset = dataset_with_uid.map_batches(
            self.filter_with_union_find,
            batch_format='pyarrow',
            batch_size=self.hashing_batch_size, # Reuse batch size? Or define separate?
            # zero_copy_batch=True, # Disable for safety as we append a column
            concurrency=self.union_find_parallel_num # Match concurrency?
        )
        logger.info(f"Duplicate set ID assignment complete. Time: {time.time() - tagging_start_time:.2f}s")

        return tagged_dataset # Return the dataset with the added column


# --- Modified dedup function ---
def dedup(ray_df, cfg):
    logger = logging.getLogger(__name__) # Use standard logger

    original_count = ray_df.count()
    logger.info(f"Cluster deduplication: starting with {original_count} records")
    if original_count == 0:
         logger.info("Cluster deduplication: Empty dataset, skipping.")
         # Ensure schema includes uid and duplicate_set_id if they are expected downstream
         try:
            # Attempt to add columns with correct types if they don't exist
            if "uid" not in ray_df.schema().names:
                ray_df = ray_df.append_column("uid", pa.array([], type=pa.int64()))
            if "duplicate_set_id" not in ray_df.schema().names:
                ray_df = ray_df.append_column("duplicate_set_id", pa.array([], type=pa.int64()))
            return ray_df, 0
         except Exception as e:
            logger.error(f"Could not add required columns to empty dataset: {e}")
            return ray_df, 0 # Return original empty df

    start_time = time.time()

    # Instantiate deduplicator (ensure args are accessed correctly via cfg.args)
    try:
        deduplicator = RayBTSMinhashDeduplicator(
            text_key=cfg.args.column,
            ngram_size=cfg.args.ngram_size,
            min_ngram_size=cfg.args.min_ngram_size,
            num_permutations=cfg.args.num_perm,
            jaccard_threshold=cfg.args.threshold,
            # Use a reasonable default or make configurable for intra-cluster dedup
            # Look for specific intra-cluster params first, fallback to general or defaults
            union_find_parallel_num=getattr(cfg.args, 'union_find_parallel_num_intra_cluster', getattr(cfg.args, 'union_find_parallel_num', 'auto')),
            union_threshold=getattr(cfg.args, 'union_threshold_intra_cluster', getattr(cfg.args, 'union_threshold', 256)),
            max_pending_edge_buffer_task=getattr(cfg.args, 'max_pending_edge_buffer_task', 20),
            num_edge_buffer_task_returns=getattr(cfg.args, 'num_edge_buffer_task_returns', 10),
            max_pending_filter_tasks=getattr(cfg.args, 'max_pending_filter_tasks', 20),
            num_filter_task_returns=getattr(cfg.args, 'num_filter_task_returns', 10),
            merge_batch_size=getattr(cfg.args, 'merge_batch_size', 1000),
            hashing_batch_size=getattr(cfg.args, 'hashing_batch_size', 10000)
        )
    except AttributeError as e:
        logger.error(f"Missing required argument in cfg.args for RayBTSMinhashDeduplicator: {e}")
        raise

    # Run deduplication to get the tagged dataset
    logger.info("Running deduplicator.run to tag duplicates within cluster...")
    tagged_dataset = deduplicator.run(ray_df).materialize() # Materialize before aggregation
    logger.info(f"Tagging completed in {time.time() - start_time:.2f}s")

    # --- Calculate Duplicate Count Post-Hoc ---
    logger.info("Cluster deduplication: Calculating duplicate count from duplicate_set_id...")
    calc_start_time = time.time()
    # Check if 'duplicate_set_id' column exists
    if "duplicate_set_id" not in tagged_dataset.schema().names:
        logger.error("Cluster deduplication: 'duplicate_set_id' column not found after tagging.")
        # Return tagged dataset but indicate 0 duplicates found
        return tagged_dataset, 0

    try:
        # Need to handle potential empty dataset after tagging (though unlikely if input wasn't empty)
        current_count = tagged_dataset.count()
        if current_count == 0:
             logger.info("Cluster deduplication: Tagged dataset is empty, no duplicates.")
             return tagged_dataset, 0

        grouped = tagged_dataset.groupby("duplicate_set_id").count().materialize()
        # grouped is now a Dataset with columns: duplicate_set_id, count()

        # Count singletons and duplicate sets
        singleton_count = grouped.filter(lambda row: row["count()"] == 1).count()
        num_duplicate_sets = grouped.filter(lambda row: row["count()"] > 1).count()

        # Calculate final count if deduplicated (keep one from each set)
        final_count_if_dedupped = singleton_count + num_duplicate_sets
        duplicate_count = original_count - final_count_if_dedupped # Use original_count here

        logger.info(f"Cluster deduplication calculation took {time.time() - calc_start_time:.2f}s")
        logger.info(f"Cluster deduplication: Original(Cluster)={original_count}, Current Tagged={current_count}, Calculated Unique Count={final_count_if_dedupped}, Calculated Duplicates Removed={duplicate_count}")

    except Exception as e:
        logger.error(f"Error during duplicate count calculation: {e}", exc_info=True)
        # Return tagged dataset but 0 duplicates as calculation failed
        return tagged_dataset, 0

    # Return the tagged dataset and the calculated duplicate count
    return tagged_dataset, duplicate_count

# --- Modified run_nd_step_for_workflow function ---
def run_nd_step_for_workflow(ray_df, args):
    logger = logging.getLogger(__name__) # Use standard logger

    logger.info(f"Starting ND step with args: {args}")

    original_count = ray_df.count()
    logger.info(f"Original record count: {original_count}")
    if original_count == 0:
        logger.info("ND step: Empty input dataset, skipping.")
        # Return empty dataset with expected columns and 0 duplicates/time
        # Assuming ray_df has schema even if empty:
        try:
             empty_tagged = ray_df
             if "uid" not in empty_tagged.schema().names:
                 empty_tagged = empty_tagged.append_column("uid", pa.array([], type=pa.int64()))
             if "duplicate_set_id" not in empty_tagged.schema().names:
                 empty_tagged = empty_tagged.append_column("duplicate_set_id", pa.array([], type=pa.int64()))
        except Exception as e:
             # If append fails on empty, return original df
             logger.error(f"Could not add required columns to empty dataset: {e}")
             empty_tagged = ray_df
        return empty_tagged, 0, 0.0


    start_time = time.time()

    logger.info("Instantiating RayBTSMinhashDeduplicator...")
    # Instantiate deduplicator
    deduplicator = RayBTSMinhashDeduplicator(
        text_key=args.column,
        ngram_size=args.ngram_size,
        min_ngram_size=args.min_ngram_size,
        num_permutations=args.num_perm,
        jaccard_threshold=args.threshold,
        # Use provided args for parallelization parameters
        union_find_parallel_num=getattr(args, 'union_find_parallel_num', 'auto'), # Default if not provided
        union_threshold=getattr(args, 'union_threshold', 256),
        max_pending_edge_buffer_task=getattr(args, 'max_pending_edge_buffer_task', 20),
        num_edge_buffer_task_returns=getattr(args, 'num_edge_buffer_task_returns', 10),
        max_pending_filter_tasks=getattr(args, 'max_pending_filter_tasks', 20),
        num_filter_task_returns=getattr(args, 'num_filter_task_returns', 10),
        merge_batch_size=getattr(args, 'merge_batch_size', 1000),
        hashing_batch_size=getattr(args, 'hashing_batch_size', 10000)
    )
    logger.info("Running deduplicator.run to tag duplicates...")
    # This now returns the dataset with the 'duplicate_set_id' column
    tagged_dataset = deduplicator.run(ray_df).materialize() # Materialize before aggregation
    tagging_end_time = time.time()
    logger.info(f"Tagging step completed in {tagging_end_time - start_time:.2f} seconds")

    # --- Calculate Duplicate Count Post-Hoc ---
    logger.info("Calculating duplicate count from duplicate_set_id...")
    calc_start_time = time.time()
    duplicate_count_to_log = 0
    final_count_if_dedupped = 0

    # Check if 'duplicate_set_id' column exists
    if "duplicate_set_id" not in tagged_dataset.schema().names:
        logger.error("'duplicate_set_id' column not found after tagging. Cannot calculate duplicate count.")
        # Fallback: use original count and 0 duplicates
        final_count_if_dedupped = original_count
        duplicate_count_to_log = 0
    else:
        try:
            # Need to get the count *after* tagging, should be same as input count unless errors occurred
            current_tagged_count = tagged_dataset.count()
            if current_tagged_count == 0:
                logger.info("Tagged dataset is empty, no duplicates found.")
                final_count_if_dedupped = 0
                duplicate_count_to_log = 0
            else:
                # Group by the new column and count sizes
                grouped = tagged_dataset.groupby("duplicate_set_id").count().materialize()
                # grouped is now a Dataset with columns: duplicate_set_id, count()

                # Count singletons and duplicate sets
                singleton_count = grouped.filter(lambda row: row["count()"] == 1).count()
                num_duplicate_sets = grouped.filter(lambda row: row["count()"] > 1).count()

                # Calculate final count if deduplicated (keep one from each set)
                final_count_if_dedupped = singleton_count + num_duplicate_sets
                # The number of duplicates REMOVED = original_count - final_count_if_dedupped
                duplicate_count_to_log = original_count - final_count_if_dedupped

            logger.info(f"Duplicate count calculation took {time.time() - calc_start_time:.2f}s")

        except Exception as e:
            logger.error(f"Error during duplicate count calculation: {e}", exc_info=True)
            # Fallback if aggregation fails
            final_count_if_dedupped = tagged_dataset.count() # Count after tagging
            duplicate_count_to_log = 0 # Assume 0 duplicates found

    total_time = time.time() - start_time # Recalculate total time including aggregation
    logger.info(f"Total time taken for ND step (including tagging and count calc): {total_time:.2f} seconds")

    # Log the calculated count
    logger.info(f"Calculated duplicate count (items to remove): {duplicate_count_to_log}")
    logger.info(f"Final record count if deduplicated: {final_count_if_dedupped}")
    logger.info(f"Record count after tagging (should match original): {tagged_dataset.count()}")

    # Return the tagged dataset and the calculated duplicate count
    # Returning the count of items that *would* be removed
    return tagged_dataset, duplicate_count_to_log, total_time
```