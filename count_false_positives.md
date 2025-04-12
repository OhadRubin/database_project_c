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