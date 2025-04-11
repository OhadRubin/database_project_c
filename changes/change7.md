Okay, let's create a detailed plan for implementing Change #7: "Log Stage 2 Timing and Cluster Size Distribution".

The goal is to capture two specific pieces of information related to the clustering process, particularly the hierarchical (Stage 2) part, and store them in the database for analysis:

1.  **Stage 2 Execution Time:** The wall-clock time spent specifically within the `stage2` function in `ray_tfidf_vec.py`. This is most relevant for the `cl_nd` workflow where deduplication happens inside `stage2`.
2.  **Cluster Size Distribution:** The final count of records within each unique combination of Stage 1 and Stage 2 cluster IDs. This gives insight into the structure and balance of the resulting clusters.

This plan involves modifications in three key files:
*   `database_project/src/db.py`: To add new columns to the database schema.
*   `database_project/src/ray_tfidf_vec.py`: To measure the Stage 2 time and calculate the cluster distribution.
*   `database_project/src/run_workflows.py`: To pass the new metrics to the database logging function.

---

**Detailed Plan for Change #7**

**Phase 1: Update Database Schema (`database_project/src/db.py`)**

1.  **File:** `database_project/src/db.py`
    *   **Location:** Inside the `BenchmarkRun` class definition (around line 28).
    *   **Existing Code Context:**
        ```python
         27 |     total_size_gb = Column(Float, nullable=True)
         28 |     
         29 |     # Relationships
        ```
    *   **Change Description:**
        *   Add a new `Column` definition for storing the Stage 2 execution time. This should be a `Float` type and nullable, as it might not be applicable to all workflows or configurations (e.g., if only Stage 1 is run, or in ND->CL if Stage 2 timing isn't explicitly separated). Name it `cl_stage2_time_sec`.
        *   Add another new `Column` definition for storing the cluster size distribution. This should be a `Text` type (to accommodate a JSON string) and nullable. Name it `cluster_size_distribution_json`.

2.  **File:** `database_project/src/db.py`
    *   **Location:** Inside the `create_from_args` class method signature (around line 102).
    *   **Existing Code Context:**
        ```python
        101 |     @classmethod
        102 |     def create_from_args(cls, session, args, duplicate_count, record_count, execution_time, 
        103 |                        num_nodes=1, notes=None, implementation="pyspark", limit_files=None, total_size_gb=None):
        104 |         """
        ```
    *   **Change Description:**
        *   Add two new parameters to the `create_from_args` method signature: `cl_stage2_time_sec=None` and `cluster_size_distribution_json=None`. Set default values to `None`.

3.  **File:** `database_project/src/db.py`
    *   **Location:** Inside the `create_from_args` class method, within the `cls(...)` constructor call (around line 150).
    *   **Existing Code Context:**
        ```python
        148 |             limit_files=limit_files if limit_files is not None else args.limit_files,
        149 |             total_size_gb=total_size_gb
        150 |         )
        151 |         session.add(run)
        ```
    *   **Change Description:**
        *   Add assignments for the new columns within the `BenchmarkRun` object creation, mapping the new method parameters to the corresponding class attributes: `cl_stage2_time_sec=cl_stage2_time_sec`, `cluster_size_distribution_json=cluster_size_distribution_json`.

**Phase 2: Measure Timing and Calculate Distribution (`database_project/src/ray_tfidf_vec.py`)**

1.  **File:** `database_project/src/ray_tfidf_vec.py`
    *   **Location:** Within the `run_cl_step_for_workflow` function, around the loop that calls `stage1` and `stage2` (around line 564).
    *   **Existing Code Context:**
        ```python
        563 |     base_cfg.partition_cols = partition_cols
        564 |     for stage, func in zip(cfg.stages_list,[
        565 |         stage1, 
        566 |         # fake_stage1,
        567 |         stage2
        568 |         ]):
        569 |         stage_cfg = base_cfg.copy_and_resolve_references()
        570 |         stage_cfg.update(stage)
        571 |         stage_cfg.args = cfg.args
        572 |         print(stage_cfg)
        573 |         ds, workflow_duplicate_count = func(ds, stage_cfg) #stage1 returns 0
        574 |     
        ```
    *   **Change Description:**
        *   Modify the loop structure to handle the return values differently, specifically capturing the timing for `stage2`.
        *   Initialize `stage2_time = 0.0` before the loop.
        *   Inside the loop, check if the current function `func` is `stage2`.
        *   If it is `stage2`, record the start time using `time.time()` before calling `func(ds, stage_cfg)`.
        *   After the `stage2` call returns, record the end time and calculate the duration, storing it in `stage2_time`.
        *   Modify the function calls for `stage1` and `stage2` so they consistently return three values: the resulting dataset, the duplicate count for that stage, and the cluster distribution JSON (which will be calculated later). `stage1` should return `None` or an empty JSON string for the distribution. `stage2` will return the calculated distribution. For now, just adjust the unpacking logic: `ds, stage_duplicate_count, _ = func(ds, stage_cfg)` (the distribution will be handled in the next step). *Correction:* We only need the *final* distribution, not per-stage. And we only need the stage2 *time*. Let's simplify.
        *   **Revised Change:** Initialize `stage2_time = 0.0` before the loop. Inside the loop, *if `func` is `stage2`*, record start time, call `ds, stage_duplicate_count = func(ds, stage_cfg)`, record end time, calculate `stage2_time`. *Otherwise* (if `func` is `stage1`), just call `ds, stage_duplicate_count = func(ds, stage_cfg)`. The `workflow_duplicate_count` should accumulate or take the value from the relevant stage (likely only `stage2` returns a meaningful count in the `cl_nd` case). The `stage1` and `stage2` functions themselves currently return `(dataset, duplicate_count)`. This signature is okay for timing.

2.  **File:** `database_project/src/ray_tfidf_vec.py`
    *   **Location:** At the end of the `run_cl_step_for_workflow` function, before the final return statement (around line 579).
    *   **Existing Code Context:**
        ```python
        577 |     final_ds:ray.data.Dataset = final_ds.repartition(40)
        578 |     
        579 |     return final_ds.materialize(), workflow_duplicate_count
        580 | 
        ```
    *   **Change Description:**
        *   After the final dataset `final_ds` is materialized and repartitioned, calculate the cluster size distribution.
        *   Get the list of cluster column names (e.g., `cluster_cols = cfg.partition_cols`).
        *   Perform a `groupby` and `count` operation on the `final_ds` using Ray Data: `distribution_ds = final_ds.groupby(cluster_cols).count()`.
        *   Fetch the results: `distribution_list = distribution_ds.take_all()`. This will likely be a list of dictionaries, e.g., `[{'cluster_A': 0, 'cluster_B': 0, 'count': 123}, ...]`.
        *   Convert this list of dictionaries into a JSON string: `import json; cluster_distribution_json = json.dumps(distribution_list)`.
        *   Modify the `return` statement to include the measured `stage2_time` and the calculated `cluster_distribution_json`. The new signature should return: `(final_ds, workflow_duplicate_count, stage2_time, cluster_distribution_json)`.

**Phase 3: Pass Metrics to Database Logging (`database_project/src/run_workflows.py`)**

1.  **File:** `database_project/src/run_workflows.py`
    *   **Location:** Within the `nd_cl` workflow block, where `run_cl_step_for_workflow` is called (around line 155).
    *   **Existing Code Context:**
        ```python
        154 |             start_time = time.time()
        155 |             clustered_ds = run_cl_step_for_workflow(intermediate_ray_ds, cfg)
        156 |             cl_time = time.time() - start_time
        ```
    *   **Change Description:**
        *   Update the call to `run_cl_step_for_workflow` to unpack the four return values: `clustered_ds, _, stage2_time_ndcl, cluster_dist_json_ndcl = run_cl_step_for_workflow(intermediate_ray_ds, cfg)`. (Note: `workflow_duplicate_count` is already captured by the ND step earlier, so we can ignore the second return value here. Stage 2 time might be 0 if only one stage is run in `nd_cl`).

2.  **File:** `database_project/src/run_workflows.py`
    *   **Location:** Within the `cl_nd` workflow block, where `run_cl_step_for_workflow` is called (around line 167).
    *   **Existing Code Context:**
        ```python
        166 |             start_time = time.time()
        167 |             clustered_ds, cl_nd_duplicates = run_cl_step_for_workflow(ray_df, cfg)
        168 |             total_duplicate_count = cl_nd_duplicates # Assign the returned count
        ```
    *   **Change Description:**
        *   Update the call to `run_cl_step_for_workflow` to unpack the four return values: `clustered_ds, cl_nd_duplicates, stage2_time_clnd, cluster_dist_json_clnd = run_cl_step_for_workflow(ray_df, cfg)`. Update `total_duplicate_count` as before.

3.  **File:** `database_project/src/run_workflows.py`
    *   **Location:** Before the call to `BenchmarkRun.create_from_args` (around line 194).
    *   **Existing Code Context:**
        ```python
        191 |         engine = init_db()
        192 |         session = get_session(engine)
        193 |         
        194 |         benchmark_run = BenchmarkRun.create_from_args(
        ```
    *   **Change Description:**
        *   Determine which stage 2 time and cluster distribution JSON to log based on the executed workflow (`args.workflow`).
        *   Create variables `final_stage2_time` and `final_cluster_dist_json`.
        *   Use an if/else block based on `args.workflow` to assign the appropriate values captured in steps 3.1 and 3.2 (e.g., `final_stage2_time = stage2_time_clnd if args.workflow == 'cl_nd' else stage2_time_ndcl`).

4.  **File:** `database_project/src/run_workflows.py`
    *   **Location:** Inside the call to `BenchmarkRun.create_from_args` (around line 195-205).
    *   **Existing Code Context:**
        ```python
        195 |             session=session,
        196 |             args=args,
        197 |             duplicate_count=total_duplicate_count, # Meaning depends on workflow
        198 |             record_count=final_record_count,       # Final count after all steps
        199 |             execution_time=actual_workflow_time,   # Total wall clock time
        200 |             implementation=args.workflow,          # Use workflow name as implementation
        201 |             num_nodes=num_nodes_used,              # Max nodes used during the workflow
        202 |             notes=benchmark_notes,
        203 |             limit_files=args.limit_files,          # Log the limit used
        204 |             total_size_gb=0            # Log calculated size
        205 |         )
        ```
    *   **Change Description:**
        *   Add the new keyword arguments to the `create_from_args` call, passing the variables determined in step 3.3: `cl_stage2_time_sec=final_stage2_time`, `cluster_size_distribution_json=final_cluster_dist_json`.

---

This plan modifies the database schema, adds measurement/calculation logic in the clustering step, and updates the main workflow script to pass and log the new metrics, adhering to the principle of minimal code change while achieving the desired logging.