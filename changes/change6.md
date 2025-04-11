Okay, let's create a detailed plan for implementing Change #6: "Log Detailed Clustering Time Breakdown (Train vs. Inference)".

The goal is to separately measure and log the time spent *training* the clustering models (TF-IDF/SVD fitting, K-Means fitting) and the time spent *applying* these models (inference/prediction) across the dataset.

**Plan Overview:**

1.  **Modify Database Schema (`db.py`):** Add two new columns (`cl_train_time_sec`, `cl_inference_time_sec`) to the `BenchmarkRun` table. Update the creation function to accept and store values for these columns.
2.  **Modify Clustering Logic (`ray_tfidf_vec.py`):**
    *   Measure the time taken specifically for model fitting (vectorizer + K-Means) within the `_fit_models_remote` function.
    *   Measure the time taken for applying the models (inference) within the `fit_predict` function, primarily around the `map_batches` calls.
    *   Adjust the return values of `_fit_models_remote`, `fit_models_remote`, `fit_predict`, `stage1`, `stage2`, and `run_cl_step_for_workflow` to propagate these separate timings. Aggregate times correctly in `stage2` and `run_cl_step_for_workflow`.
3.  **Modify Workflow Orchestration (`run_workflows.py`):**
    *   Capture the separate training and inference times returned by `run_cl_step_for_workflow`.
    *   Pass these new timing values to the `BenchmarkRun.create_from_args` function when logging results.

---

**Detailed Plan:**

**Step 1: Modify Database Schema (`db.py`)**

1.  **File:** `database_project/src/db.py`
    *   **Line Numbers:** Around 28 (Add new columns), 102 (Modify function signature), 135 (Modify constructor call).
    *   **Existing Code (around line 27):**
        ```python
        27 |     total_size_gb = Column(Float, nullable=True)
        28 |     
        ```
    *   **Change Description:** Add two new `Float` columns to the `BenchmarkRun` class definition to store the clustering training and inference times. Make them nullable.

    *   **Existing Code (around line 102):**
        ```python
        101 |     @classmethod
        102 |     def create_from_args(cls, session, args, duplicate_count, record_count, execution_time, 
        103 |                        num_nodes=1, notes=None, implementation="pyspark", limit_files=None, total_size_gb=None):
        ```
    *   **Change Description:** Modify the `create_from_args` function signature to accept two new optional arguments: `cl_train_time_sec` and `cl_inference_time_sec`, defaulting to `None`.

    *   **Existing Code (around line 135):**
        ```python
        135 |         run = cls(
        136 |             input_file=args.input_file or args.table,
        137 |             output_dir=args.output,
        # ... existing arguments ...
        149 |             total_size_gb=total_size_gb
        150 |         )
        ```
    *   **Change Description:** Add the new `cl_train_time_sec` and `cl_inference_time_sec` parameters to the `BenchmarkRun` object instantiation within the `create_from_args` function.

**Step 2: Modify Clustering Logic (`ray_tfidf_vec.py`)**

1.  **File:** `database_project/src/ray_tfidf_vec.py`
    *   **Line Numbers:** Around 350 (Modify `_fit_models_remote`), 370 (Modify `fit_models_remote`), 407 (Modify `fit_predict`), 442 (Modify `stage1`), 456 (Modify `fit_predict_remote`), 460 (Modify `stage2`), 545 (Modify `run_cl_step_for_workflow`).

    *   **Function:** `_fit_models_remote`
    *   **Line Numbers:** Around 360 and 367.
    *   **Existing Code (around line 360-366):**
        ```python
        360 |     print(f"[{stage_label}] Fitting vectorizer on {len(texts)} samples...")
        361 |     vectorizer = get_sklearn_feature_pipeline(cfg.tfidf)
        362 |     embeddings = vectorizer.fit_transform(texts)
        363 |     print(f"[{stage_label}] Vectorizer fitting done. Embedding shape: {embeddings.shape}")
        364 |     print(f"[{stage_label}] Fitting K-means with {cfg.kmeans.n_clusters} clusters...")
        365 |     kmeans = fit_kmeans(embeddings,  cfg.kmeans)
        366 |     print(f"[{stage_label}] K-means fitting done.")
        367 |     return vectorizer, kmeans
        ```
    *   **Change Description:** Wrap the lines 361-366 (vectorizer fitting and K-Means fitting) with `start_time = time.time()` and `end_time = time.time()`. Calculate `train_time = end_time - start_time`. Change the return statement (line 367) to `return vectorizer, kmeans, train_time`.

    *   **Function:** `fit_models_remote`
    *   **Line Number:** 371
    *   **Existing Code:**
        ```python
        370 | @ray.remote
        371 | def fit_models_remote(cfg, ds):
        372 |     return _fit_models_remote(cfg, ds)
        ```
    *   **Change Description:** Ensure this function correctly returns the three values from `_fit_models_remote` (vectorizer, kmeans, train_time). The existing code already does this implicitly if `_fit_models_remote` is changed as described above.

    *   **Function:** `fit_predict`
    *   **Line Numbers:** Around 408, 418, 439, 440.
    *   **Existing Code (around line 408):**
        ```python
        408 |     models_s1_ref = fit_models_remote.options(
        # ... options ...
        413 |     )
        ```
    *   **Change Description:** Update the call site to expect three return values from the `fit_models_remote.remote` call's object reference. Store the training time reference. Retrieve the actual training time using `ray.get()` on the reference (e.g., `_, _, stage_train_time = ray.get(models_s1_ref)` after checking the models are needed, or adjust how references are handled).

    *   **Existing Code (around line 419-438):**
        ```python
        419 |     emb_tagged_ds_A = ds.map_batches(
        # ... TFIDFInferenceModel map_batches ...
        426 |     )
        427 |     # print(f"Schema after TFIDFInferenceModel:", emb_tagged_ds_A.schema())
        428 |     # print(f"Sample row after TFIDFInferenceModel:", emb_tagged_ds_A.take(1))
        429 |     tagged_ds_A = emb_tagged_ds_A.map_batches(
        # ... KMeansInferenceModel map_batches ...
        438 |     )
        439 | 
        440 |     return tagged_ds_A
        ```
    *   **Change Description:** Add `inf_start_time = time.time()` before line 419. Add `inf_end_time = time.time()` after line 438. Calculate `inference_time = inf_end_time - inf_start_time`. Change the return statement (line 440) to `return tagged_ds_A, stage_train_time, inference_time`.

    *   **Function:** `stage1`
    *   **Line Numbers:** Around 444, 448.
    *   **Existing Code:**
        ```python
        443 |     start_time = time.time()
        444 |     tagged_ds_A = fit_predict(ds, cfg).materialize()
        # ... timing print ...
        448 |     return tagged_ds_A, 0
        ```
    *   **Change Description:** Update line 444 to capture the three return values from `fit_predict`: `tagged_ds_A, stage1_train_time, stage1_inference_time = fit_predict(ds, cfg)`. Materialize `tagged_ds_A` separately if needed. Change the return statement (line 448) to `return tagged_ds_A, 0, stage1_train_time, stage1_inference_time`.

    *   **Function:** `fit_predict_remote`
    *   **Line Number:** 458
    *   **Existing Code:**
        ```python
        457 | @ray.remote
        458 | def fit_predict_remote(ds: ray.data.Dataset, cfg):
        459 |     return fit_predict(ds.materialize(), cfg).materialize()
        ```
    *   **Change Description:** Update line 459 to capture and return the three values from `fit_predict`: `clustered_ds, train_time, inference_time = fit_predict(ds.materialize(), cfg)`. Return `clustered_ds.materialize(), train_time, inference_time`.

    *   **Function:** `stage2`
    *   **Line Numbers:** Around 470-487, 506.
    *   **Existing Code (around line 472):**
        ```python
        472 |         s2_clustered_ds_ref = fit_predict_remote.remote(ds_cluster_data, cfg)
        # ... conditional dedup ...
        481 |             processed_refs.append((s2_clustered_ds_ref, ray.put(0))) # Use ray.put(0)
        # ... retrieve results ...
        486 |     results_list = ray.get([ref_pair[0] for ref_pair in processed_refs]) # Get datasets
        487 |     count_results = ray.get([ref_pair[1] for ref_pair in processed_refs]) # Get counts (now always has a value)
        # ... aggregate ...
        506 |     return final_ds.materialize(), total_cluster_duplicates
        ```
    *   **Change Description:**
        *   Modify line 472 to capture the three object refs: `s2_clustered_ds_ref, train_time_ref, inf_time_ref = fit_predict_remote.remote(...)`.
        *   Adjust the `processed_refs.append(...)` calls (lines 478 and 481) to store tuples like `(final_ds_ref, dupe_count_ref, train_time_ref, inf_time_ref)`. For the non-dedup case (line 481), it would be `(s2_clustered_ds_ref, ray.put(0), train_time_ref, inf_time_ref)`.
        *   After line 487, add code to retrieve and aggregate the times:
            *   `train_times = ray.get([ref_pair[2] for ref_pair in processed_refs])`
            *   `inf_times = ray.get([ref_pair[3] for ref_pair in processed_refs])`
            *   `total_stage2_train_time = sum(train_times)`
            *   `total_stage2_inference_time = sum(inf_times)`
        *   Change the return statement (line 506) to `return final_ds.materialize(), total_cluster_duplicates, total_stage2_train_time, total_stage2_inference_time`.

    *   **Function:** `run_cl_step_for_workflow`
    *   **Line Numbers:** Around 549, 564-573, 579.
    *   **Existing Code (around line 549):**
        ```python
        549 |     workflow_duplicate_count = 0 # Initialize count
        ```
    *   **Change Description:** Initialize total train and inference time accumulators after line 549: `total_train_time = 0.0`, `total_inference_time = 0.0`.

    *   **Existing Code (around line 573):**
        ```python
        564 |     for stage, func in zip(cfg.stages_list,[
        # ... stages ...
        568 |         ]):
        # ... config setup ...
        573 |         ds, workflow_duplicate_count = func(ds, stage_cfg) #stage1 returns 0
        ```
    *   **Change Description:** Modify line 573 to expect four return values: `ds, stage_duplicates, stage_train_time, stage_inference_time = func(ds, stage_cfg)`. Add lines immediately after to accumulate times: `total_train_time += stage_train_time`, `total_inference_time += stage_inference_time`. Update the `workflow_duplicate_count` appropriately (it should only be non-zero from `stage2` if deduping): `workflow_duplicate_count += stage_duplicates`.

    *   **Existing Code (around line 579):**
        ```python
        579 |     return final_ds.materialize(), workflow_duplicate_count
        ```
    *   **Change Description:** Change the return statement (line 579) to `return final_ds.materialize(), workflow_duplicate_count, total_train_time, total_inference_time`.

**Step 3: Modify Workflow Orchestration (`run_workflows.py`)**

1.  **File:** `database_project/src/run_workflows.py`
    *   **Line Numbers:** Around 112 (Initialize variables), 155 (ND->CL call), 167 (CL->ND call), 194 (DB logging call).

    *   **Change Description:** Near the start of the `if __name__ == "__main__":` block (e.g., after line 117), initialize variables to hold the timings: `cl_train_time = 0.0`, `cl_inference_time = 0.0`.

    *   **Workflow:** `nd_cl`
    *   **Line Number:** Around 155.
    *   **Existing Code:**
        ```python
        155 |             clustered_ds = run_cl_step_for_workflow(intermediate_ray_ds, cfg)
        ```
    *   **Change Description:** Update the call to expect four return values: `clustered_ds, _, cl_train_time, cl_inference_time = run_cl_step_for_workflow(intermediate_ray_ds, cfg)`. (The `_` ignores the duplicate count which should be 0 here).

    *   **Workflow:** `cl_nd`
    *   **Line Number:** Around 167.
    *   **Existing Code:**
        ```python
        167 |             clustered_ds, cl_nd_duplicates = run_cl_step_for_workflow(ray_df, cfg)
        ```
    *   **Change Description:** Update the call to expect four return values: `clustered_ds, cl_nd_duplicates, cl_train_time, cl_inference_time = run_cl_step_for_workflow(ray_df, cfg)`.

    *   **Database Logging:**
    *   **Line Number:** Around 194.
    *   **Existing Code:**
        ```python
        194 |         benchmark_run = BenchmarkRun.create_from_args(
        195 |             session=session,
        196 |             args=args,
        # ... existing args ...
        205 |         )
        ```
    *   **Change Description:** Add the new timing arguments to the `BenchmarkRun.create_from_args` call:
        ```python
        # Inside the create_from_args call, add:
                     cl_train_time_sec=cl_train_time,
                     cl_inference_time_sec=cl_inference_time,
        # Make sure they are placed correctly within the arguments list
        ```

---

**Summary of Changes:**

This plan adds two new timing metrics to the database and modifies the clustering code (`ray_tfidf_vec.py`) to measure and propagate these times through its internal functions (`_fit_models_remote`, `fit_predict`, `stage1`, `stage2`, `run_cl_step_for_workflow`). The main workflow script (`run_workflows.py`) is updated to receive these times and log them. The changes focus on adding timing logic and modifying function signatures/calls, aiming for minimal structural alteration to the existing code. The inference time measurement is pragmatic, timing the relevant section containing `map_batches` calls rather than attempting complex per-batch timing extraction.