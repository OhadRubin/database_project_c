Suggested Changes:
-------------------------------------------------------------------

1. **Suggested change: Log ND Step Execution Time**
 
*   **Goal:** Quantify the cost of the initial Near-Duplicate Detection step.
*   **Insight:** Adding a dedicated `nd_time_sec` column to `BenchmarkRun` and logging the time returned by `run_nd_step_for_workflow` provides a direct, persistent measure of the ND component's cost. This allows you to analyze how ND parameters (threshold, num_perm) affect its runtime and isolates its contribution to the total time in the ND->CL workflow. Comparing this value across runs helps understand the performance trade-offs of performing ND first.
 
2. **Suggested change: Log Clustering Configuration File Path**
 
*   **Goal:** Improve reproducibility by tracking the specific clustering config used.
*   **Insight:** Storing the exact path (`args.config_file`) to the YAML configuration file used for clustering (`config_file_path` column in `BenchmarkRun`) directly links a specific run's results in the database to the parameters defined in that file. This is crucial for reproducibility and debugging, ensuring you know which set of clustering parameters (batch sizes, dimensions, cluster counts, resource hints) generated the observed performance and output characteristics.

3. **Suggested change: Log Intermediate Record Count after ND Step**

*   **Goal:** Measure the data reduction achieved by the ND step in the ND->CL workflow.
*   **Insight:** Logging the record count *after* the ND step completes (`nd_output_count` column in `BenchmarkRun`) quantifies the effectiveness of the deduplication stage. You can see precisely how many records were removed and, therefore, the size of the dataset being fed into the subsequent clustering step. This metric is vital for understanding the performance impact of ND->CL – if ND removes many records, the subsequent CL step should theoretically be faster.

4. **Suggested change: Log Full Configuration Details (Args + YAML)**

*   **Goal:** Capture the complete set of parameters used for a run for maximum reproducibility.
*   **Insight:** Storing a JSON serialization of *all* relevant configurations – both command-line arguments (`args`) and the loaded clustering configuration (`cfg`) – into a `config_details_json` text column in `BenchmarkRun` provides the ultimate record for reproducibility. This eliminates ambiguity about default values or slight variations in config files, ensuring you can precisely reconstruct the conditions under which a specific result was obtained.

5. **Suggested change: Log Detailed Clustering Time Breakdown (Train vs. Inference)**

*   **Goal:** Understand bottlenecks within the clustering process itself.
*   **Insight:** Adding separate columns like `cl_train_time_sec` and `cl_inference_time_sec` to `BenchmarkRun` provides critical insight into the clustering step's internal behavior. Model training (TF-IDF/SVD fitting, K-Means fitting) often happens once or on a sample and might scale differently than inference (applying TF-IDF/SVD, predicting clusters), which runs on the entire dataset. Knowing this breakdown helps identify whether the bottleneck is in learning the model structure or applying it, guiding optimization efforts (e.g., faster inference algorithms, more efficient training).

6. **Suggested change: Log Stage 2 Timing and Cluster Size Distribution**

*   **Goal:** Analyze the performance of the hierarchical step and the structure of the final clusters.
*   **Insight:** Logging the specific time spent in the `stage2` function (`cl_stage2_time_sec`) isolates the cost of the per-cluster processing loop, which is particularly relevant for the CL->ND workflow (where ND happens inside stage 2). Additionally, logging the final distribution of records across clusters (`cluster_size_distribution_json`) reveals the outcome of the multi-stage process. It shows how balanced the clusters are, identifies potentially tiny or huge clusters, and gives a quantitative view of the final data organization achieved by the chosen clustering parameters.


**Implementation details:**

**1. `/database_project/src/db.py` (Benchmarking Schema & Logging)**

*   **Expanded `BenchmarkRun` Schema:** This is the most significant change. Several **new columns** were added to capture more granular performance and configuration details:
    *   `nd_time_sec`: Specific timing for the Near-Duplicate (ND) step.
    *   `nd_output_count`: Record count *after* the ND step (useful for ND->CL).
    *   `config_file_path`: Explicitly logs the clustering config file used.
    *   `cl_train_time_sec`, `cl_inference_time_sec`: Breakdown of Clustering (CL) step timing.
    *   `cl_stage2_time_sec`: Specific timing for the complex Stage 2 logic (CL within CL).
    *   `config_details_json`: Stores the *entire configuration* (args + loaded YAML) as JSON, allowing exact reproducibility checks.
    *   `cluster_size_distribution_json`: Stores the final count of records per cluster as JSON, enabling analysis of cluster balance.
*   **Updated `create_from_args`:** Modified to accept and store values for *all* the new `BenchmarkRun` columns listed above.
*   **Decoupled Metric Commits:** `add_resource_metrics` and `add_accuracy_metrics` **no longer commit internally**. The caller (in `run_workflows.py`) is now responsible for committing after adding metrics, allowing the main run and its associated metrics potentially be committed more atomically.
*   **Schema Fix:** Corrected typo `resulte_mb` to `disk_write_mb` in `ResourceMetric`.


**2. `/database_project/src/ray_tfidf_vec.py` (CL Engine & Workflow)**

*   **Decoupled Model Fitting:** Fitting TF-IDF/SVD and KMeans (`_fit_models_remote`) is now a separate remote task launched by `fit_models_remote`. Crucially, it now **returns the training time** along with the models as separate `ObjectRefs`.
*   **Timing Breakdown in `fit_predict`:** This function now explicitly times and returns separate `train_time` and `inference_time`.
*   **Enhanced `run_cl_step_for_workflow`:**
    *   Returns **aggregated timings** (`total_train_time`, `total_inference_time`) and specific `stage2_time`.
    *   Calculates and returns the **final cluster size distribution** as a JSON string.
    *   Dynamically executes stages based on the config file structure.

**3. `/database_project/src/run_workflows.py` (Orchestration & Benchmarking)**
*   **Capture Detailed Metrics:** The main script now **captures the new detailed timings** (`nd_step_time`, `cl_train_time`, `cl_inference_time`, `cl_stage2_time`)

