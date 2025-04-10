Okay, I understand. The previous writeup was structured like a detailed research paper based on the `toc.md`. Let's condense and refocus it to directly address the points (a) through (f) outlined in `instructions.md`, emphasizing the integration of course topics and the specific implementation.

---

# Final Project Report: Comparing Scalable Near-Duplicate Detection and Clustering Workflows

**Course:** Advanced Topics in Data Management

**(Adapted to `instructions.md` requirements)**

---

**(a) General Description of the Project**

*   **Objectives:** This project investigates the optimal ordering of Near-Duplicate Detection (NDD) and Clustering (CL) operations for large-scale text datasets. The primary goal is to compare the performance and output characteristics of two workflow approaches:
    1.  **ND->CL:** Performing NDD first, followed by CL on the deduplicated data.
    2.  **CL->ND:** Performing CL first, followed by NDD *within* each resulting cluster.
*   **High-Level Solution Overview:** We integrate algorithms related to two key course topics: **Approximate Similarity Search (MinHash/LSH)** and **Large-Scale Data Processing (MapReduce principles via Ray)**. We implemented:
    *   A scalable NDD engine using MinHash/LSH with a custom distributed Union-Find algorithm on Ray.
    *   A multi-stage CL engine using TF-IDF, SVD, and a custom K-Means implementation accelerated with JAX, distributed using Ray Data and Actors.
    *   An orchestrator (`run_workflows.py`) to execute both the ND->CL and CL->ND workflows on a slice of the C4 dataset using a multi-node Ray cluster (running on TPUs).
    *   A benchmarking system (`db.py`) using SQLAlchemy to log run parameters, execution times, and output statistics (record/duplicate counts).

**(b) Detailed Explanations of Algorithms, Methods, and Integration**

*   **Course Topics Integrated:**
    1.  **Approximate Similarity Search (Syllabus Week 13: LSH/MinHash):** We implemented MinHash signature generation and Locality Sensitive Hashing (LSH) based on the principles discussed (Broder, 1997).
    2.  **Large-Scale Data Processing (Syllabus Weeks 10-12: MapReduce, Spark, Big Data Frameworks):** We leveraged the Ray framework, which provides primitives analogous to MapReduce/Spark (distributed datasets, tasks, actors), to parallelize both the NDD and CL algorithms across a cluster.

*   **Algorithms Implemented (Core Contributions):**
    1.  **Distributed BTS Union-Find for NDD (`ray_minhash.py`):**
        *   **Algorithm:** Implements the core logic of MinHash signature calculation and LSH banding to find candidate duplicate pairs. The implementation is adapted from data-juicer's MinHash implementation. My specific contribution was adapting this distributed algorithm into a single, self-contained file that could be easily integrated with our workflow system.
        *   **Implementation:** Uses Ray actors (`IdGenerator`, `EdgeBuffer`, `BTSUnionFind`) to distribute Union-Find operations across nodes. Each `BTSUnionFind` actor manages a subset of IDs, performs local unions, communicates via EdgeBuffers, and participates in iterative synchronization until convergence. Implements path compression and union-by-size heuristics for efficiency.
    2.  **Distributed Multi-Stage Clustering (`ray_tfidf_vec.py`):**
        *   **Algorithm:** Implements a two-stage hierarchical clustering process. Stage 1 clusters the entire dataset (or a sample); Stage 2 clusters the data *within* each Stage 1 cluster. It uses standard TF-IDF vectorization (with number normalization), Truncated SVD for dimensionality reduction, and K-Means.
        *   **Implementation:** While using `scikit-learn` for the TF-IDF/SVD pipeline, the implementation features:
            *   A **custom K-Means class** supporting **online updates** (fitting iteratively on batches) and optional **balanced clustering** using the Auction Algorithm (`auction_lap`).
            *   **JAX acceleration** for K-Means distance calculations (`create_jax_pairwise_distance`, `compile_nearest_cluster`), enabling efficient use of TPUs/GPUs.
            *   **Distribution via Ray:** Model fitting (`fit_models_remote`) is done in a dedicated Ray actor (potentially on a sample). Inference (`TFIDFInferenceModel`, `KMeansInferenceModel`) is parallelized across the dataset using Ray Data's `map_batches` and dedicated actors, leveraging specified CPU/TPU resources. Stage 2 processing (fitting, predicting, and optionally deduplicating per-cluster) is parallelized by launching remote Ray tasks for each Stage 1 cluster.

*   **Integration:**
    *   The `run_workflows.py` script acts as the orchestrator.
    *   It loads data into a `Ray Dataset`.
    *   For **ND->CL:** It calls the NDD engine (`run_nd_step_for_workflow`), which returns a deduplicated `Ray Dataset`. This dataset is then passed to the CL engine (`run_cl_step_for_workflow` with `should_dedup=False`).
    *   For **CL->ND:** It calls the CL engine (`run_cl_step_for_workflow` with `should_dedup=True`). Inside the CL engine's `stage2` function, after performing clustering for a given Stage 1 cluster, it calls the NDD engine's deduplication function (`dedup_remote`) on that cluster's `Ray Dataset`. The results from all clusters are then unioned.
    *   Data flows between stages primarily as `Ray Datasets`. Configuration parameters (like NDD settings) are passed via the config object (`cfg`).

**(c) Instructions on How to Run the Project**

1.  **Prerequisites:** Python 3.10, `git`, access to a cluster environment (designed for Google Cloud TPUs but adaptable), potentially `gcloud` CLI for cluster setup, SSH access between nodes.
2.  **Setup Cluster Nodes (Manual or via `run.sh` logic):**
    *   Ensure all nodes have Python 3.10 and required dependencies (Install `ray==2.43.0`, `numpy~=1.0`, `scikit-learn`, `jax`, `sqlalchemy`, `psycopg2`, `pandas`, etc. using `pip`).
    *   Clone the repository: `git clone https://github.com/OhadRubin/database_project_c && cd database_project_c` (or pull updates).
    *   Set Environment Variables: Export `POSTGRES_ADDRESS="postgresql+psycopg2://user:pass@host:port/db"` for benchmark logging.
    *   Download Data: Run `python3.10 database_project/src/download_c4.py`. Ensure 40 files are downloaded (e.g., to `/dev/shm/c4_files`).
    *   Setup GCS Fuse (Optional but used in `run.sh`): Mount GCS bucket if needed, e.g., `gcsfuse meliad2_us2_backup /mnt/gcs_bucket ...`
    *   Start Ray Cluster: On the designated head node run `ray start --head --resources='{"TPU-v4-8-head": 1}' ...`. On worker nodes run `ray start --address="<head_ip>:6379" --resources='{"TPU-v4-8-head": 1}' ...`. Wait for all nodes to join (check with `ray status` or the script's loop).
3.  **Execute Workflow (on Head Node):**
    *   Choose the workflow (`nd_cl` or `cl_nd`).
    *   Run the main script, adjusting parameters as needed:
        ```bash
        # Example for CL->ND (as in run.sh)
        python3.10 database_project/src/run_workflows.py \
            --workflow cl_nd \
            --input_file "/dev/shm/c4_files/c4-train.*.json.gz" \
            --output /path/to/desired/output_directory \
            --config_file database_project/src/configs/base.yml \
            --limit_files 40 \
            --notes "CL->ND run on 40 C4 files" \
            # Add NDD params if defaults are not desired: --threshold 0.7 --num_perm 256 etc.

        # Example for ND->CL
        python3.10 database_project/src/run_workflows.py \
            --workflow nd_cl \
            # ... other parameters as above ...
            --notes "ND->CL run on 40 C4 files"
        ```
    *   The script will execute the chosen workflow and log results to the configured database.

**(d) Example Runs**

The following commands were used for the experiments processing 40 C4 files (~11.88 GB) reported in the analysis:

*   **CL->ND Workflow Run:**
    ```bash
    python3.10 database_project/src/run_workflows.py --workflow cl_nd --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --use_ray True --limit_files 40
    ```
    *(Corresponds approximately to Run ID 16 in the sample `viewer.ipynb` data, though parameters might differ slightly from default)*

*   **ND->CL Workflow Run:**
    ```bash
    python3.10 database_project/src/run_workflows.py --workflow nd_cl --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --use_ray True --limit_files 40
    ```
    *(Corresponds approximately to Run ID 15 in the sample `viewer.ipynb` data, though parameters might differ slightly from default)*

**(e) Code Documentation**

*   **Structure:** The code is organized within the `database_project/src` directory.
    *   `run_workflows.py`: Main orchestrator script.
    *   `ray_minhash.py`: Contains the NDD engine implementation (MinHash, LSH, BTS Union-Find Actors).
    *   `ray_tfidf_vec.py`: Contains the CL engine implementation (TF-IDF/SVD, custom K-Means, Ray actors for fitting/inference, multi-stage logic).
    *   `db.py`: Defines the SQLAlchemy database schema and helper functions for logging benchmark results.
    *   `configs/base.yml`: Configuration file for clustering parameters (batch sizes, dimensions, cluster counts, resource allocation hints).
    *   `download_c4.py`: Script to download the dataset.
*   **Documentation:** Code includes inline comments explaining key sections. Function and class docstrings provide high-level descriptions. The `db.py` module clearly defines the database schema used for logging experimental results. The `instructions.md` file (this context) and `README.md` files provide setup and usage guidance.

**(f) Details of Analyses Performed and Results**

*   **Analysis Goal:** To compare the ND->CL and CL->ND workflows in terms of performance (end-to-end execution time) and output characteristics (number of duplicates identified, final record count) on a ~12GB slice (40 files) of the C4 dataset using a 10-node Ray cluster.
*   **Metrics Collected (Logged via `db.py`):**
    *   `execution_time`: Total wall-clock time for the workflow.
    *   `duplicate_count`: Total number of duplicates identified (interpretation differs by workflow).
    *   `record_count`: Number of records in the final dataset.
    *   Input parameters (`threshold`, `num_perm`, `limit_files`, etc.).
*   **Results (Based on sample data in `viewer.ipynb` - Run ID 15 for ND->CL, Run ID 16 for CL->ND, using 40 files on 10 nodes):**

    | Workflow | Execution Time (s) | Duplicate Count | Final Record Count | Input Records | Retention | Notes                                   |
    | :------- | :----------------- | :-------------- | :----------------- | :------------ | :-------- | :-------------------------------------- |
    | ND->CL   | X               | X        | X        | X        | ~X%    | Global deduplication                 |
    | CL->ND   | X               | X           | X        | X        | ~X% | Intra-cluster dedup |

    

*   **Discussion & Conclusions:**
    *   **Performance:** The X workflow appears significantly faster  in these sample runs. This suggests that for this dataset size and configuration, the cost of global NDD followed by CL was less/higher than the cost of CL followed by parallel intra-cluster NDD. The overhead of spawning many NDD tasks within `stage2` might contribute to CL->ND's longer runtime.
    *   **Output:** X achieves global deduplication, resulting in a smaller final dataset (higher retention means fewer duplicates removed). X only guarantees uniqueness *within* clusters, leading to a larger final dataset as inter-cluster duplicates are missed. The choice depends on the application's requirement for global vs. local uniqueness.
    *   **Trade-offs:** X provides stronger deduplication guarantees potentially faster. X might be considered if NDD is extremely expensive and most duplicates are expected within clusters, *and* global uniqueness is not strictly required, but our results didn't show a performance benefit here.
    *   **Recommendation:** Based on this experiment, **X is recommended** as it was faster and provides global deduplication. 


Additional list of potential experiments, categorized by the aspect they investigate:

**1. Scalability Analysis:**

*   **Experiment 1.1: Data Size Scaling:**
    *   **Objective:** Evaluate how each workflow's performance scales as the input data volume increases.
    *   **Procedure:** Run both ND->CL and CL->ND workflows on increasing subsets of the C4 dataset (e.g., using `limit_files` = 10, 20, 40, 80, 160 files, if feasible). Keep cluster size (e.g., 10 nodes) and algorithm parameters constant.
    *   **Metrics:** Measure end-to-end execution time, final record count, and total duplicate count for each run.
    *   **Analysis:** Plot Time vs. Data Size (GB or # Documents) for both workflows. Does one workflow exhibit better linearity or a lower slope? Does the relative performance difference change significantly with size? Analyze throughput (records/sec).

*   **Experiment 1.2: Cluster Size Scaling (Weak Scaling):**
    *   **Objective:** Evaluate how each workflow benefits from adding more computational resources while keeping the *per-node* workload roughly constant (or analyzing strong scaling with fixed total data size).
    *   **Procedure:** Fix the total data size (e.g., 40 or 80 files). Run both workflows on varying numbers of cluster nodes (e.g., 2, 5, 10 nodes). Adjust data partitioning/parallelism settings potentially.
    *   **Metrics:** Measure end-to-end execution time, final record count, duplicate count.
    *   **Analysis:** Plot Time vs. Number of Nodes for both workflows. Calculate speedup. Does one workflow parallelize more effectively? Does CL->ND's nested parallelism benefit more or less than ND->CL's sequential parallelism?

**2. Sensitivity to NDD Parameters:**

*   **Experiment 2.1: Varying Similarity Threshold (`threshold`):**
    *   **Objective:** Understand how the NDD strictness affects the performance and output of both workflows.
    *   **Procedure:** Fix data size and cluster size. Run both workflows using different Jaccard similarity thresholds for the NDD step (e.g., 0.6, 0.7, 0.8, 0.9).
    *   **Metrics:** Execution time, duplicate count, final record count.
    *   **Analysis:** How does changing the threshold affect the *relative* time difference between workflows? Does a higher threshold (fewer candidates) make CL->ND comparatively faster? How significantly does the final record count differ between workflows at different thresholds?

*   **Experiment 2.2: Varying Number of Permutations (`num_perm`):**
    *   **Objective:** Assess the impact of MinHash signature fidelity on both workflows.
    *   **Procedure:** Fix data size, cluster size, and threshold. Run both workflows using different numbers of permutations (e.g., 128, 256, 512).
    *   **Metrics:** Execution time, duplicate count, final record count.
    *   **Analysis:** How does `num_perm` affect performance (potentially increasing hashing time but maybe reducing candidate pairs) and output counts for each workflow?

**3. Sensitivity to CL Parameters (Focus on Impact on CL->ND):**

*   **Experiment 3.1: Varying Stage 1 Cluster Count (`k1`):**
    *   **Objective:** Investigate how the initial clustering granularity impacts the CL->ND workflow's effectiveness and performance.
    *   **Procedure:** Fix data size, cluster size, and NDD parameters. Run both workflows, but primarily analyze the CL->ND results while varying the number of Stage 1 clusters (e.g., `k1` = 5, 10, 20, 50). The ND->CL workflow's CL step will also be affected, providing a baseline CL time comparison.
    *   **Metrics:** Execution time, duplicate count (especially for CL->ND), final record count, potentially cluster size distribution.
    *   **Analysis:** Does increasing `k1` in CL->ND significantly decrease the *intra-cluster* duplicate count found (as duplicates might be split across more clusters)? How does `k1` affect the total runtime of CL->ND (trade-off between CL time and parallel NDD time)?

**4. Output Quality Analysis (Requires Sampling/Ground Truth):**

*   **Experiment 4.1: Duplicate Pair Accuracy (Requires Labeled Data):**
    *   **Objective:** Go beyond simple counts and evaluate the *correctness* of the identified duplicates.
    *   **Procedure:** Create or obtain a subset of the data with known ground-truth duplicate pairs. Run both workflows on this subset. Compare the set of duplicate *pairs* identified by each workflow against the ground truth.
    *   **Metrics:** Precision, Recall, F1-score for duplicate pair detection.
    *   **Analysis:** Does CL->ND suffer significantly lower Recall due to missing inter-cluster duplicates? Does it achieve higher Precision on the pairs it *does* find?

*   **Experiment 4.2: Impact on Clustering Quality:**
    *   **Objective:** Assess whether the order of operations affects the quality of the final clusters.
    *   **Procedure:** Take the final output datasets from comparable ND->CL and CL->ND runs. Sample documents from the clusters.
    *   **Metrics:** Calculate cluster quality metrics like Silhouette Score (requires embeddings, potentially recalculated on the final data) or analyze topic coherence qualitatively/using topic modeling on cluster samples.
    *   **Analysis:** Does removing duplicates first (ND->CL) lead to "cleaner" or more coherent clusters compared to CL->ND where near-duplicates might remain within different Stage 2 clusters (if they fell into different Stage 1 clusters)?

