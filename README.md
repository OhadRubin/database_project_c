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
    *(Corresponds approximately to Run ID 16 in the database, though parameters might differ slightly from default)*

*   **ND->CL Workflow Run:**
    ```bash
    python3.10 database_project/src/run_workflows.py --workflow nd_cl --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --use_ray True --limit_files 40
    ```
    *(Corresponds approximately to Run ID 15 in the database, though parameters might differ slightly from default)*

**(e) Code Documentation**

*   **Structure:** The code is organized within the `database_project/src` directory.
    *   `run_workflows.py`: Main orchestrator script.
    *   `ray_minhash.py`: Contains the NDD engine implementation (MinHash, LSH, BTS Union-Find Actors).
    *   `ray_tfidf_vec.py`: Contains the CL engine implementation (TF-IDF/SVD, custom K-Means, Ray actors for fitting/inference, multi-stage logic).
    *   `db.py`: Defines the SQLAlchemy database schema and helper functions for logging benchmark results.
    *   `configs/base.yml`: Configuration file for clustering parameters (batch sizes, dimensions, cluster counts, resource allocation hints).
    *   `download_c4.py`: Script to download the dataset.
    *   `create_plots.py`: Script to query the benchmark database and generate detailed performance/results plots.
    *   `create_plots_simple.py`: Script containing hardcoded summary data (derived from experiments) used to generate the specific plots referenced in this README's results section.
*   **Documentation:** Code includes inline comments explaining key sections. Function and class docstrings provide high-level descriptions. The `db.py` module clearly defines the database schema used for logging experimental results. The main instructions.md and this README.md file provide setup and usage guidance.

**(f) Details of Analyses Performed and Results**

*   **Analysis Goal:** To compare the ND->CL and CL->ND workflows in terms of performance (end-to-end execution time) and output characteristics (number of duplicates identified, final record count) on a ~12GB slice (40 files) of the C4 dataset using a 10-node Ray cluster.
*   **Metrics Collected (Logged via `db.py` and analyzed):**
    *   `execution_time`: Total wall-clock time for the workflow.
    *   `duplicate_count`: Total number of duplicates identified (interpretation differs: global for ND->CL, sum of intra-cluster for CL->ND).
    *   `record_count`: Number of records in the final dataset.
    *   Input parameters (`threshold`, `num_perm`, `limit_files`, etc.).
    *   `false_positive_rate` (Macro): Average false positive rate across duplicate sets (pairs within a set incorrectly identified as duplicates).
    *   `false_positive_count` (Micro): Total count of false positive pairs across all sets, normalized by total pairs considered.
*   **Results Summary:** The following table summarizes key metrics for the 12GB dataset size (40 files on 10 nodes), using representative values directly from the hardcoded data in `create_plots_simple.py`:

    | Workflow | Execution Time (s) | Duplicate Count | Final Record Count | Input Records | Retention | Notes                                   |
    | :------- | :----------------- | :-------------- | :----------------- | :------------ | :-------- | :-------------------------------------- |
    | ND->CL   | 790                | ~420,000        | ~9,580,000         | ~10,000,000   | ~95.8%    | Global deduplication                    |
    | CL->ND   | 735                | ~405,000        | ~9,595,000         | ~10,000,000   | ~96.0%    | Intra-cluster dedup only                |
    
    *Note: Results shown for 12GB data, Threshold=0.7, NumPerm=256. Input record count is approximate; final counts calculated based on duplicate removal.*

*   **Discussion & Conclusions:**
    *   **Performance:** The **CL->ND** workflow was consistently faster than the ND->CL workflow across the tested dataset sizes (3GB, 6GB, 12GB), with a ~7% speed advantage at 12GB (735s vs 790s). This suggests that for this dataset size and configuration, the overhead of performing clustering first followed by parallel intra-cluster NDD was lower than performing global NDD first.
    *   **Output:** The **ND->CL** workflow achieves global deduplication, identifying slightly more duplicates (~420k vs ~405k at T=0.7 for 12GB) and resulting in a slightly smaller final dataset compared to CL->ND. The **CL->ND** workflow only guarantees uniqueness *within* the Stage 1 clusters, potentially missing duplicates that span across different initial clusters, thus retaining slightly more records. The choice depends on the application's requirement for global vs. local uniqueness.
    *   **Trade-offs:** **ND->CL** provides stronger (global) deduplication guarantees but was slower in these experiments. **CL->ND** was faster but provides weaker (intra-cluster) deduplication guarantees. CL->ND might be preferable when runtime is the primary concern and perfect global deduplication is not strictly necessary, or if most duplicates are expected to fall within the initial coarse clusters.
    *   **Recommendation:** Based on this experiment, **CL->ND is recommended if execution speed is the highest priority**, as it was consistently faster. However, if **global deduplication is essential**, then **ND->CL is the appropriate choice**, despite being slightly slower in these tests.
    *   **False Positives:** The analysis (visualized in `create_plots_simple.py`) indicated that both workflows exhibited comparable false positive rates (both macro and micro) under the tested conditions (T=0.7). Micro false positive rates showed a slight tendency to increase with larger dataset sizes for both methods.

*   **Visualization:** The specific summary plots referenced in this analysis (visualizing execution time, duplicate counts, and false positive rates against parameters like dataset size, threshold, and num_perm) were generated using `create_plots_simple.py`. This script utilizes hardcoded representative results derived from the database logs of the experiments. (See also `create_plots.py` for generating plots dynamically from the full database).


**1. Macro False Positive Rate vs Dataset Size (GB)**

| Dataset Size (GB) | Macro FPR (nd_cl) | Macro FPR (cl_nd) |
| :---------------- | :---------------- | :---------------- |
| 3                 | 0.453             | 0.441             |
| 6                 | 0.468             | 0.416             |
| 12                | 0.471             | 0.431             |

**2. Micro False Positive Rate vs Dataset Size (GB)**

| Dataset Size (GB) | Micro FPR (nd_cl) | Micro FPR (cl_nd) |
| :---------------- | :---------------- | :---------------- |
| 3                 | 0.778             | 0.760             |
| 6                 | 0.793             | 0.790             |
| 12                | 0.823             | 0.818             |

**3. Duplicate Count vs Similarity Threshold**

| Threshold | Duplicate Count (nd_cl) | Duplicate Count (cl_nd) |
| :-------- | :---------------------- | :---------------------- |
| 0.6       | 530,000                 | 510,000                 |
| 0.7       | 420,000                 | 405,000                 |
| 0.8       | 240,000                 | 240,000                 |
| 0.9       | 85,000                  | 86,000                  |

**4. Duplicate Count vs Number of Permutations**

| Permutations | Duplicate Count (nd_cl) | Duplicate Count (cl_nd) |
| :----------- | :---------------------- | :---------------------- |
| 128          | 402,500                 | 390,000                 |
| 256          | 421,600                 | 407,500                 |
| 512          | 392,000                 | 380,000                 |

**5. CL Inference Time (seconds) vs Dataset Size (GB)**

| Dataset Size (GB) | CL Inference Time (nd_cl) | CL Inference Time (cl_nd) |
| :---------------- | :------------------------ | :------------------------ |
| 3                 | 150                       | 175                       |
| 6                 | 200                       | 213                       |
| 12                | 295                       | 298                       |

**6. Total Execution Time (seconds) vs Dataset Size (GB)**

| Dataset Size (GB) | Total Execution Time (nd_cl) | Total Execution Time (cl_nd) |
| :---------------- | :--------------------------- | :--------------------------- |
| 3                 | 570                          | 530                          |
| 6                 | 635                          | 550                          |
| 12                | 790                          | 735                          |