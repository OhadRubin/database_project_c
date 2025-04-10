
# Final Project Report: Scalable Near-Duplicate Detection and Clustering Workflows on Ray

**Course:** Advanced Topics in Data Management

**(Based on provided file structure and Table of Contents)**

---

**Table of Contents**

**(Matches the provided `toc.md`)**

---

**Prelude: The Challenge of Scale and Similarity**

Modern datasets are growing at an unprecedented rate, presenting significant challenges for effective data management and analysis. Two fundamental operations often required are Near-Duplicate Detection (NDD) – identifying items that are highly similar but not necessarily identical – and Clustering (CL) – grouping similar items together. Performing these tasks efficiently at scale, especially when dealing with billions of documents like those found in web crawls (e.g., the C4 dataset), demands sophisticated algorithms and distributed computing frameworks.

---

**Chapter 1: Setting the Stage: Introduction and Motivation**

*   **1.1 The Twin Giants: NDD and CL in Modern Data**
    *   **1.1.1 Defining Near-Duplication vs. Exact Duplication:** Exact duplicates are byte-for-byte identical. Near-duplicates, the focus here, exhibit high content similarity despite minor differences (e.g., formatting changes, small edits, boilerplate variations). NDD is crucial for data cleaning, reducing redundancy, and improving the quality of downstream tasks like model training.
    *   **1.1.2 The Role of Clustering in Data Understanding and Organization:** Clustering groups documents based on semantic similarity, helping to discover topics, organize large corpora, enable focused analysis within groups, and potentially improve the efficiency of other operations by processing clusters independently.

*   **1.2 The Tyranny of N<sup>2</sup>: Why Scalability Matters**
    *   **1.2.1 Computational Complexity of Pairwise Comparisons:** Calculating similarity between all pairs of *N* documents requires O(N<sup>2</sup>) comparisons. For datasets like C4 with billions of documents, this is computationally infeasible.
    *   **1.2.2 Data Volume Growth Trends:** Web crawls, user-generated content, and scientific datasets continue to expand exponentially, making scalable solutions not just desirable but essential.

*   **1.3 The Central Question: *Order Out of Chaos?* - Investigating ND->CL vs. CL->ND**
    The conventional approach in large-scale data processing pipelines often involves running NDD *before* CL. This project investigates the trade-offs and potential benefits of reversing this order: performing CL first, then applying NDD *within* the resulting clusters. Our central research question is: **For large text corpora, what are the performance and output characteristic trade-offs between performing Near-Duplicate Detection then Clustering (ND->CL) versus Clustering then Near-Duplicate Detection (CL->ND)?**

    *   **1.3.1 Hypothesis for ND->CL Advantages:**
        *   *1.3.1.A Reduction in Clustering Input Size:* Removing duplicates first reduces the number of documents fed into the potentially expensive clustering step.
        *   *1.3.1.B Potential for Simplified Downstream Processing:* A globally deduplicated dataset might be easier to manage later.
    *   **1.3.2 Hypothesis for CL->ND Advantages:**
        *   *1.3.2.A Focused NDD within Similar Groups:* NDD is applied only to pairs *within* the same cluster, drastically reducing the total number of candidate pairs considered compared to global NDD, potentially speeding up the NDD part.
        *   *1.3.2.B Potential for Higher Quality Duplicates (Fewer False Positives across dissimilar clusters):* By restricting NDD to clusters of already similar documents, we might reduce the chance of incorrectly identifying documents from vastly different topics as near-duplicates (a potential risk with LSH's probabilistic nature if parameters aren't perfectly tuned globally).

*   **1.4 Our Compass: A Ray-Powered Approach to Workflow Comparison**
    To investigate this question at scale, we leverage the Ray framework.
    *   **1.4.1 Rationale for Choosing the Ray Framework:**
        *   *1.4.1.A Actor Model Suitability for Distributed State:* Ray's actor model is well-suited for managing distributed stateful computations, as demonstrated in our implementations of the distributed Union-Find (`BTSUnionFind` in `ray_minhash.py`) and the K-Means training/inference actors (`fit_models_remote`, `TFIDFInferenceModel`, `KMeansInferenceModel` in `ray_tfidf_vec.py`).
        *   *1.4.1.B Ray Data for Scalable Transformations:* Ray Data provides a high-level API for distributed dataset loading (`ray.data.read_json`), parallel batch processing (`map_batches`), filtering (`filter`), and shuffling (`repartition`), forming the backbone of our data pipeline orchestration in `run_workflows.py`.

*   **1.5 Blueprint of the Investigation: Report Roadmap**
    This report details our investigation: Chapter 2 covers the foundational algorithms. Chapters 3-7 describe the system architecture and implementation details of the NDD engine, CL engine, orchestrator, and benchmarking system. Chapters 8-9 outline the experimental setup using the C4 dataset on a TPU cluster. Chapters 10-13 present and discuss the performance and output results comparing the ND->CL and CL->ND workflows. Finally, Chapter 14 concludes and suggests future directions.

---

**Chapter 2: Foundations: Algorithms and Frameworks**

*   **2.1 Finding Needles in Haystacks: The Art of Approximate Similarity**
    *   **2.1.1 The Essence of Overlap: Jaccard Similarity:**
        *   *2.1.1.A Set Representation of Documents (Shingling):* Documents are converted into sets of overlapping substrings (n-grams or shingles). Our implementation uses n-grams of tokens.
            *   *2.1.1.A.i N-Gram Generation:* The `ngrams` function (in `ray_minhash.py`), adapted from NLTK, uses `itertools.tee` to efficiently generate n-grams from a sequence of tokens.
            *   *2.1.1.A.ii Handling Preprocessing:* The `tokenize` function splits text based on non-alphanumeric characters (using the `NON_ALPHA` regex) and optionally converts to lowercase before generating n-grams.
        *   *2.1.1.B Formal Definition:* J(A, B) = |A ∩ B| / |A ∪ B|.

    *   **2.1.2 Fingerprinting the Document: MinHash Signatures (Broder, 1997 Revisited):**
        *   *2.1.2.A Permutation-Based MinHashing Concept:* Applying a random permutation to all shingles and finding the first shingle in the permuted order yields one hash value. Repeating with many permutations creates a signature.
        *   *2.1.2.B The Hashing Trick:* Instead of permutations, multiple hash functions are applied to each shingle. The minimum hash value observed for each function across all shingles in a document forms the signature.
            *   *2.1.2.B.i Universal Hash Functions:* We simulate permutations using hash functions of the form `(ax + b) mod P`, where `P` is a large prime (like `MERSENNE_PRIME` = 2<sup>61</sup>-1).
            *   *2.1.2.B.ii Applying Multiple Hash Functions:* The `calc_minhash` method uses precomputed random pairs (`self.perm_a`, `self.perm_b`) and applies the hashing trick via efficient NumPy broadcasting. It uses `sha1_hash32` to initially hash the n-gram bytes.
        *   *2.1.2.C Property:* The probability that the MinHash values for two documents collide for a single hash function is approximately equal to their Jaccard Similarity.

    *   **2.1.3 Hashing for Locality: The LSH Banding Strategy:**
        *   *2.1.3.A Dividing Signatures into Bands and Rows:* The MinHash signature (length `num_perm`) is divided into `b` bands, each containing `r` rows (`num_bands`, `num_rows_per_band` in `RayBTSMinhashDeduplicator`).
        *   *2.1.3.B Hashing Bands to Buckets:* Each band (a vector of `r` hash values) is hashed into a bucket. Documents sharing *at least one* identical band hash are considered candidate pairs.
            *   *2.1.3.B.i Band Vector Representation:* Achieved by slicing the signature array (`hash_values[start:end]`).
            *   *2.1.3.B.ii Hash Function for Bands:* In `calc_minhash`, the band index and the bytes of the band vector are combined to form the hash key sent to the Union-Find actors.
        *   *2.1.3.C The S-Curve:* This strategy creates an S-shaped curve where pairs with similarity above a certain threshold (related to `b` and `r`) have a high probability of being detected, while pairs below have a low probability.
        *   *2.1.3.D Parameter Tuning:* The choice of `b` and `r` for a given `num_perm` determines the similarity threshold `t` targeted by the S-curve.
            *   *2.1.3.D.i False Positive/Negative Trade-off:* More bands (smaller `r`) reduce false negatives but increase false positives (more candidates). Fewer bands (larger `r`) increase false negatives but reduce false positives.
            *   *2.1.3.D.ii Optimal Parameter Calculation:* The `optimal_param` function (from `ray_minhash.py`, based on datasketch logic) calculates `b` and `r` to minimize a weighted sum of false positive and false negative probabilities for a target `threshold`.

*   **2.2 Grouping the Masses: Techniques for Document Clustering**
    *   **2.2.1 Weighing the Words: TF-IDF Vectorization:**
        *   *2.2.1.A Term Frequency (TF):* Measures how frequently a term appears in a document. Scikit-learn's default often uses log normalization.
        *   *2.2.1.B Inverse Document Frequency (IDF):* Measures how informative a term is. Rare terms get higher IDF scores. Scikit-learn's default uses smooth IDF.
        *   *2.2.1.C Combining TF and IDF:* TF * IDF gives high weight to terms frequent in a document but rare overall.
        *   *2.2.1.D Vector Space Model:* Represents documents as high-dimensional sparse vectors based on TF-IDF scores. Our implementation uses `NumberNormalizingVectorizer` from `ray_tfidf_vec.py` which preprocesses numbers.

    *   **2.2.2 Sculpting the Feature Space: Dimensionality Reduction via SVD:**
        *   *2.2.2.A Motivation:* High-dimensional TF-IDF vectors suffer from the curse of dimensionality and noise. SVD helps find a lower-dimensional latent semantic space.
        *   *2.2.2.B Truncated SVD:* We use `sklearn.decomposition.TruncatedSVD` (within the pipeline in `get_sklearn_feature_pipeline`) to reduce the dimensionality of the TF-IDF matrix to `n_components` (e.g., 128 from `base.yml`).
        *   *2.2.2.C Latent Semantic Analysis (LSA):* SVD applied to TF-IDF is often interpreted as LSA, capturing underlying topics.

    *   **2.2.3 The Centroid Dance: K-Means Clustering:**
        *   *2.2.3.A Algorithm Steps:* Initialize *k* centroids, assign points to nearest centroid, update centroids to the mean of assigned points, repeat until convergence.
        *   *2.2.3.B Objective Function:* Minimize the sum of squared distances between points and their assigned centroid (Within-Cluster Sum of Squares).
        *   *2.2.3.C Challenges:* Sensitive to initial centroid placement, requires specifying *k* beforehand. Our implementation (`KMeans` class in `ray_tfidf_vec.py`) addresses some challenges with online updates and optional balancing.

*   **2.3 Conquering Scale: Paradigms for Distributed Computing**
    *   **2.3.1 The MapReduce Legacy (Dean & Ghemawat, 2008 Insights):**
        *   A foundational model for processing large datasets in parallel across a cluster. Key phases are Map (process data chunks independently), Shuffle/Sort (group by key), and Reduce (aggregate results per key). Fault tolerance is a key feature. While not directly using Hadoop MapReduce, the principles influence distributed algorithm design. The BTS Union-Find can be seen as having Map-like (hashing, local grouping) and Reduce-like (merging, communication) phases.
    *   **2.3.2 The Actor Model Ascendant: An Introduction to Ray:**
        *   *2.3.2.A Ray Core Concepts:* Ray enables distributed Python via simple decorators. `@ray.remote` functions become stateless Tasks, `@ray.remote class` definitions become stateful Actors. `ray.put` places objects in the distributed object store, `ray.get` retrieves them. This model underpins our `IdGenerator`, `EdgeBuffer`, `BTSUnionFind`, `TFIDFInferenceModel`, and `KMeansInferenceModel`.
        *   *2.3.2.B Ray Data:* Provides a dataset abstraction (`ray.data.Dataset`) for distributed data processing. We use `read_json` for input, `map_batches` for applying transformations (like MinHashing or TF-IDF inference) in parallel, `filter` for selecting data subsets (e.g., for stage 2 clustering), and `repartition`/`materialize` for controlling data layout and execution.
        *   *2.3.2.C Scheduling and Resource Management:* Ray manages task/actor placement and scheduling based on resource requests (e.g., `num_cpus=`, `resources={"TPU-v4-8-head": 1}` used in `fit_models_remote` and `KMeansInferenceModel` calls in `ray_tfidf_vec.py`).

---

**Act I: Forging the Pipeline - System Architecture and Implementation**

---

**Chapter 3: The Grand Design: System Architecture Overview**

*   **3.1 Visualizing the Flow: Architectural Diagram (ND->CL vs. CL->ND)**

    *(Conceptual Diagram Description)*

    *   **Input:** C4 JSON.gz files read using `ray.data.read_json`.
    *   **Workflow Choice (via `run_workflows.py` `--workflow` arg):**
        *   **ND->CL:**
            1.  `Ray Dataset` -> **ND Engine (`ray_minhash.py::run_nd_step_for_workflow`)**: Input: Dataset[text], Output: Deduplicated Dataset[text] + `duplicate_count`.
            2.  Deduplicated `Ray Dataset` -> **CL Engine (`ray_tfidf_vec.py::run_cl_step_for_workflow`)**: Input: Dataset[text], Output: Clustered Dataset[text, cluster_A, cluster_B]. Involves Stage 1 (global fit/predict) and Stage 2 (fit/predict within Stage 1 clusters).
        *   **CL->ND:**
            1.  `Ray Dataset` -> **CL Engine (`ray_tfidf_vec.py::run_cl_step_for_workflow` with `should_dedup=True`)**: Input: Dataset[text], Output: Clustered & Deduplicated Dataset[text, cluster_A, cluster_B].
                *   Stage 1: Global fit/predict -> Dataset[text, cluster_A].
                *   Stage 2: For each cluster_A:
                    *   Fit/predict Stage 2 -> Dataset[text, cluster_A, cluster_B].
                    *   **ND Engine (`ray_minhash.py::dedup_remote`)** called on cluster's data -> Deduplicated Dataset[text, cluster_A, cluster_B].
                *   Union results.
    *   **Output:** Final Ray Dataset (schema depends on workflow, includes cluster columns).
    *   **Benchmarking:** Metadata (params, time, counts) logged to **Benchmarking DB (`db.py`)** via `run_workflows.py`.

*   **3.2 Core Modules:**
    *   **ND Engine:** Implemented in `ray_minhash.py`. Performs MinHash-LSH using BTS Union-Find.
    *   **CL Engine:** Implemented in `ray_tfidf_vec.py`. Performs multi-stage TF-IDF/SVD + K-Means clustering. Can optionally invoke the ND engine internally for the CL->ND workflow.
    *   **Orchestrator:** The main script `run_workflows.py`. Parses arguments, loads data, invokes the appropriate engine(s) based on the selected workflow, and logs results.
    *   **Benchmarking DB:** Defined in `db.py` using SQLAlchemy. Stores run parameters, execution time, output counts, and potentially resource/accuracy metrics.

*   **3.3 Data Highway: Leveraging Ray Datasets for Distributed Flow**
    *   **3.3.1 Reading Input Data:** `ray.data.read_json(input_file, override_num_blocks=cfg.num_blocks)` reads potentially many input files (controlled by `limit_files`) into a distributed Ray Dataset.
    *   **3.3.2 Batching and Parallel Operations (`map_batches`):** This is the core mechanism for applying our custom logic (MinHashing, TF-IDF inference, K-Means inference) in parallel across the cluster.
        *   *3.3.2.A Batch Format:* We primarily use `batch_format="pandas"` to work with Pandas DataFrames within the mapping functions/actors.
        *   *3.3.2.B Concurrency Control:* Parameters like `concurrency` in `map_batches` calls (e.g., `cfg.tfidf.inference.concurrency` in `ray_tfidf_vec.py`) control the maximum number of parallel tasks executing the map function.
    *   **3.3.3 Materialization and Repartitioning:** `.materialize()` forces execution of upstream transformations. `.repartition()` (used in `run_workflows.py` and `run_cl_step_for_workflow`) controls the number of blocks (partitions) in the dataset, affecting parallelism and potentially mitigating memory issues. `.sort()` is used in `stage2` to order the final output.

---

**Chapter 4: The Deduplication Engine: Near-Duplicate Detection (`ray_minhash.py`)**

*   **4.1 From Text to Sets: Tokenization and N-Gram Shingling (`tokenize`, `ngrams`)**
    *   **4.1.1 Regular Expression:** `NON_ALPHA = re.compile("[^A-Za-z_0-9]")` is used to split text into tokens based on non-alphanumeric characters.
    *   **4.1.2 `ngrams` Function:** Takes a sequence of tokens and an integer `n`. Uses `itertools.tee` to create `n` iterators over the sequence, advances them appropriately, and then `zip`s them to yield tuples representing the n-grams. Avoids creating intermediate lists.
    *   **4.1.3 `tokenize` Function:** Takes a string, splits it using `NON_ALPHA`, generates n-grams using `ngrams` (with `ngram_size` and `min_ngram_size` parameters), joins the tokens within each n-gram back into a string, encodes them to UTF-8 bytes, and returns a set of these bytes.

*   **4.2 Crafting the Signatures: Parallel MinHash Computation (`calc_minhash` method)**
    *   **4.2.1 Per-Document Hashing and Signature Generation:**
        *   For each document, `tokenize` generates shingles.
        *   `sha1_hash32` converts each shingle byte string to an initial 32-bit integer hash.
        *   The hashing trick `(hv * perm_a + perm_b) % MERSENNE_PRIME` is applied using NumPy broadcasting (`hv[:, None] * self.perm_a[None, :] ...`) across all shingles (`hv`) and all precomputed permutations (`self.perm_a`, `self.perm_b`).
        *   The minimum resulting hash value for each permutation is computed across all shingles (`phv.min(axis=0)`), forming the MinHash signature (`hash_values`).
    *   **4.2.2 Batching:** Ray Data's `map_batches` implicitly handles batching; `calc_minhash` processes one document at a time within the loop iterating over the batch passed from `map_batches`.
    *   **4.2.3 Handling Empty Documents:** If `tokenize` yields no tokens, a predefined `self.empty_hash_value` is used, directing these documents to a specific `BTSUnionFind` actor (`self.empty_hash_table_id`).

*   **4.3 Finding Candidate Kin: LSH Banding Implementation (within `calc_minhash`)**
    *   **4.3.1 Generating Band Hashes:**
        *   The code iterates through the precomputed `self.hash_ranges`, which define the start and end indices for each band in the signature.
        *   For each band `i`, the corresponding slice of the signature (`hash_values[start:end]`) is extracted. Its bytes (`.tobytes()`) are concatenated with the band index bytes (`i.to_bytes(4, 'big')`) to form a unique `hash_value` for that document band.
    *   **4.3.2 Grouping Documents:** The first hash value within the band (`hash_values[start]`) is used to determine the target `BTSUnionFind` actor ID (`% self.union_find_parallel_num`). The `(hash_value, uid)` tuple is appended to the `pairs` dictionary, keyed by the target actor ID. This dictionary is then used to send batches of work to the appropriate actors.

*   **4.4 *Innovation Spotlight:* Unifying Duplicates at Scale with Distributed BTS Union-Find**
    This project implements a distributed Balanced Transitive Sync (BTS) Union-Find algorithm using Ray actors, designed for massive scale NDD.
    *   **4.4.1 Coordinating IDs: `IdGenerator` Actor:** A simple Ray actor (`IdGenerator`) manages a global counter (`next_id`) to assign unique integer IDs (`uid`) to each document across the distributed dataset via atomic remote calls to `get_next_id.remote(count)`.
    *   **4.4.2 Handling the Edge Flood: `EdgeBuffer` Actors:** These actors (`EdgeBuffer`) act as mailboxes. When a `BTSUnionFind` actor needs to communicate a potential edge (a required union operation) to another actor responsible for one of the involved UIDs, it places the edge in the target actor's corresponding `EdgeBuffer` via `set_edges`. The target actor later retrieves these edges via `get_edges`. `clear` resets the buffer.
    *   **4.4.3 The Core Logic: `BTSUnionFind` Actor:** Each actor is responsible for a subset of the UIDs based on hashing (`uid // BATCH_SIZE % self.parallel_num`).
        *   *4.4.3.A State:* Key state includes `hash_table` (for LSH band hash collisions), `parent` (the core Union-Find data structure mapping UID to parent UID), `old_parent` (for convergence checking), `parallel_id` (its own ID), `remote_edge_buffers` (references to other actors' buffers).
        *   *4.4.3.B Hash Table Grouping (`add_key_value_pairs`):* Receives `(hash_value, uid)` pairs from `calc_minhash`. Stores UIDs sharing the same LSH band hash (`hash_value`) in the local `hash_table`. If a bucket exceeds `union_threshold`, `union_list` is called immediately to merge those UIDs locally.
        *   *4.4.3.C Local Unions (`union_list`, `flush_key_value_pairs`):* `union_list` finds the minimum root among a list of UIDs and updates the `parent` dictionary to point them all to that minimum root. `flush_key_value_pairs` processes any remaining entries in the `hash_table` at the end of the hashing phase.
        *   *4.4.3.D Balanced Union-Find Iterations (`balanced_union_find`):* This is the main loop for merging components.
            *   Processes edges currently buffered locally (`self.edge_buffer`) using `self.union`.
            *   Fetches edges sent from other actors by calling `get_edges` on its own `EdgeBuffer` actor (using `ray.wait` for asynchronous batching).
            *   Applies `self.union` to these fetched remote edges.
            *   Calls `self.rebalancing` to improve load balance.
            *   Returns `True` if the `parent` dictionary changed, indicating convergence has not yet been reached.
        *   *4.4.3.E Path Compression (`find`):* Implements standard path compression: when finding the root of `x`, it recursively finds the ultimate root and then updates `self.parent[x]` to point directly to it.
        *   *4.4.3.F Union Operation (`union`):* Finds the roots of `x` and `y`. If different, it links the root of the larger ID to the root of the smaller ID (heuristic union-by-rank/size).
    *   **4.4.4 Balancing and Communication Strategy:**
        *   *4.4.4.A Edge Distribution (`distribute_edge`, `edge_redistribution`):* `distribute_edge` determines the responsible actor(s) for a `(u, v)` edge based on hashing `u` and `v`. It adds the edge to `self.edge_list_dict` keyed by the target actor ID(s). `edge_redistribution` is called initially to distribute all initial parent links discovered locally.
        *   *4.4.4.B Inter-Actor Communication (`set_edge_buffer`):* After populating `self.edge_list_dict`, this method separates edges destined for the current actor (`self.edge_buffer`) from those for remote actors. It then sends the remote edges dictionary to the actor's own `EdgeBuffer` via a remote call.
        *   *4.4.4.C Rebalancing (`rebalancing`):* Aims to ensure that the representative element (root) for a connected component resides on the correct actor node for better data locality. It identifies representative UIDs per partition within a component and updates parent pointers accordingly.
        *   *4.4.4.D Final Filtering (`dup_idx`, `squeeze`):* After convergence, `filter_with_union_find` queries each `BTSUnionFind` actor using `dup_idx.remote(query)` to identify which of its UIDs are *not* the root of their component (i.e., are duplicates). `squeeze` cleans up the actor's state after merging is complete.

---

**Chapter 5: The Clustering Engine: Hierarchical Grouping (`ray_tfidf_vec.py`)**

*   **5.1 Vectorizing the Corpus: The Distributed TF-IDF/SVD Pipeline**
    *   **5.1.1 `NumberNormalizingVectorizer`:** A custom class inheriting `TfidfVectorizer`. It overrides `build_tokenizer` to wrap the default tokenizer with `number_normalizer`, which replaces tokens starting with a digit with a special "#NUMBER" token.
    *   **5.1.2 Scikit-learn Pipeline Integration (`get_sklearn_feature_pipeline`):** Defines a `sklearn.pipeline.Pipeline` containing:
        1.  `'tfidf'`: An instance of `NumberNormalizingVectorizer` with custom `stop_words` (including "#NUMBER").
        2.  `'svd'`: `TruncatedSVD` for dimensionality reduction (using `n_components` and `random_seed` from config).
        3.  `'normalizer'`: `Normalizer` to scale vectors to unit length (often helps K-Means).
    *   **5.1.3 Distributed Training (`_fit_models_remote`, `fit_models_remote`):**
        *   A sample of the dataset (`ds.limit(cfg.max_docs)`) is taken.
        *   The sample is collected to the driver node (`sample_ds.to_pandas()`) - **Note: This can be a bottleneck for very large `max_docs`.**
        *   The scikit-learn pipeline (`vectorizer`) is fitted on the sample texts (`vectorizer.fit_transform(texts)`).
        *   The custom `KMeans` model is then fitted on the resulting embeddings (`fit_kmeans(embeddings, cfg.kmeans)`).
        *   This entire fitting process runs within a Ray actor (`@ray.remote def fit_models_remote`) scheduled with potentially significant CPU resources (`num_cpus=cfg.tfidf.train.num_cpus`) and potentially a TPU (`resources={"TPU-v4-8-head": 1}`) for the K-Means part.
        *   The fitted `vectorizer` and `kmeans` objects are returned, implicitly serialized by Ray and stored in the object store via the returned `ObjectRef`.
    *   **5.1.4 Distributed Inference (`TFIDFInferenceModel`):**
        *   An actor class designed for `map_batches`.
        *   `__init__`: Takes the `ObjectRef` of the fitted models, uses `ray.get` to retrieve the actual `vectorizer` model (deserializing it).
        *   `__call__`: Takes a batch (Pandas DataFrame), extracts the "text" column, applies the *fitted* `self.vectorizer.transform` to get embeddings, and adds these embeddings as a new "embeddings" column to the batch DataFrame.

*   **5.2 *Innovation Spotlight:* Accelerated and Adaptable K-Means**
    This project utilizes a custom `KMeans` implementation designed for flexibility and performance in a distributed setting.
    *   **5.2.1 Custom `KMeans` Class:** Defined in `ray_tfidf_vec.py`. Stores `n_clusters`, `cluster_centers`, `device`, and `balanced` flag. Includes `load`/`save` methods (using `pickle`) and an `initialize` method (random sampling).
    *   **5.2.2 Online K-Means Updates (`fit` method with `online=True`):** The `fit` method supports an `online` mode. When `online=True`, it initializes centroids only on the first call (`iter_k == 0`) and then refines the existing `self.cluster_centers` based on each subsequent batch of data passed to it. Convergence within a batch step is checked via `center_shift` or `iter_limit`. It handles potential empty clusters during updates by reassigning a random point from the current batch. This is used in `fit_kmeans` which iterates through the embeddings using a DataLoader.
    *   **5.2.3 Balanced Clustering (`balanced=True` and `auction_lap`):** If `balanced=True`, instead of simple nearest centroid assignment, the `fit` method calculates the full distance matrix between batch points and centroids and then uses the `auction_lap` function (an implementation of the auction algorithm for linear assignment) to find a balanced assignment (roughly equal points per cluster). This is more computationally expensive than standard assignment.
    *   **5.2.4 GPU/TPU Acceleration with JAX:**
        *   *5.2.4.A JAX Pairwise Distance (`_jax_pairwise_distance`, `create_jax_pairwise_distance`):* Provides a JAX-based function to compute pairwise Euclidean distances efficiently, leveraging JAX's NumPy-like API and XLA compilation. `reshape_for_jax` prepares the data for `pmap`.
        *   *5.2.4.B Pmap for Parallelism:* `jax.pmap` is used to automatically parallelize the distance calculation across available JAX devices (GPUs or TPU cores).
        *   *5.2.4.C JAX-based Nearest Cluster Assignment (`compile_nearest_cluster`):* This function compiles a JAX computation graph using `pmap` and `pad_shard_unpad` (for TPU efficiency) to find the index of the nearest cluster center for a batch of embeddings. This compiled function (`tagging_func`) is used during inference.
    *   **5.2.5 Assigning Clusters with Ray (`KMeansInferenceModel`):**
        *   An actor class for `map_batches`.
        *   `__init__`: Retrieves the fitted `kmeans` model via `ray.get`. It also compiles the JAX nearest cluster assignment function (`compile_nearest_cluster`) and stores it as `self.tagging_func`. Stores the `cluster_col_name` from the config.
        *   `__call__`: Takes a batch containing the "embeddings" column, converts embeddings to a NumPy array, applies the compiled `self.tagging_func` to get cluster assignments, adds the assignments as a new column (`self.cluster_col_name`), and drops the "embeddings" column.

*   **5.3 Building the Hierarchy: Multi-Stage Clustering (`stage1`, `stage2`)**
    *   **5.3.1 Stage 1 (`stage1`):** A wrapper function that calls `fit_predict`. `fit_predict` orchestrates the fitting (`fit_models_remote`) and prediction (`TFIDFInferenceModel`, `KMeansInferenceModel`) for a single clustering stage, resulting in a Ray Dataset tagged with the first-level cluster IDs (e.g., `cluster_A`).
    *   **5.3.2 Stage 2 (`stage2`):** Implements the second level of clustering.
        *   Filters the Stage 1 tagged dataset into separate Ray Datasets, one for each Stage 1 cluster ID (`og_ds.filter(expr=f"{stage1_cluster_col_name} == {cluster_id}")`).
        *   Launches a parallel `fit_predict_remote` task *for each* Stage 1 cluster dataset, fitting new TF-IDF/KMeans models specific to that cluster's data and predicting Stage 2 cluster IDs (e.g., `cluster_B`).
        *   **Conditional Deduplication:** If `cfg.should_dedup` is True (set by `run_workflows.py` for the CL->ND workflow), it launches an additional `dedup_remote` task (calling the ND engine) on the result of `fit_predict_remote` for each cluster.
        *   Collects the results (references to the final datasets for each cluster) using `ray.get`.
        *   Combines the processed datasets from all Stage 1 clusters back into a single Ray Dataset using `final_ds.union(*ds_list[1:])`.
        *   Sorts the final dataset based on the cluster ID columns (`final_ds.sort(cfg.partition_cols[:2])`).

---

**Chapter 6: The Maestro: Workflow Orchestration (`run_workflows.py`)**

*   **6.1 Command-Line Interface (`create_parser`):** Uses `argparse` to define command-line arguments, including:
    *   `--workflow`: Selects between `nd_cl` and `cl_nd`.
    *   `--input_file`, `--output`, `--limit_files`: Data source and destination.
    *   `--threshold`, `--ngram_size`, etc.: Parameters for the ND step.
    *   `--config_file`: Path to the YAML config for the CL step (`base.yml`).
    *   `--use_ray`: (Potentially less relevant now as both workflows heavily use Ray).
    *   `--notes`: For logging benchmark details.

*   **6.2 Ray Initialization and Cluster Discovery:** Initializes Ray connection (`ray.init(address='auto')`) and determines the number of nodes (`len(ray.nodes())`) for logging purposes.

*   **6.3 Configuration Loading:** Reads the clustering configuration from the YAML file (`read_config`) specified by `--config_file`. Injects the parsed command-line arguments (`args`) into the config object (`cfg.args`) so they are accessible within the CL/ND steps if needed (e.g., for deduplication parameters in `stage2`).

*   **6.4 Data Loading and Initial Repartitioning:** Loads the JSON data using `ray.data.read_json` based on `--input_file` and `--limit_files`. Optionally repartitions the initial dataset (`ds.repartition(1000)`).

*   **6.5 Conducting the ND->CL Symphony (`workflow == "nd_cl"`):**
    *   Invokes the ND step: `intermediate_ray_ds, nd_duplicates, nd_time = run_nd_step_for_workflow(ray_df, args)`. This returns the deduplicated Ray Dataset and the count of duplicates found globally.
    *   The intermediate dataset is materialized and repartitioned (`intermediate_ray_ds.repartition(1000).materialize()`).
    *   Sets `cfg.base_stage.should_dedup = False` to prevent deduplication inside the CL step.
    *   Invokes the CL step: `clustered_ds = run_cl_step_for_workflow(intermediate_ray_ds, cfg)`. This performs the multi-stage clustering on the already deduplicated data.

*   **6.6 Weaving ND into CL: The CL->ND Approach (`workflow == "cl_nd"`):**
    *   Sets `cfg.base_stage.should_dedup = True`. This flag is passed down to the `stage2` function within `run_cl_step_for_workflow`.
    *   Invokes the CL step: `clustered_ds = run_cl_step_for_workflow(ray_df, cfg)`.
    *   Inside `run_cl_step_for_workflow`, the `stage2` function is called. Because `cfg.should_dedup` is True, `stage2` will execute `fit_predict_remote` *and then* `dedup_remote` for each Stage 1 cluster, performing NDD locally within each cluster *after* Stage 2 clustering.

*   **6.7 Final Output Writing and Benchmarking Integration:**
    *   *(Note: The explicit writing of the final `clustered_ds` to disk (e.g., `clustered_ds.write_parquet(final_output_path)`) seems missing in the provided `run_workflows.py` snippet, but would typically occur here or at the end of `run_cl_step_for_workflow`.)*
    *   Captures total wall clock time (`actual_workflow_time`).
    *   Connects to the database using `db.init_db()` and `db.get_session()`.
    *   Calls `BenchmarkRun.create_from_args` to log the run details to the database, passing command-line args, total duplicate count (interpretation depends on workflow), final record count, total execution time, workflow name (`args.workflow`) as implementation, number of nodes, notes, file limit, and data size.

---

**Chapter 7: The Scribe: Benchmarking and Logging (`db.py`)**

*   **7.1 Designing the Ledger: Database Schema Rationale:**
    A relational database schema (implemented using SQLAlchemy) is used to systematically record and compare workflow executions.
    *   **7.1.1 `BenchmarkRun` Table:** The central table, storing one entry per workflow execution. Fields include:
        *   `id`: Primary key.
        *   `timestamp`: When the run started.
        *   `input_file`, `output_dir`: Data locations.
        *   `implementation`: Identifier for the run type (e.g., 'nd_cl', 'cl_nd').
        *   `num_nodes`: Number of Ray nodes used.
        *   ND Parameters: `threshold`, `ngram_size`, `min_ngram_size`, `num_perm`.
        *   Results: `execution_time` (total wall clock), `duplicate_count` (total duplicates removed/identified), `record_count` (final count).
        *   Run Context: `notes`, `limit_files`, `total_size_gb`.
    *   **7.1.2 `ResourceMetric` Table:** (Optional, requires `psutil` and monitoring) Stores resource usage summaries linked to a `BenchmarkRun`. Fields: `cpu_percent_avg/max`, `memory_usage_avg/max_mb`, `network_sent/recv_mb`, `disk_read/write_mb`.
    *   **7.1.3 `AccuracyMetric` Table:** (Optional, requires ground truth) Stores accuracy metrics (Precision, Recall, F1) compared to a reference, linked to a `BenchmarkRun`. Fields: `reference_implementation`, `true_positives`, `false_positives`, `false_negatives`, `precision`, `recall`, `f1_score`.
    *   **7.1.4 Relationships:** One-to-many relationships defined using `relationship` from `BenchmarkRun` to `ResourceMetric` and `AccuracyMetric`, with `back_populates` for bidirectional access and `cascade="all, delete-orphan"` to manage dependent records.

*   **7.2 Capturing the Run: Storing Parameters and Results via SQLAlchemy:**
    *   **7.2.1 Database Initialization (`init_db`):** Creates the SQLAlchemy engine. It checks the `POSTGRES_ADDRESS` environment variable to connect to PostgreSQL; otherwise, it defaults to a local SQLite file (`benchmark_results.db`). Creates tables if they don't exist (`Base.metadata.create_all`).
    *   **7.2.2 Session Management (`get_session`, `object_session`):** `get_session` creates a new session for database interactions. `object_session` retrieves the session associated with an ORM object.
    *   **7.2.3 Data Insertion Methods:**
        *   `BenchmarkRun.create_from_args`: Used by `run_workflows.py`. Takes the parsed `args`, calculated results (`duplicate_count`, `record_count`, `execution_time`), and run context to create and commit a new `BenchmarkRun` record.
        *   `BenchmarkRun.create_from_spark_run`: (Likely from older usage) Similar, but tailored for parameters common in the Spark script.
        *   `BenchmarkRun.add_resource_metrics`, `BenchmarkRun.add_accuracy_metrics`: Helper methods on the `BenchmarkRun` object to easily add related resource or accuracy metrics after the main run is recorded. The `monitor_resources` function shows an example pattern for collecting and adding resource metrics.

---

**Act II: The Crucible - Experimental Setup and Execution**

---

**Chapter 8: Preparing the Arena: Environment and Dataset**

*   **8.1 The Testbed: Hardware Configuration**
    *   **8.1.1 TPU Node Specifications:** Experiments were conducted on a cluster of Google Cloud TPU VMs. Based on `run.sh`, the nodes appear to be `v4-8` type, each having multiple TPU cores (typically 4 per v4 chip, 8 chips per VM = 32 cores) and significant High Bandwidth Memory (HBM). Each node also has host CPUs and RAM.
    *   **8.1.2 Number of Nodes Used:** The `run.sh` script dynamically determines the number of available TPU nodes (`N_NODES=$(gcloud ... | wc -l)`) and waits for that many nodes to join the Ray cluster. Based on the project context, this was likely 10 nodes. `ray.nodes()` confirms the cluster size before proceeding.
    *   **8.1.3 Network Interconnect:** TPU v4 pods utilize high-speed inter-chip interconnects (ICI) for efficient communication, crucial for distributed tasks like the shuffle phases in NDD or collective operations potentially used by JAX.

*   **8.2 Software Stack: Key Libraries and Versions**
    *   Python: 3.10 (specified in `run_workflows.py` and `run.sh`)
    *   Ray: 2.43.0 (specified in `run.sh` and `ray_minhash.py`)
    *   NumPy: ~1.0 (specified in `run.sh` and `ray_minhash.py`)
    *   Scikit-learn: (Version not specified, but used for TF-IDF/SVD/Normalization in `ray_tfidf_vec.py`)
    *   JAX: (Version not specified, but used for K-Means acceleration in `ray_tfidf_vec.py`)
    *   SQLAlchemy: (Version not specified, used for DB logging in `db.py`)
    *   Pandas: (Version not specified, used for batch format in Ray Data)
    *   (Potentially others: `psycopg2` for PostgreSQL, `pyarrow`, `tqdm`, `ml_collections`, `httpx`, `asyncio`)

*   **8.3 Fueling the Fire: The C4 Dataset Slice**
    *   **8.3.1 Source:** The English split (`en`) of the Colossal Clean Crawled Corpus (C4) dataset from AllenAI, hosted on Hugging Face. Accessed via URLs like `https://huggingface.co/datasets/allenai/c4/resolve/.../en/c4-train.{index:05d}-of-01024.json.gz` as seen in `download_c4.py`.
    *   **8.3.2 Number of Files Processed:** The `run.sh` script downloads and processes the first 40 files (`index=0` to `39`) using `download_c4.py` and sets `limit_files=40` when calling `run_workflows.py`.
    *   **8.3.3 Total Uncompressed Size:** Based on the sample data in `viewer.ipynb` (run ID 7 or 13, `limit_files=40`), the total size processed is approximately **11.88 GB**.
    *   **8.3.4 Sample Record Structure:** Each JSON object within the files contains fields like `text` (the document content), `timestamp`, and `url`. The workflows primarily use the `text` field.

*   **8.4 Ignition Sequence: Breakdown of `run.sh` Script Logic**
    The `run.sh` script automates the setup and execution on each cluster node:
    1.  **Dependency Installation:** Installs required Python packages (`ray`, `numpy`).
    2.  **Code Update:** Clones or pulls the latest code from the GitHub repository.
    3.  **Environment Variable Setup:** Sets `POSTGRES_ADDRESS` for connecting to the benchmarking database.
    4.  **Data Download and Verification:** Creates `/dev/shm/c4_files`, runs `download_c4.py`, and loops until exactly 40 files are present in `/dev/shm` (using shared memory for faster access).
    5.  **GCS Fuse Mount:** Mounts a GCS bucket (`meliad2_us2_backup`) to `/mnt/gcs_bucket` using `gcsfuse`. Uses `/dev/shm/gcs_cache` for caching, enabling flags for parallel downloads to potentially speed up access to other data/models stored on GCS (like the output base dir `/mnt/gcs_bucket/ray_clustering_output` in `base.yml`).
    6.  **Ray Cluster Formation:** Checks if Ray is running. If not, the designated head node (`v4-8-node-2`'s IP) starts Ray in head mode (`ray start --head`), while other nodes join using `--address="$HEAD_IP:6379"`. Specifies `resources='{"TPU-v4-8-head": 1}'` to make the TPU resources visible to Ray's scheduler.
    7.  **Cluster Readiness Check:** Enters a loop, periodically checking `len(ray.nodes())` until the expected number of nodes (`$N_NODES`) have joined the cluster.

---

**Chapter 9: Running the Gauntlet: Execution Procedures**

*   **9.1 Invoking the Workflows: Command-Line Examples**
    The `run.sh` script executes the main workflow script (`run_workflows.py`) only on the head node after the cluster is ready. The exact commands executed (for `limit_files=40`) were:

    *   **For `nd_cl` workflow (if `WORKFLOW="nd_cl"` was set in `run.sh`):**
        ```bash
        python3.10 database_project/src/run_workflows.py --workflow nd_cl --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --use_ray True --limit_files 40
        ```
    *   **For `cl_nd` workflow (as set in the provided `run.sh`):**
        ```bash
        python3.10 database_project/src/run_workflows.py --workflow cl_nd --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --use_ray True --limit_files 40
        ```
    *   **9.1.3 Key Parameter Values Used:**
        *   `workflow`: `nd_cl` or `cl_nd`
        *   `input_file`: `"/dev/shm/c4_files/c4-train.*.json.gz"`
        *   `output`: `/dev/shm/c4_outputs` (Intermediate/temporary, final might go to GCS via config)
        *   `limit_files`: `40`
        *   `threshold` (ND default): `0.7`
        *   `num_perm` (ND default): `256`
        *   `config_file` (CL default): `"database_project/src/configs/base.yml"`

*   **9.2 Monitoring the Process: Ray Dashboard and Logging**
    *   **9.2.1 Accessing the Ray Dashboard:** Once the Ray head node starts, it typically prints the URL for the dashboard (usually `http://<head_node_ip>:8265`). This web UI is crucial for monitoring.
    *   **9.2.2 Key Dashboard Metrics:**
        *   *Cluster View:* Verify all nodes are connected and see their resource utilization (CPU, RAM, GPU/TPU if configured).
        *   *Actors Tab:* Monitor the state (Alive, Pending, Dead) and resource usage of actors like `IdGenerator`, `EdgeBuffer`, `BTSUnionFind`, `TFIDFInferenceModel`, `KMeansInferenceModel`. Useful for spotting bottlenecks or failures.
        *   *Tasks Tab:* View the progress of Ray tasks, especially those generated by `map_batches`.
        *   *Logical View:* Visualize the DAG of tasks and actors.
        *   *Object Store:* Monitor memory usage in Ray's distributed object store.
    *   **9.2.3 Interpreting Log Output:**
        *   `run_workflows.py` uses standard Python logging. INFO messages track workflow progress ("Executing ND -> CL workflow...", "Running ND step...", etc.).
        *   Ray logs (from driver and workers) provide detailed information about task/actor execution, potential serialization issues, or resource warnings. Errors often appear here first.
        *   Our custom modules (`ray_minhash`, `ray_tfidf_vec`) also use logging (`logger.info(...)`) to report progress (e.g., "MinHash time =", "merge time =", "[Stage X] Fitting vectorizer...").

---

**Act III: The Reckoning - Results and Analysis**

*(Note: The following sections describe the analysis based on the expected data logged to the database (`db.py`) and potentially viewed via `viewer.ipynb`. Specific numbers are illustrative or refer to the sample data in `viewer.ipynb`.)*

---

**Chapter 10: Performance Under the Microscope**

*   **10.1 The Stopwatch Test: End-to-End Execution Time Comparison**
    *   **10.1.1 Wall Clock Time Measurements:** The primary metric is `BenchmarkRun.execution_time` recorded in the database by `run_workflows.py`. We compare the `execution_time` for runs where `implementation` is 'nd_cl' vs. 'cl_nd' (assuming runs corresponding to IDs 15 and 16 in `viewer.ipynb`'s sample data represent these, though they used different parameters/code versions).
        *   Example (from sample): Run 15 (ND->CL, ID=15) took ~165s. Run 16 (CL->ND, ID=16) took ~388s. *Caution: These sample runs might not be directly comparable due to potential differences in code or parameters beyond the workflow order.* A dedicated pair of runs with identical parameters except workflow order is needed for a fair comparison.
    *   **10.1.2 Time Breakdown (Approximation):**
        *   *ND Step:* `run_nd_step_for_workflow` returns `nd_time`. Log messages in `ray_minhash.py` also report "MinHash time" and "merge time".
        *   *CL Step:* Can be estimated by timing around the `run_cl_step_for_workflow` call in `run_workflows.py`, or by summing timings logged within `ray_tfidf_vec.py` (e.g., "Stage X complete. Time taken: ..."). The CL->ND workflow's time includes both clustering and the nested deduplication.

*   **10.2 Throughput Analysis:** Calculated as (Total Input Records) / `execution_time`. The total input records for 40 files is ~14.25 million (from Run ID 16 sample).
    *   Example (using sample Run 15): ~14.25M / 165s ≈ 86,000 records/sec.
    *   Example (using sample Run 16): ~14.25M / 388s ≈ 37,000 records/sec.
    *   *Comparison depends heavily on the actual timings of comparable runs.*

*   **10.3 Visualizing Speed:** Performance Graphs
    *   [Insert Bar Chart comparing `execution_time` for 'nd_cl' vs 'cl_nd' runs with identical parameters, using data queried from the `benchmark_runs` table via `viewer.ipynb` or similar.]
    *   [Optional: Insert Stacked Bar Chart showing approximate time breakdown (ND vs CL) for each workflow.]

---

**Chapter 11: Examining the Output: Deduplication and Record Counts**

*   **11.1 Quantifying Redundancy: Duplicate Counts Comparison**
    *   **11.1.1 `duplicate_count` for ND->CL:** The `BenchmarkRun.duplicate_count` value directly reflects the total number of duplicate documents identified and removed by the initial global ND step. (e.g., Run 15 shows 421,609).
    *   **11.1.2 `duplicate_count` for CL->ND:** The `BenchmarkRun.duplicate_count` logged by `run_workflows.py` in this case represents the *sum* of duplicates found *within each cluster* during the `stage2` execution. It does *not* represent globally unique documents removed, as duplicates might exist across different clusters but wouldn't be compared. (e.g., Run 16 shows 0, which seems incorrect based on the logic - this might indicate an issue in how duplicates were counted or aggregated in that specific run/code version, or that deduplication was skipped despite `should_dedup=True`). Assuming correct aggregation, this number is expected to be *lower* than the ND->CL count because inter-cluster duplicates are missed.
    *   **11.1.3 Comparison Table/Chart:**
        [Insert Table/Chart comparing `BenchmarkRun.duplicate_count` for 'nd_cl' vs 'cl_nd' runs, highlighting the different interpretation.]

*   **11.2 The Final Tally: Resulting Record Counts Comparison**
    *   **11.2.1 `record_count` Comparison:** The `BenchmarkRun.record_count` reflects the number of documents remaining *after* the entire workflow completes.
        *   Example (sample): Run 15 (ND->CL) resulted in ~13.83M records. Run 16 (CL->ND) resulted in ~14.25M records.
        *   The CL->ND workflow is expected to yield a *higher* final `record_count` than ND->CL because it only removes duplicates *within* clusters, potentially leaving duplicates that exist across different clusters.
    *   **11.2.2 Percentage of Records Retained:** (Final `record_count` / Original Input Count) * 100.
        *   Example (sample Run 15): (~13.83M / ~14.25M) * 100 ≈ 97.0%
        *   Example (sample Run 16): (~14.25M / ~14.25M) * 100 ≈ 100% (Again, the 0 duplicate count seems suspect here). If duplicates *were* removed, this percentage would be slightly lower but likely > 97.0%.

*   **11.3 Interpreting Output Differences:**
    [Insert Table summarizing `execution_time`, `duplicate_count`, `record_count`, and Retention % for comparable 'nd_cl' and 'cl_nd' runs, based on data from `viewer.ipynb`/DB queries.]
    The key difference lies in the scope of deduplication: ND->CL performs global deduplication, while CL->ND performs local (within-cluster) deduplication. This directly impacts the final record count and the interpretation of the duplicate count metric.

---

**Chapter 12: (Optional) Peeking Inside the Clusters (`examine_clusters.ipynb`)**

*This chapter describes analysis that *could* be performed using the notebook if the final clustered datasets were saved and loaded.*

*   **12.1 Qualitative Analysis: Sample Documents**
    *   **12.1.1 Method:** Load the final clustered dataset (output from `run_cl_step_for_workflow`). Group the data by the final cluster IDs (`cluster_A`, `cluster_B`). Randomly select a few clusters and sample several documents (`text` field) from within each selected cluster.
    *   **12.1.2 Examples:** Present snippets of text from sampled documents within the same cluster to demonstrate topic cohesion. Compare examples from different clusters to show separation. Analyze if clusters from the ND->CL output appear different qualitatively from the CL->ND output (e.g., are CL->ND clusters "cleaner" due to intra-cluster deduplication?).

*   **12.2 Quantitative Metrics:**
    *   **12.2.1 Calculation Method:** If feasible (computationally), calculate metrics like the Silhouette Score (`sklearn.metrics.silhouette_score`) on a sample of the data and embeddings (embeddings would need to be regenerated or saved). Calculate cluster size distribution by counting documents per final cluster ID.
    *   **12.2.2 Results and Interpretation:** Compare Silhouette scores between the two workflows (a higher score indicates better-defined clusters). Analyze the cluster size distributions – are they skewed? Does the K-Means `balanced=True` option (if used) result in more uniform sizes? Does the workflow order impact these metrics?
    *   [Insert Chart of Cluster Size Distribution for ND->CL.]
    *   [Insert Chart of Cluster Size Distribution for CL->ND.]
    *   [Insert Table comparing Silhouette Scores (if calculated).]

---

**Chapter 13: Synthesizing the Findings: Discussion**

*   **13.1 Order Matters: Evaluating the ND->CL vs. CL->ND Trade-offs**
    *   **13.1.1 Performance Implications:**
        *   *Input Size Impact:* ND->CL reduces the input size for the CL step. If CL is the bottleneck and NDD significantly reduces data, ND->CL might be faster overall. CL->ND applies NDD to smaller chunks (clusters), potentially speeding up the NDD *part*, but the overall time depends on the CL overhead and the degree of parallelism achieved in the nested NDD calls. Our sample results (Run 15 vs 16) *suggest* ND->CL was significantly faster, but requires confirmation with truly comparable runs.
        *   *Parallelism Differences:* Both workflows leverage Ray's parallelism. ND->CL parallelizes global NDD, then parallelizes CL stages. CL->ND parallelizes CL stages, and *within* Stage 2, it parallelizes NDD calls across different Stage 1 clusters. The efficiency depends on cluster balance and resource allocation.
    *   **13.1.2 Output Characteristics and Quality Differences:**
        *   *Inter- vs. Intra-cluster Duplicates:* ND->CL removes duplicates globally. CL->ND only removes duplicates *within* clusters, meaning near-duplicates assigned to different initial clusters will *both* be retained. This leads to a higher final record count for CL->ND.
        *   *Impact on Dataset Composition:* The final dataset from ND->CL is globally unique (according to the NDD parameters). The CL->ND dataset is unique *within* each cluster but may contain global duplicates. The choice depends on whether global uniqueness or faster processing (potentially) is prioritized.
    *   **13.1.3 Resource Utilization Patterns (Hypothesized):**
        *   *Memory:* Global NDD (ND->CL) might have high memory peaks during the shuffle/merge phase of the BTS Union-Find. CL->ND might have lower peaks during NDD (as it operates on smaller chunks) but could have high memory usage during the CL fitting stages if the sample size (`max_docs`) is large. K-Means itself, especially with JAX on TPUs, might utilize significant HBM.
        *   *Network:* Global NDD involves significant communication for edge distribution and fetching in BTS Union-Find. CL->ND's NDD communication is localized per cluster, but the initial CL stages also involve data movement. The overall network bottleneck depends on the specific implementation details and data distribution.

*   **13.2 Answering the Research Question: Which Workflow Prevails? (Context Matters!)**
    Based on the expected trade-offs and initial sample results:
    *   **13.2.1 Scenarios Favoring ND->CL:** When global deduplication is strictly required, and/or when the initial NDD step significantly reduces the dataset size, potentially making the subsequent CL step much faster. If overall wall-clock time is the primary concern and the NDD reduction is substantial, this might be preferred. The sample data (Run 15 vs 16) points towards ND->CL being faster.
    *   **13.2.2 Scenarios Favoring CL->ND:** When processing time for NDD is the main bottleneck *and* the data clusters well naturally. If near-duplicates are overwhelmingly likely to fall within the same cluster anyway, CL->ND might offer comparable *quality* of deduplication within topics while potentially reducing the NDD computation burden by avoiding global comparisons. However, it sacrifices global uniqueness.
    *   **13.2.3 Recommendation based on C4 Experiment:** Based on the sample results showing a potentially significant speed advantage for ND->CL (165s vs 388s), **ND->CL appears to be the more performant workflow for this specific C4 dataset slice and implementation**. It also guarantees global deduplication. However, the CL->ND run's zero duplicate count needs investigation. If CL->ND were implemented purely for speed without requiring global uniqueness, its performance relative to ND->CL would need re-evaluation with confirmed timings.

*   **13.3 Triumphs and Tribulations: Strengths and Limitations**
    *   **13.3.1 Strengths:**
        *   *Scalability:* Demonstrated use of Ray and Ray Data for processing a significant dataset (~12GB, 14M docs) on a multi-node cluster.
        *   *Advanced Distributed Algorithms:* Implementation of distributed BTS Union-Find for NDD and integration of JAX-accelerated K-Means for CL.
        *   *Modularity:* Separation of ND, CL, orchestration, and benchmarking concerns into different modules/scripts.
        *   *Workflow Comparison Framework:* Successfully set up to compare the two distinct workflow orders.
    *   **13.3.2 Limitations:**
        *   *Accuracy Metric Simplicity:* Relies on `duplicate_count` and `record_count`. True pairwise accuracy (Precision/Recall/F1 against ground truth) is not implemented (requires labeled duplicate pairs).
        *   *Parameter Tuning:* NDD (threshold, num_perm, bands/rows) and CL (k, SVD components) parameters were likely chosen manually (defaults or from `base.yml`). Optimal values might differ.
        *   *Fixed Algorithm Choices:* Explores only MinHash/LSH for NDD and TF-IDF/SVD/KMeans for CL.
        *   *CL Training Bottleneck:* Collecting samples to the driver (`.to_pandas()`) for CL model fitting can limit scalability for larger training sample sizes.

*   **13.4 Reflections on Distributed Implementation Challenges**
    *   **13.4.1 Debugging:** Debugging failures in distributed Ray applications (actors dying, tasks failing, serialization errors, hangs) can be complex, often requiring careful examination of logs across multiple nodes and understanding Ray's execution model. Issues in the complex BTS Union-Find communication logic would be particularly challenging.
    *   **13.4.2 Memory Management:** Ray's object store can become a bottleneck if tasks generate large intermediate objects or if data skew occurs. Tuning batch sizes (`batch_size` in configs, `BATCH_SIZE` in `ray_minhash`), using `.repartition()`, and efficient data handling within tasks/actors are crucial. The CL fitting sample collection is a potential memory pressure point on the driver.
    *   **1.4.3 Integrating Different Libraries:** Combining Scikit-learn (CPU-based), JAX (TPU/GPU-based), NumPy, Pandas, and Ray requires careful management of data formats (Pandas vs NumPy vs JAX arrays), device placement (CPU vs TPU), and serialization. Ensuring models trained in one library (sklearn) can be efficiently used within Ray actors alongside data processed by another (JAX) needs careful design (as seen in `TFIDFInferenceModel` and `KMeansInferenceModel`).

---

**Epilogue: The Road Ahead**

---

**Chapter 14: Conclusion and Future Directions**

*   **14.1 Summary of the Expedition: Key Findings Recapitulated**
    This project successfully implemented and compared two distinct workflows (ND->CL and CL->ND) for processing a large text corpus (40 files, ~12GB of C4) using the Ray distributed computing framework. We implemented a scalable MinHash-LSH NDD engine featuring a distributed BTS Union-Find algorithm and a multi-stage clustering engine using TF-IDF/SVD and a custom, JAX-accelerated K-Means. A benchmarking system was established to log parameters and performance metrics. Preliminary analysis (based on sample data) suggests that the conventional ND->CL workflow may offer better overall performance for this dataset while guaranteeing global deduplication, whereas CL->ND sacrifices global uniqueness for potentially localized NDD efficiency.

*   **14.2 The Horizon Beckons: Promising Avenues for Future Work**
    *   **14.2.1 Exploring Alternative Algorithms:**
        *   *NDD:* Implement and compare SimHash (better for cosine similarity) or other LSH families.
        *   *Clustering:* Explore distributed density-based algorithms like DBSCAN (using Ray) which don't require specifying *k*, or hierarchical clustering methods. Implement fully distributed TF-IDF/KMeans fitting to avoid the driver bottleneck.
    *   **14.2.2 Enhancing Accuracy Metrics:**
        *   *Pairwise Comparison:* Develop or adapt a method to compare the identified duplicate pairs against a ground truth set (sampled or synthetic) to calculate true Precision, Recall, and F1-score for NDD quality, storing these in the `AccuracyMetric` table.
        *   *Ground Truth:* Create a reliable ground truth subset for accurate evaluation.
    *   **14.2.3 Hyperparameter Optimization and Sensitivity Analysis:**
        *   *Ray Tune:* Utilize Ray Tune to perform automated hyperparameter search for both NDD (threshold, num_perm, bands/rows) and CL (k, n_components) parameters, optimizing for execution time and/or output quality metrics.
        *   *Sensitivity:* Analyze how performance and output characteristics change with variations in key parameters like Jaccard threshold, number of clusters, or SVD components.
    *   **14.2.4 Real-time/Streaming Integration:** Adapt the NDD and CL components (especially the online K-Means) to work with streaming data sources using frameworks like Ray Streaming.
    *   **14.2.5 Advanced Resource Monitoring and Profiling:** Integrate Ray's built-in profiling tools or external memory profilers to get a deeper understanding of resource bottlenecks (CPU, memory, network, object store usage) within specific tasks and actors for further optimization. Populate the `ResourceMetric` table more consistently.

---

**References**

*   Broder, A. Z. (1997). On the resemblance and containment of documents. *Proceedings Compression and Complexity of Sequences 1997*, 21–29. IEEE.
*   Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified Data Processing on Large Clusters. *Communications of the ACM, 51*(1), 107–113.
*   Moritz, P., Nishihara, R., Wang, S., Tumanov, A., Liaw, R., Liang, E., Elibol, M., Yang, Z., Paul, W., Jordan, M. I., & Stoica, I. (2018). Ray: A Distributed Framework for Emerging AI Applications. *13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)*.
*   *Paper on BTS Union-Find (if available, e.g., https://ieeexplore.ieee.org/document/10598116)*
*   Ray Documentation: https://docs.ray.io/
*   Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
*   JAX Documentation: https://jax.readthedocs.io/
*   SQLAlchemy Documentation: https://docs.sqlalchemy.org/
*   C4 Dataset: Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research, 21*(140), 1-67.

---

**Appendices**

*   **A: Full Clustering Configuration (`database_project/src/configs/base.yml`)**
    ```yaml
    # (Content of base.yml pasted here, potentially with added comments)
    ray_max_docs_limit: null
    base_dir: /mnt/gcs_bucket/ray_clustering_output
    num_blocks: 1000
    base_stage:
      max_docs: 5000 # Sample size for fitting models
      tfidf:
          train:
            n_components: 128 # Dimensions after SVD
            random_seed: 42
            batch_size: 1024 # Batch size for TF-IDF fitting? (Check usage)
            num_cpus: 210 # CPUs for model fitting actor

          inference:
            num_cpus: 5 # CPUs per inference actor
            batch_size: 1024 # Batch size for TF-IDF transform
            concurrency: 400 # Max parallel TF-IDF inference actors

      kmeans:
          train:
            batch_size: 2048 # Batch size for online K-Means updates
            iter_limit: 5 # Max iterations per online update batch

          inference:
            batch_size: 8192 # Batch size for K-Means prediction
            num_cpus: 20 # CPUs per K-Means inference actor
            concurrency: 10 # Max parallel K-Means inference actors

    stages_list:
      - name: stage1
        pretty_name: "Stage 1"
        cluster_col_name: cluster_A # Output column name for this stage
        kmeans:
          n_clusters: 10 # Number of clusters for Stage 1

      - name: stage2
        pretty_name: "Stage 2"
        cluster_col_name: cluster_B # Output column name for this stage

        # Override base stage configs specifically for Stage 2
        tfidf:
          train:
            num_cpus: 110 # Less CPUs perhaps as fitting happens per cluster
          inference:
            concurrency: 30 # Adjust concurrency for Stage 2

        kmeans:
          n_clusters: 10 # Number of sub-clusters within each Stage 1 cluster
          inference:
            concurrency: 2 # Adjust concurrency for Stage 2 K-Means inference

    # Added based on run_workflows.py logic
    # base_stage:
    #   should_dedup: false # Default: no deduplication in CL step
                           # Set to true by run_workflows.py for cl_nd workflow
    ```

*   **B: Detailed Database Schema (`database_project/src/db.py` Classes)**
    ```python
    # (Relevant SQLAlchemy class definitions from db.py pasted here)

    class BenchmarkRun(Base):
        __tablename__ = 'benchmark_runs'
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.utcnow) # Run start time
        input_file = Column(String(255)) # Path pattern to input data
        output_dir = Column(String(255)) # Path to output (may be intermediate)
        notes = Column(Text, nullable=True) # User notes
        duplicate_count = Column(Integer) # Total duplicates identified/removed
        record_count = Column(Integer) # Records remaining after workflow
        implementation = Column(String(50)) # e.g., 'nd_cl', 'cl_nd'
        num_nodes = Column(Integer) # Ray cluster size
        threshold = Column(Float) # NDD Jaccard threshold
        ngram_size = Column(Integer) # NDD n-gram size
        min_ngram_size = Column(Integer) # NDD min doc size for n-grams
        num_perm = Column(Integer) # NDD number of permutations
        execution_time = Column(Float) # Total wall clock time (seconds)
        limit_files = Column(Integer, nullable=True) # Max input files processed
        total_size_gb = Column(Float, nullable=True) # Size of input processed (GB)
        # Relationships
        resource_metrics = relationship("ResourceMetric", back_populates="benchmark_run", cascade="all, delete-orphan")
        accuracy_metrics = relationship("AccuracyMetric", back_populates="benchmark_run", cascade="all, delete-orphan")
        # ... (methods create_from_args, add_resource_metrics, etc.)

    class ResourceMetric(Base):
        __tablename__ = 'resource_metrics'
        id = Column(Integer, primary_key=True)
        result_id = Column(Integer, ForeignKey('benchmark_runs.id')) # Link to BenchmarkRun
        cpu_percent_avg = Column(Float) # Avg CPU %
        cpu_percent_max = Column(Float) # Max CPU %
        memory_usage_avg_mb = Column(Float) # Avg RAM (MB)
        memory_usage_max_mb = Column(Float) # Max RAM (MB)
        network_sent_mb = Column(Float) # Network sent (MB)
        network_recv_mb = Column(Float) # Network received (MB)
        disk_read_mb = Column(Float) # Disk read (MB)
        disk_write_mb = Column(Float) # Disk written (MB)
        # Relationship
        benchmark_run = relationship("BenchmarkRun", back_populates="resource_metrics")

    class AccuracyMetric(Base):
        __tablename__ = 'accuracy_metrics'
        id = Column(Integer, primary_key=True)
        result_id = Column(Integer, ForeignKey('benchmark_runs.id')) # Link to BenchmarkRun
        reference_implementation = Column(String(100)) # Against which run compared
        true_positives = Column(Integer) # Correctly identified duplicate pairs
        false_positives = Column(Integer) # Incorrectly identified duplicate pairs
        false_negatives = Column(Integer) # Missed duplicate pairs
        precision = Column(Float) # TP / (TP + FP)
        recall = Column(Float) # TP / (TP + FN)
        f1_score = Column(Float) # Harmonic mean of precision and recall
        # Relationship
        benchmark_run = relationship("BenchmarkRun", back_populates="accuracy_metrics")
    ```

*   **C: Illustrative Code Snippets**
    *   *BTS Union-Find Communication Snippet (`balanced_union_find` in `BTSUnionFind`):*
        ```python
        # Process local edges
        for x, y in self.edge_buffer:
            self.union(x, y)
        self.edge_buffer = []

        # Fetch and process remote edges
        result_refs = []
        for remote_edge_buffer in self.remote_edge_buffers:
            # ... (ray.wait logic for batching) ...
            result_refs.append(
                remote_edge_buffer.get_edges.remote(self.parallel_id))
        edge_list = ray.get(result_refs)
        for edges in edge_list:
            for x, y in edges:
                self.union(x, y)
        ```
    *   *Multi-Stage CL with Conditional ND (`stage2` in `ray_tfidf_vec.py`):*
        ```python
        stage1_datasets = [...] # Filter by stage 1 cluster
        ds_ref_list = []
        for ds in stage1_datasets:
            # Fit/Predict Stage 2 cluster
            ds_ref = fit_predict_remote.remote(ds, cfg)
            # Conditionally deduplicate within cluster
            if cfg.should_dedup:
                ds_ref = dedup_remote.remote(ds_ref, cfg)
            ds_ref_list.append(ds_ref)
        # Combine results
        ds_list = ray.get(ds_ref_list)
        final_ds = ds_list[0].union(*ds_list[1:])
        ```
    *   *Workflow Orchestration (`run_workflows.py`):*
        ```python
        if args.workflow == "nd_cl":
            # Stage 1: ND
            intermediate_ray_ds, nd_duplicates, nd_time = run_nd_step_for_workflow(ray_df, args)
            # ... repartition ...
            cfg.base_stage.should_dedup = False
            # Stage 2: CL
            clustered_ds = run_cl_step_for_workflow(intermediate_ray_ds, cfg)
            # ... timing and logging ...
        elif args.workflow == "cl_nd":
            cfg.base_stage.should_dedup = True
            # Stage 1+2: CL+ND (ND happens inside run_cl_step_...)
            clustered_ds = run_cl_step_for_workflow(ray_df, cfg)
            # ... timing and logging ...
        # ... Log results to DB using BenchmarkRun.create_from_args ...
        ```

*   **D: Raw Experimental Data Tables**
    [Insert Formatted Table(s) showing key columns from the `benchmark_runs` table for the main comparison runs, potentially generated from `viewer.ipynb` or direct DB queries.]