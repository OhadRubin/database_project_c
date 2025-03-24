# Project: Scalable Near-Duplicate Document Detection with Clustered MinHash and MapReduce

## 1. Introduction and Objectives (Course-Focused)

This project for the Advanced Topics in Data Management course integrates two key topics: **large-scale data processing (MapReduce)** and **approximate similarity search (MinHash)**, with a novel optimization using **clustering (TF-IDF and k-means)**. The problem addressed is the efficient identification of near-duplicate documents within a large corpus, a common requirement in data cleaning, plagiarism detection, and web-scale information retrieval.  Direct pairwise comparisons are computationally infeasible (O(n<sup>2</sup>)) for large document collections.


### 1.1 Implemented Paper Algorithms

Our implementation is based on the following papers studied in the course:

1. **MapReduce**: Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified Data Processing on Large Clusters. Communications of the ACM, 51(1), 107-113.
   - We implement the core MapReduce paradigm as presented in this seminal paper, including the Map, Shuffle, and Reduce phases.

2. **MinHash **: Broder, A. Z. (1997). On the resemblance and containment of documents. Proceedings of the Compression and Complexity of Sequences, 21-29.
   - We implement the MinHash algorithm for signature generation and the banding technique for Locality Sensitive Hashing as described in this paper.


## 2. Background (Tailored to Course Topics)

This section briefly reviews the core concepts, directly linking them to the course material.

### 2.1. Approximate Similarity Search and MinHash

As covered in the course, exact similarity search (e.g., calculating Jaccard similarity for all document pairs) is computationally expensive.  MinHash provides a probabilistic approximation of Jaccard similarity, allowing for efficient identification of *candidate* similar pairs.  It falls under the broader category of *Locality Sensitive Hashing (LSH)* techniques, which aim to hash similar items to the same bucket with high probability.

*   **Jaccard Similarity:**  Measures the overlap between sets of n-grams.
*   **MinHash:**  Generates a short "signature" for each document by applying multiple hash functions and taking the minimum hash value for each.  The probability of signature collision approximates Jaccard similarity.
* **LSH** Devide the signature matrix into b bands consisting of r rows to identify candidate similar documents.

### 2.2. Clustering with TF-IDF

We utilize *Term Frequency-Inverse Document Frequency (TF-IDF)* to represent documents as numerical vectors, capturing the importance of terms within each document relative to the entire corpus.  This is a standard technique in information retrieval, and its use in clustering was discussed in class as a way to group documents with similar content.

*   **TF (Term Frequency):**  Reflects how often a term appears in a document.
*   **IDF (Inverse Document Frequency):**  Reflects how rare a term is across the entire corpus.
*   **TF-IDF:**  The product of TF and IDF, highlighting terms that are both frequent in a document and relatively rare in the corpus.
*   **K-means Clustering:** A standard clustering algorithm that partitions data points (in our case, TF-IDF vectors) into *k* clusters, minimizing the within-cluster variance.

### 2.3. MapReduce: Distributed Data Processing

MapReduce is a fundamental paradigm for large-scale data processing, covered extensively in the course. It provides a framework for distributing computations across a cluster of machines, enabling us to handle datasets that exceed the memory capacity of a single machine.

*   **Map Phase:** Applies a user-defined `map` function to each input data chunk *independently*.  This produces intermediate key-value pairs.  The independence of the `map` operations is key to parallelization.
*   **Reduce Phase:**  Applies a user-defined `reduce` function to groups of intermediate key-value pairs that share the same key.  This aggregates the data. The `reduce` operations can also be performed in parallel.

## 3. Analysis and Results

We will compare three different methods for near-duplicate document detection:

### Method 1: MinHash with Spark (multi-worker)
This method implements the standard MinHash algorithm using Apache Spark with multiple workers, leveraging distributed computation across a cluster.

### Method 2: MinHash with Spark (single-worker)
This method uses the same MinHash implementation but runs on a single worker, providing a baseline for comparison with the distributed approach.

### Method 3: MinHash with Hierarchical TF-IDF Clustering (single-worker)
This is our novel approach that uses TF-IDF clustering as a preprocessing step to reduce the number of comparisons needed for MinHash, running on a single worker.

### Evaluation Plan
Our evaluation will consist of:
1. Timing each method to compare performance
2. Analyzing how many duplicates the approximate Hierarchical TF-IDF method missed compared to the standard MinHash approach
3. Monitoring network usage, as Method 1 will use significantly more network bandwidth due to its distributed nature