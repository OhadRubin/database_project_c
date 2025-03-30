import os

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.10"
import numpy as np
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
# from pyspark.mllib.feature import HashingTF, IDF

from pyspark.mllib.linalg import Vectors

conf = SparkConf()
conf.set("spark.app.name", "MinHashLSH")
conf.set("spark.debug.maxToStringFields", "100")
conf.set("spark.local.dir", "/dev/shm/pyspark_dir") #TODO: move in arguements
conf.set("spark.driver.memory", "64g")
conf.set("spark.executor.memory", "64g")
spark = SparkSession.builder.config(conf=conf).getOrCreate()


from splearn.rdd import ArrayRDD
from splearn.feature_extraction.text import SparkCountVectorizer, SparkHashingVectorizer
from splearn.feature_extraction.text import SparkTfidfTransformer
from splearn.decomposition import SparkTruncatedSVD
from splearn.pipeline import SparkPipeline


import pandas as pd



# num_nodes=1
# df = spark.read.option("header", "true").csv("/home/ohadr/database_project_c/test_data/sample.csv")


def tfidf_vec(df, column, n_components=128):
    # input: spark dataframe with a column of text that is named column
    # output: spark dataframe with a column of text that is named column and a column of ids that is named __id__
    # and a vector column of tfidf vectors of dimension 128 values that is named tfidf
    df = df.withColumn("__id__", F.monotonically_increasing_id()).cache()
    records = df.select("__id__", column).rdd
    text_rdd = records.map(lambda x: x[1])
    id_rdd = records.map(lambda x: x[0])

    X_rdd = ArrayRDD(text_rdd)

    spark_vectorizer = SparkCountVectorizer()

    dist_pipeline = SparkPipeline((
        ('vect', spark_vectorizer),
        ('tfidf', SparkTfidfTransformer()),
        ('pca', SparkTruncatedSVD(n_components=n_components))
    ))

    pipeline = dist_pipeline.fit(X_rdd)  # SparseRDD
    result_dist = pipeline.transform(X_rdd)  # SparseRDD
    vector_rdd = result_dist.unblock()
    
    # Zip the IDs with the vectors
    recon_rdd = id_rdd.zip(vector_rdd)
    
    # Convert RDD to DataFrame
    from pyspark.ml.linalg import Vectors as MLVectors
    from pyspark.sql.types import StructType, StructField, LongType, ArrayType, DoubleType
    
    # Define schema for the DataFrame
    schema = StructType([
        StructField("__id__", LongType(), False),
        StructField("tfidf_features", ArrayType(DoubleType()), False)
    ])
    
    # Create a DataFrame with the ID and feature vectors
    vector_df = spark.createDataFrame(
        recon_rdd.map(lambda x: (x[0], x[1].tolist())),
        schema
    )
    
    vector_df.collect()
    vector_df.show()
    # Convert array to ML Vector for easier use with MLlib
    from pyspark.sql.functions import udf
    from pyspark.ml.linalg import Vectors as MLVectors, VectorUDT
    
    # UDF to convert array to Vector
    array_to_vector = udf(lambda x: MLVectors.dense(x), VectorUDT())
    
    # Apply the UDF to create the vector column
    vector_df = vector_df.withColumn("tfidf", array_to_vector("tfidf_features")).drop("tfidf_features")
    
    # Join back with the original dataframe
    result_df = df.join(vector_df, on="__id__", how="inner").cache()
    
    return result_df


def tfidf_minhash(df, column, num_perm, ngram_size, min_ngram_size, threshold, n_components=128):
    # this function first performs tfidf vectorization, followed by kmeans clustring, then it performs minhash lsh on each cluster.
    
    # First, perform TF-IDF vectorization
    df = tfidf_vec(df, column, n_components)
    # now we have a dataframe with a column of text that is named column and a column of ids that is named __id__
    # and a vector column of tfidf vectors of dimension 128 values that is named tfidf
    
    # # Import necessary functions from deduplication_spark
    # from pyspark.ml.clustering import KMeans
    # from pyspark.ml.evaluation import ClusteringEvaluator
    # import math
    # from src.deduplication_spark import minhash_lsh, optimal_param, MERSENNE_PRIME, MAX_HASH, RNG
    # import numpy as np
    
    # # Determine the number of clusters - a reasonable heuristic
    # # More data needs more clusters
    # num_records = df.count()
    # k = max(2, min(100, int(math.sqrt(num_records / 10))))
    
    # # Apply KMeans clustering
    # kmeans = KMeans(k=k, seed=42, featuresCol="tfidf")
    # model = kmeans.fit(df)
    
    # # Add cluster predictions to the dataframe
    # clustered_df = model.transform(df)
    
    # # Generate parameters for minhash LSH
    # B, R = optimal_param(threshold, num_perm)
    # HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    # PERMUTATIONS = np.array(
    #     [
    #         (
    #             RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
    #             RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
    #         )
    #         for _ in range(num_perm)
    #     ],
    #     dtype=np.uint64,
    # ).T
    
    # # Group data by cluster for separate deduplication
    # clusters = clustered_df.select("prediction").distinct().collect()
    # deduplicated_dfs = []
    # total_duplicates = 0
    
    # # Process each cluster separately
    # for cluster in clusters:
    #     cluster_id = cluster["prediction"]
    #     cluster_df = clustered_df.filter(F.col("prediction") == cluster_id)
        
    #     # Skip small clusters (less than 2 records)
    #     if cluster_df.count() < 2:
    #         deduplicated_dfs.append(cluster_df.drop("prediction"))
    #         continue
        
    #     # Apply minhash LSH on this cluster
    #     from src.deduplication_spark import deduplicate, generate_hash_values, generate_edges
        
    #     # Get original record count for this cluster
    #     cluster_count = cluster_df.count()
        
    #     # Prepare the edges for deduplication
    #     records = cluster_df.select("__id__", column).rdd
    #     records = records.repartition(num_perm * 2).cache()
        
    #     edges = (
    #         records.flatMap(
    #             lambda x: generate_hash_values(
    #                 content=x[1],
    #                 idx=x[0],
    #                 num_perm=num_perm,
    #                 ngram_size=ngram_size,
    #                 hashranges=HASH_RANGES,
    #                 permutations=PERMUTATIONS,
    #                 min_ngram_size=min_ngram_size,
    #             )
    #         )
    #         .groupBy(lambda x: (x[0], x[1]))
    #         .flatMap(lambda x: generate_edges([i[2] for i in x[1]]))
    #         .distinct()
    #         .cache()
    #     )
        
    #     # Deduplicate this cluster
    #     dedup_cluster_df, cluster_duplicates = deduplicate(edges, cluster_df)
    #     deduplicated_dfs.append(dedup_cluster_df.drop("prediction"))
    #     total_duplicates += cluster_duplicates
    
    # # Combine all deduplicated clusters
    # from functools import reduce
    # if deduplicated_dfs:
    #     final_df = reduce(lambda df1, df2: df1.union(df2), deduplicated_dfs)
    # else:
    #     final_df = df  # In case no deduplication happened
    
    # return final_df, total_duplicates

# # Add a utility function to test both approaches
# def test_deduplication(input_file, column="text", threshold=0.7, num_perm=256, 
#                        ngram_size=5, min_ngram_size=5, n_components=128, method="minhash_lsh"):
#     """
#     Test deduplication on a sample file using different methods
    
#     Parameters:
#     -----------
#     input_file : str
#         Path to input file (CSV or JSON)
#     column : str
#         Column containing text to deduplicate
#     threshold : float
#         Similarity threshold
#     num_perm : int
#         Number of permutations for minhash
#     ngram_size : int
#         Size of n-grams
#     min_ngram_size : int
#         Minimum size of documents to process
#     n_components : int
#         Number of dimensions for TF-IDF
#     method : str
#         Method to use ("minhash_lsh" or "tfidf_minhash")
        
#     Returns:
#     --------
#     tuple
#         (deduplicated dataframe, number of duplicates found)
#     """
#     # Determine file type and load
#     file_extension = os.path.splitext(input_file.strip(".gz"))[1].lower()
    
#     if file_extension == '.csv':
#         df = spark.read.option("header", "true").csv(input_file)
#     elif file_extension.endswith('.json'):
#         df = spark.read.json(input_file)
#     elif file_extension in ['.parquet', '.pq']:
#         df = spark.read.parquet(input_file)
#     else:
#         raise ValueError(f"Unsupported file format: {file_extension}")
    
#     # Apply selected deduplication method
#     if method == "minhash_lsh":
#         from src.deduplication_spark import minhash_lsh
#         return minhash_lsh(df, column, num_perm, ngram_size, min_ngram_size, threshold)
#     elif method == "tfidf_minhash":
#         return tfidf_minhash(df, column, num_perm, ngram_size, min_ngram_size, threshold, n_components)
#     else:
#         raise ValueError(f"Unknown method: {method}")
    
# # Add a main function to allow running from command line
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Text deduplication with TF-IDF and MinHash LSH")
#     parser.add_argument("--input_file", type=str, required=True, help="Path to input file (CSV or JSON)")
#     parser.add_argument("--output", type=str, required=True, help="Output directory for deduplicated data")
#     parser.add_argument("--column", type=str, default="text", help="Column containing text to deduplicate")
#     parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
#     parser.add_argument("--num_perm", type=int, default=256, help="Number of permutations for MinHash")
#     parser.add_argument("--ngram_size", type=int, default=5, help="Size of n-grams")
#     parser.add_argument("--min_ngram_size", type=int, default=5, help="Minimum document size to process")
#     parser.add_argument("--n_components", type=int, default=128, help="Number of dimensions for TF-IDF vectors")
#     parser.add_argument("--method", type=str, default="tfidf_minhash", 
#                         choices=["minhash_lsh", "tfidf_minhash"], 
#                         help="Deduplication method to use")
    
#     args = parser.parse_args()
    
#     # Create output directory if it doesn't exist
#     if not os.path.exists(args.output):
#         os.makedirs(args.output)
#         print(f"Created output directory: {args.output}")
    
#     # Perform deduplication
#     import time
#     start_time = time.time()
    
#     # Get original file size
#     if os.path.exists(args.input_file):
#         original_size_mb = os.path.getsize(args.input_file) / (1024 * 1024)
#         print(f"Original file size: {original_size_mb:.2f} MB")
    
#     # Run deduplication
#     deduplicated_df, duplicate_count = test_deduplication(
#         input_file=args.input_file,
#         column=args.column,
#         threshold=args.threshold,
#         num_perm=args.num_perm,
#         ngram_size=args.ngram_size,
#         min_ngram_size=args.min_ngram_size,
#         n_components=args.n_components,
#         method=args.method
#     )
    
#     # Get record counts
#     original_count = deduplicated_df.count() + duplicate_count
    
#     # Write output
#     output_path = os.path.join(args.output, f"dedup_{args.method}")
#     deduplicated_df.write.parquet(output_path, mode="overwrite")
    
#     # Calculate time and print summary
#     elapsed_time = time.time() - start_time
    
#     print("\nDeduplication Summary:")
#     print(f"Method: {args.method}")
#     print(f"Original records: {original_count}")
#     print(f"Deduplicated records: {deduplicated_df.count()}")
#     print(f"Duplicates removed: {duplicate_count} ({duplicate_count/original_count*100:.2f}%)")
#     print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
#     print(f"Output saved to: {output_path}")
    
#     # Optional: save summary to a log file
#     with open(os.path.join(args.output, "deduplication_log.txt"), "a") as f:
#         f.write(f"\n--- Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
#         f.write(f"Method: {args.method}\n")
#         f.write(f"Input: {args.input_file}\n")
#         f.write(f"Original records: {original_count}\n")
#         f.write(f"Deduplicated records: {deduplicated_df.count()}\n")
#         f.write(f"Duplicates removed: {duplicate_count} ({duplicate_count/original_count*100:.2f}%)\n")
#         f.write(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
#         f.write(f"Parameters: threshold={args.threshold}, num_perm={args.num_perm}, n_components={args.n_components}\n")
#         f.write("----------------------------------------\n")
    
#     # Stop Spark session
#     # spark.stop()
    
    
    