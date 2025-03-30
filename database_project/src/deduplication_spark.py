import os

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.10"

import sys
from itertools import tee
from logging import Logger
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy.integrate import quad as integrate
import glob

SEED = 42

RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))





from typing import List, Iterable
from itertools import tee
import re
import hashlib
import re
import struct
import numpy as np

MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
def ngrams(sequence: List[str], n: int, min_ngram_size: int = 5) -> Iterable:
    """
    Code taken from NLTK, without padding.

    Parameters
    ----------
    sequence : list
        The sequence of items to be converted into n-grams.
    n : int
        The order of the n-grams to be extracted.
    min_ngram_size : int
        The minimum number of items in the sequence to generate n-grams.

    Returns
    -------
    Iterable
        The n-grams generated from the sequence.

    Examples
    --------
    >>> list(ngrams(['a', 'b', 'c', 'd'], 2))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    >>> list(ngrams(['a', 'b', 'c', 'd'], 3))
    [('a', 'b', 'c'), ('b', 'c', 'd')]
    """
    if len(sequence) < min_ngram_size:
        return []

    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
        The first 4 bytes (32 bits) of the SHA1 hash of the input data.

    Examples
    --------
    >>> sha1_hash32(b"hello")
    499578026
    >>> bin(sha1_hash32(b"hello"))
    '0b11101110001101111010010101010'
    >>> sha1_hash32(b"hello world").bit_length()
    30
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]

from typing import Set

def tokenize(content: str, ngram_size: int, min_ngram_size: int) -> Set[str]:
    tokens = {
        " ".join(t)
        for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)
    }
    return tokens


def hash_content(content: str, num_perm: int, ngram_size: int, min_ngram_size: int, permutations: np.ndarray):
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokens = tokenize(content, ngram_size, min_ngram_size)
    hv = np.array(
        [sha1_hash32(token.encode("utf-8")) for token in tokens], dtype=np.uint64
    )
    a, b = permutations
    phv = np.bitwise_and(
        ((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH
    )
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    return hashvalues


# requirements: pyspark

# Connected Components in MapReduce and Beyond
def large_star_map(edge):
    return [(edge[0], edge[1]), (edge[1], edge[0])]


def large_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n > x]


def small_star_map(edge):
    x, y = edge
    if y <= x:
        return (x, y)
    else:
        return (y, x)


def small_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n != minimum]




def generate_hash_values(
    content: str,
    idx: int,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
    min_ngram_size: int,
) -> List[Tuple[int, bytes, int]]:
    """
    Generate the MinHashLSH values for a given document.

    Parameters
    ----------
    content : str
        The content of the document.
    idx : int
        The index of the document.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of the n-grams.
    hashranges : list
        The ranges of offsets for each hash value.
    permutations : np.ndarray
        The permutations for the hash values.
    min_ngram_size : int
        The minimum number of items in the sequence to generate n-grams.

    Returns
    -------
    List[Tuple[int, bytes, int]]
        The list of (band_idx, hash value, idx) for the document.
    """

    hashvalues = hash_content(content, num_perm, ngram_size, min_ngram_size, permutations)
    
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return [(band_idx, H, idx) for band_idx, H in enumerate(Hs)]


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.

    Examples
    --------
    >>> optimal_param(0.7, 256)
    (25, 10)
    """

    def false_positive_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


def generate_edges(nodes: List[int]) -> List[Tuple[int, int]]:
    """
    Generate edges from a cluster. Instead of generating N^2 edges, we only need all nodes align to a single node, since
    we will be running connected components on the edges later.

    Parameters
    ----------
    nodes : List[int]
        The list of nodes in the cluster.

    Returns
    -------
    List[Tuple[int, int]]
        The list of edges.
    """
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]


def jaccard_hash(hashvalues_a, hashvalues_b) -> float:
    return float(np.count_nonzero(hashvalues_a == hashvalues_b)) / float(
            len(hashvalues_a)
        )

class Document:
    def __init__(self, key, hashvalues):
        self.key = key
        self.hashvalues = hashvalues

class DocumentPair:
    def __init__(self, doc1, doc2):
        self.doc1 = doc1
        self.doc2 = doc2

class SimilarityPair:
    def __init__(self, pair, similarity):
        self.pair = pair
        self.similarity = similarity

def calculate_pair_similarity(doc_pair):
    """Calculate Jaccard similarity between document pair."""
    return jaccard_hash(doc_pair.doc1.hashvalues, doc_pair.doc2.hashvalues)

def deduplicate(edges, df):
    a = edges
    while True:
        b = (
            a.flatMap(large_star_map)
            .groupByKey()
            .flatMap(large_star_reduce)
            .distinct()
            .cache()
        )
        a = (
            b.map(small_star_map)
            .groupByKey()
            .flatMap(small_star_reduce)
            .distinct()
            .cache()
        )
        changes = a.subtract(b).union(b.subtract(a)).collect()
        if len(changes) == 0:
            break

    results = a.collect()
    if len(results) == 0:
        log.info("No components found.")
        return df, 0
        
    # Count the distinct duplicate groups
    duplicate_sets = len(set(r[1] for r in results))
    log.info(f"Found {duplicate_sets} distinct duplicate sets")

    components = spark.createDataFrame(results, schema=["__id__", "component"]).sort(
        ["component", "__id__"]
    )
    components.show()
    df = df.join(components, on="__id__", how="left")
    deduplicated_df = df.filter(F.col("component").isNull()).drop("__id__", "component").cache()
    duplicates_count = df.count() - deduplicated_df.count()
    return deduplicated_df, duplicates_count



def find_matching_pairs(candidate_pairs_rdd, similarity_threshold=0.8):
    """Find pairs with similarity above threshold."""
    # Calculate similarity scores for each pair
    pairs_with_similarity = candidate_pairs_rdd.map(
        lambda pair: SimilarityPair(pair, calculate_pair_similarity(pair))
    )

    # Filter pairs above similarity threshold
    matching_pairs = pairs_with_similarity.filter(
        lambda sim_pair: sim_pair.similarity > similarity_threshold
    )

    # Extract just the document keys from the pairs
    document_pairs = matching_pairs.map(
        lambda sim_pair: {sim_pair.pair.doc1.key, sim_pair.pair.doc2.key}
    )

    return document_pairs

def generate_minhash_signatures_brute(corpus_rdd, num_perm, ngram_size, min_ngram_size, permutations):
    return corpus_rdd.map(lambda x: (x[0], hash_content(x[1], num_perm, ngram_size, min_ngram_size, permutations)))


from src.tfidf_vec import tfidf_minhash
def minhash_lsh(df, column, num_perm, ngram_size, min_ngram_size, threshold):
    B, R = optimal_param(threshold, num_perm)
    log.info(f"Using optimal parameters: {B=}, {R=}")
    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    PERMUTATIONS = np.array(
        [
            (
                RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(num_perm)
        ],
        dtype=np.uint64,
    ).T

    # Get original record count
    original_count = df.count()
    
    df = df.withColumn("__id__", F.monotonically_increasing_id()).cache()
    records = df.select("__id__", column).rdd
    records = records.repartition(num_perm * 2).cache()

    edges = (
        records.flatMap(
            lambda x: generate_hash_values(
                content=x[1],
                idx=x[0],
                num_perm=num_perm,
                ngram_size=ngram_size,
                hashranges=HASH_RANGES,
                permutations=PERMUTATIONS,
                min_ngram_size=min_ngram_size,
            )
        )
        .groupBy(lambda x: (x[0], x[1]))
        .flatMap(lambda x: generate_edges([i[2] for i in x[1]]))
        .distinct()
        .cache()
    )
    # Count records after deduplication
    deduplicated_df, duplicate_count = deduplicate(edges, df)
    dedup_count = deduplicated_df.count()
    duplicate_count = original_count - dedup_count
    
    
    return deduplicated_df, duplicate_count



def create_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Near-deduplicating data with PySpark"
    )
    parser.add_argument(
        "--table", type=str, help="BigQuery table to deduplicate"
    )
    parser.add_argument(
        "--input_file", type=str, help="Local file to deduplicate (CSV or Parquet)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Similarity threshold"
    )
    parser.add_argument(
        "--min_ngram_size", type=int, default=5, help="Shorter docs will be removed"
    )
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size")
    parser.add_argument(
        "--num_perm", type=int, default=256, help="Number of permutations"
    )

    parser.add_argument(
        "--column", "-c", type=str, default="text", help="Column to deduplicate"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--limit_files", type=int, default=None, help="Limit the number of files to process"
    )
    parser.add_argument(
        "--implementation", type=str, default="minhash_lsh", help="Implementation to use"
    )
    parser.add_argument(
        "--use_ray", type=bool, default=False, help="Use Ray for parallel execution"
    )
    parser.add_argument(
        "--mock", type=bool, default=False, help="Mock the execution"
    )
    args = parser.parse_args()

    # Ensure at least one input source is provided
    if args.table is None and args.input_file is None:
        parser.error("Either --table or --input_file must be provided")
    return args


def get_total_size_gb(files):
    """
    Calculate total size of files in gigabytes.
    
    Parameters:
        files: List[str] - List of file paths
        
    Returns:
        float: Total size in GB
    """
    total_bytes = sum(os.path.getsize(f) for f in files)
    return total_bytes / (1024 * 1024 * 1024)  # Convert bytes to GB



if __name__ == "__main__":
    # assert "POSTGRES_ADDRESS" in os.environ, "POSTGRES_ADDRESS must be set"
    # Check if output directory exists, if not create it
    args = create_parser()
    if not args.mock:
        if args.use_ray:
            import ray
            import raydp
            ray.init(address='auto')
            num_nodes = len([x for x in ray.nodes() if x["alive"]])
            spark = raydp.init_spark(
                    app_name="MinHashLSH",
                    num_executors=num_nodes,
                    executor_cores=235, # how many tasks the executor can run in parallel
                    executor_memory="100g",
                    configs = {
                            'spark.local.dir': '/dev/shm/pyspark_dir',  # TODO: move in arguements
                            'spark.debug.maxToStringFields': '100',
                            # 'spark.ray.raydp_spark_master.actor.resource.CPU': 0,
                            # 'spark.ray.raydp_spark_master.actor.resource.spark_master': 1,  # Force Spark driver related actor run on headnode
                            'spark.driver.memory': '64g',
                        })
            
        else:
            conf = SparkConf()
            conf.set("spark.app.name", "MinHashLSH")
            conf.set("spark.debug.maxToStringFields", "100")
            conf.set("spark.local.dir", "/dev/shm/pyspark_dir") #TODO: move in arguements
            conf.set("spark.driver.memory", "64g")
            conf.set("spark.executor.memory", "64g")
            spark = SparkSession.builder.config(conf=conf).getOrCreate()
            num_nodes=1
            
        log: Logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(__name__)  # type: ignore
        if not os.path.exists(args.output):
            os.makedirs(args.output)
            log.info(f"Created output directory: {args.output}")


        # Load data from either BigQuery or local file
        if args.table:
            df = spark.read.format("bigquery").option("table", args.table).load()
        else:
            file_extension = os.path.splitext(args.input_file.strip(".gz"))[1].lower()
            
            if file_extension == '.csv':
                df = spark.read.option("header", "true").csv(args.input_file)
                
            elif file_extension.endswith('.json'):
                input_file = args.input_file
                if args.limit_files is not None:
                    input_file = glob.glob(input_file)[:args.limit_files]
                    print(f"Processing {len(input_file)} files")
                    print(f"Total size: {get_total_size_gb(input_file):.2f} GB")
                    
                df = spark.read.json(input_file)
            elif file_extension in ['.parquet', '.pq']:
                df = spark.read.parquet(args.input_file)
            else:
                log.error(f"Unsupported file format: {file_extension}")
                sys.exit(1)
        
        
        import time
        
        # Track original record count
        original_count = df.count()
        
        start_time = time.time()
        # assert args.implementation == "minhash_lsh"
        if args.implementation == "minhash_lsh":
            df, duplicate_count = minhash_lsh(df, args.column, args.num_perm, args.ngram_size, args.min_ngram_size, args.threshold)
        elif args.implementation == "tfidf_minhash":
            df, duplicate_count = tfidf_minhash(df, args.column, args.num_perm, args.ngram_size, args.min_ngram_size, args.threshold)
        dedup_count = original_count-duplicate_count
        log.info(f"Original records: {original_count}, Deduplicated: {dedup_count}, Duplicates: {duplicate_count}")
        dedup_time = time.time() - start_time
        print(f"Deduplication took {dedup_time/60:.2f} minutes")
        
        start_time = time.time()
        df.write.option("maxRecordsPerFile", 300_000).option(
            "intermediateFormat", "orc"
        ).parquet(args.output, mode="overwrite")
        write_time = time.time() - start_time
        print(f"Writing output took {write_time/60:.2f} minutes")
        
        # Get final record count
        record_count = df.count()
        total_time = dedup_time + write_time
    else:
        duplicate_count=0
        record_count=0
        record_count=0
        total_time=0
    
    try:
        from src.db import init_db, get_session, BenchmarkRun
        
        engine = init_db()
        session = get_session(engine)
        
        # Calculate total size if using input files with limit
        total_size_gb = None
        if args.input_file and args.limit_files is not None:
            input_files = glob.glob(args.input_file)[:args.limit_files]
            total_size_gb = get_total_size_gb(input_files)
        
        benchmark = BenchmarkRun.create_from_args(
            session=session,
            args=args,
            duplicate_count=duplicate_count,
            record_count=record_count,
            execution_time=total_time,
            notes=args.implementation,
            limit_files=args.limit_files,
            total_size_gb=total_size_gb,
            num_nodes=num_nodes
        )
        
        print(f"Benchmark data saved with ID: {benchmark.id}")
    except Exception as e:
        print(f"Error saving benchmark data: {e}")

# /dev/shm/c4_files/*.json.gz
# python3.10 database_project/src/deduplication_spark.py --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --limit_files 1
#python3.10 database_project/src/deduplication_spark.py --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --limit_files 10