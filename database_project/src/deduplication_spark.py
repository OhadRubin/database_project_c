import os

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.10"

import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
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


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



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
    total_bytes = sum(os.path.getsize(f) for f in files)
    return total_bytes / (1024 * 1024 * 1024)  # Convert bytes to GB




from minhash import minhash_lsh
from ray_tfidf_vec import tfidf_minhash_ray

if __name__ == "__main__":
    args = create_parser()
    if args.implementation == "minhash_lsh":
        record_count, total_time, num_nodes, duplicate_count = minhash_lsh(args)
    elif args.implementation == "tfidf_minhash_ray":
        record_count, total_time, num_nodes, duplicate_count = tfidf_minhash_ray(args)
    else:
        assert False, f"Implementation {args.implementation} not supported"

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
# python3.10 database_project/src/deduplication_spark.py --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --limit_files 1 --implementation tfidf_minhash
#python3.10 database_project/src/deduplication_spark.py --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --limit_files 10