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


tfidf_minhash = None  # This will be populated after adding the file to SparkContext
minhash_lsh = None  # This will be populated after adding the file to SparkContext

if __name__ == "__main__":

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
                            "spark.driver.maxResultSize": "10g"
                        })
            
        else:
            conf = SparkConf()
            conf.set("spark.app.name", "MinHashLSH")
            conf.set("spark.debug.maxToStringFields", "100")
            conf.set("spark.local.dir", "/dev/shm/pyspark_dir") #TODO: move in arguements
            conf.set("spark.driver.memory", "64g")
            conf.set("spark.executor.memory", "64g")
            conf.set("spark.driver.maxResultSize", "10g")
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
        # Get the current file's directory to make paths relative
        # /home/ohadr/database_project_c/database_project/src/deduplication_spark.py
        current_dir = os.path.dirname(os.path.abspath(__file__))

        spark.sparkContext.addPyFile(os.path.join(current_dir, "tfidf_vec.py"))
        spark.sparkContext.addPyFile(os.path.join(current_dir, "minhash.py"))
        
        # Now we can import the function directly
        from tfidf_vec import tfidf_minhash
        from minhash import minhash_lsh
        
        # Track original record count
        original_count = df.count()
        
        start_time = time.time()
        # assert args.implementation == "minhash_lsh"
        if args.implementation == "minhash_lsh":
            df, duplicate_count = minhash_lsh(spark, df, args.column, args.num_perm, args.ngram_size, args.min_ngram_size, args.threshold)
        elif args.implementation == "tfidf_minhash":
            df, duplicate_count = tfidf_minhash(spark, df, args.column, args.num_perm, args.ngram_size, args.min_ngram_size, args.threshold)
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
# python3.10 database_project/src/deduplication_spark.py --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --limit_files 1 --implementation tfidf_minhash
#python3.10 database_project/src/deduplication_spark.py --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --limit_files 10