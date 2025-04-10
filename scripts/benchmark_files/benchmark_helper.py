#!/usr/bin/env python3.10
import sys
import os
import time
import multiprocessing
import threading
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db import (
    init_db, get_session, BenchmarkRun, 
    ResourceMetric, AccuracyMetric, monitor_resources
)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark helper for deduplication runs")
    parser.add_argument(
        "--db_path", type=str, default="sqlite:///benchmark_results.db",
        help="Database URI (default: sqlite:///benchmark_results.db)"
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Monitor system resources during benchmark"
    )
    parser.add_argument(
        "--notes", type=str, default=None,
        help="Additional notes about this benchmark run"
    )
    parser.add_argument(
        "--implementation", type=str, default="pyspark",
        help="Implementation type (default: pyspark)"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1,
        help="Number of nodes in the cluster (default: 1)"
    )
    
    # Pass remaining arguments to the deduplication script
    args, remaining = parser.parse_known_args()
    return args, remaining

def count_records(output_dir):
    """Count records in output parquet files"""
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder.appName("RecordCounter").getOrCreate()
        df = spark.read.parquet(output_dir)
        count = df.count()
        spark.stop()
        return count
    except Exception as e:
        print(f"Error counting records: {e}")
        return 0

def run_benchmark(args, dedup_args):
    """
    Run the benchmark and log results to database
    
    Parameters:
    -----------
    args : argparse.Namespace
        Arguments for this script
    dedup_args : list
        Arguments to pass to deduplication_spark.py
    """
    # Initialize database
    engine = init_db(args.db_path)
    session = get_session(engine)
    
    # Parse deduplication args
    from old.deduplication_spark import create_parser
    dedup_parser = create_parser()
    dedup_args_ns = dedup_parser.parse_args(dedup_args)
    
    # Record start time
    start_time = time.time()
    
    # Create a dictionary to store run stats
    stats = {
        'duplicate_count': 0,
        'record_count': 0,
        'monitor_thread': None,
        'benchmark_run': None
    }
    
    # Start resource monitoring in a separate thread if requested
    if args.monitor:
        def start_monitoring():
            # Create a temporary entry to get an ID
            temp_run = BenchmarkRun(
                input_file=dedup_args_ns.input_file or dedup_args_ns.table or '',
                output_dir=dedup_args_ns.output,
                implementation=args.implementation,
                num_nodes=args.num_nodes,
                notes=f"TEMP: {args.notes}"
            )
            session.add(temp_run)
            session.commit()
            
            stats['benchmark_run'] = temp_run
            # Start monitoring
            monitor_resources(temp_run.id, session)
        
        stats['monitor_thread'] = threading.Thread(target=start_monitoring)
        stats['monitor_thread'].daemon = True
        stats['monitor_thread'].start()
    
    # Run the deduplication script
    command = [sys.executable, "database_project/src/deduplication_spark.py"] + dedup_args
    print(f"Running: {' '.join(command)}")
    
    try:
        import subprocess
        process = subprocess.run(command, check=True)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Count records in output
        if os.path.exists(dedup_args_ns.output):
            stats['record_count'] = count_records(dedup_args_ns.output)
            
            # Try to calculate duplicate count from original file
            # This is approximate and may need adjustment based on how
            # your deduplication code reports duplicates
            try:
                from pyspark.sql import SparkSession
                
                spark = SparkSession.builder.appName("DuplicateCounter").getOrCreate()
                if dedup_args_ns.input_file:
                    if dedup_args_ns.input_file.endswith('.csv'):
                        input_df = spark.read.option("header", "true").csv(dedup_args_ns.input_file)
                    elif dedup_args_ns.input_file.endswith(('.json', '.json.gz')):
                        input_df = spark.read.json(dedup_args_ns.input_file)
                    elif dedup_args_ns.input_file.endswith(('.parquet', '.pq')):
                        input_df = spark.read.parquet(dedup_args_ns.input_file)
                    
                    original_count = input_df.count()
                    stats['duplicate_count'] = original_count - stats['record_count']
                spark.stop()
            except Exception as e:
                print(f"Error calculating duplicate count: {e}")
        
        # If monitoring thread is running, stop it
        if args.monitor and stats['monitor_thread'] and stats['monitor_thread'].is_alive():
            print("Press Ctrl+C in the monitoring thread to stop collection...")
        
        # Create or update benchmark run entry
        if stats['benchmark_run']:
            # Update the existing entry
            run = stats['benchmark_run']
            run.duplicate_count = stats['duplicate_count']
            run.record_count = stats['record_count']
            run.threshold = dedup_args_ns.threshold
            run.ngram_size = dedup_args_ns.ngram_size
            run.min_ngram_size = dedup_args_ns.min_ngram_size
            run.num_perm = dedup_args_ns.num_perm
            run.execution_time = execution_time
            run.notes = args.notes
            session.commit()
        else:
            # Create a new entry
            run = BenchmarkRun.create_from_args(
                session, 
                dedup_args_ns,
                duplicate_count=stats['duplicate_count'],
                record_count=stats['record_count'],
                execution_time=execution_time,
                num_nodes=args.num_nodes,
                notes=args.notes,
                implementation=args.implementation
            )
        
        print(f"\nBenchmark completed in {execution_time:.2f} seconds")
        print(f"Results saved to database with ID: {run.id}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running deduplication: {e}")
        if stats['benchmark_run']:
            # Clean up failed run
            session.delete(stats['benchmark_run'])
            session.commit()

if __name__ == "__main__":
    args, remaining = parse_args()
    run_benchmark(args, remaining) 