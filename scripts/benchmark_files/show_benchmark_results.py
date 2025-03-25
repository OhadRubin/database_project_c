#!/usr/bin/env python3.10
import sys
import os
import argparse
from datetime import datetime
from tabulate import tabulate

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db import init_db, get_session, BenchmarkRun, ResourceMetric, AccuracyMetric

def parse_args():
    parser = argparse.ArgumentParser(description="Display benchmark results")
    parser.add_argument(
        "--db_path", type=str, default="sqlite:///benchmark_results.db",
        help="Database URI (default: sqlite:///benchmark_results.db)"
    )
    parser.add_argument(
        "--id", type=int, default=None,
        help="Show details for a specific benchmark run ID"
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Limit number of results (default: 10)"
    )
    parser.add_argument(
        "--implementation", type=str, default=None,
        help="Filter by implementation type"
    )
    parser.add_argument(
        "--compare", nargs='+', type=int, default=None,
        help="Compare multiple benchmark run IDs"
    )
    parser.add_argument(
        "--sort", type=str, default="timestamp",
        choices=["timestamp", "execution_time", "record_count", "duplicate_count", "threshold"],
        help="Sort results by field (default: timestamp)"
    )
    parser.add_argument(
        "--desc", action="store_true",
        help="Sort in descending order"
    )
    return parser.parse_args()

def show_benchmark_run(session, run_id):
    """Show details for a specific benchmark run"""
    run = session.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
    if not run:
        print(f"Benchmark run with ID {run_id} not found")
        return
    
    print(f"\n{'='*80}")
    print(f"Benchmark Run #{run.id}")
    print(f"{'='*80}")
    print(f"Timestamp: {run.timestamp}")
    print(f"Implementation: {run.implementation}")
    print(f"Input file: {run.input_file}")
    print(f"Output directory: {run.output_dir}")
    print(f"Number of nodes: {run.num_nodes}")
    
    print(f"\n--- Parameters ---")
    print(f"Threshold: {run.threshold}")
    print(f"N-gram size: {run.ngram_size}")
    print(f"Min N-gram size: {run.min_ngram_size}")
    print(f"Number of permutations: {run.num_perm}")
    
    print(f"\n--- Results ---")
    print(f"Execution time: {run.execution_time:.2f} seconds")
    print(f"Records after deduplication: {run.record_count}")
    print(f"Duplicate sets found: {run.duplicate_count}")
    
    # Show resource metrics if available
    resource_metrics = session.query(ResourceMetric).filter(ResourceMetric.result_id == run_id).first()
    if resource_metrics:
        print(f"\n--- Resource Metrics ---")
        print(f"CPU usage (avg/max): {resource_metrics.cpu_percent_avg:.1f}% / {resource_metrics.cpu_percent_max:.1f}%")
        print(f"Memory usage (avg/max): {resource_metrics.memory_usage_avg_mb:.1f}MB / {resource_metrics.memory_usage_max_mb:.1f}MB")
        print(f"Network (sent/recv): {resource_metrics.network_sent_mb:.1f}MB / {resource_metrics.network_recv_mb:.1f}MB")
        print(f"Disk I/O (read/write): {resource_metrics.disk_read_mb:.1f}MB / {resource_metrics.disk_write_mb:.1f}MB")
    
    # Show accuracy metrics if available
    accuracy_metrics = session.query(AccuracyMetric).filter(AccuracyMetric.result_id == run_id).first()
    if accuracy_metrics:
        print(f"\n--- Accuracy Metrics ---")
        print(f"Reference implementation: {accuracy_metrics.reference_implementation}")
        print(f"True positives: {accuracy_metrics.true_positives}")
        print(f"False positives: {accuracy_metrics.false_positives}")
        print(f"False negatives: {accuracy_metrics.false_negatives}")
        print(f"Precision: {accuracy_metrics.precision:.4f}")
        print(f"Recall: {accuracy_metrics.recall:.4f}")
        print(f"F1 score: {accuracy_metrics.f1_score:.4f}")
    
    if run.notes:
        print(f"\n--- Notes ---")
        print(run.notes)
    
    print(f"{'='*80}\n")

def compare_benchmark_runs(session, run_ids):
    """Compare multiple benchmark runs"""
    runs = session.query(BenchmarkRun).filter(BenchmarkRun.id.in_(run_ids)).all()
    if not runs:
        print("No benchmark runs found with the specified IDs")
        return
    
    # Basic info
    headers = ["ID", "Implementation", "Execution Time (s)", "Records", "Duplicates", 
               "Threshold", "N-gram", "Min N-gram", "# Perm"]
    rows = []
    
    for run in runs:
        rows.append([
            run.id,
            run.implementation,
            f"{run.execution_time:.2f}",
            run.record_count,
            run.duplicate_count,
            run.threshold,
            run.ngram_size,
            run.min_ngram_size,
            run.num_perm
        ])
    
    print("\nComparison of Benchmark Runs")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Compare resource metrics if available
    resource_metrics = session.query(ResourceMetric).filter(ResourceMetric.result_id.in_(run_ids)).all()
    if resource_metrics:
        headers = ["Run ID", "CPU Avg (%)", "CPU Max (%)", "Mem Avg (MB)", 
                  "Mem Max (MB)", "Network (MB)", "Disk I/O (MB)"]
        rows = []
        
        for metric in resource_metrics:
            rows.append([
                metric.result_id,
                f"{metric.cpu_percent_avg:.1f}",
                f"{metric.cpu_percent_max:.1f}",
                f"{metric.memory_usage_avg_mb:.1f}",
                f"{metric.memory_usage_max_mb:.1f}",
                f"{metric.network_sent_mb + metric.network_recv_mb:.1f}",
                f"{metric.disk_read_mb + metric.disk_write_mb:.1f}"
            ])
        
        print("\nResource Metrics Comparison")
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Compare accuracy metrics if available
    accuracy_metrics = session.query(AccuracyMetric).filter(AccuracyMetric.result_id.in_(run_ids)).all()
    if accuracy_metrics:
        headers = ["Run ID", "Reference", "TP", "FP", "FN", "Precision", "Recall", "F1 Score"]
        rows = []
        
        for metric in accuracy_metrics:
            rows.append([
                metric.result_id,
                metric.reference_implementation,
                metric.true_positives,
                metric.false_positives,
                metric.false_negatives,
                f"{metric.precision:.4f}",
                f"{metric.recall:.4f}",
                f"{metric.f1_score:.4f}"
            ])
        
        print("\nAccuracy Metrics Comparison")
        print(tabulate(rows, headers=headers, tablefmt="grid"))

def list_benchmark_runs(session, limit=10, implementation=None, sort_by="timestamp", desc=False):
    """List benchmark runs with optional filtering and sorting"""
    query = session.query(BenchmarkRun)
    
    if implementation:
        query = query.filter(BenchmarkRun.implementation == implementation)
    
    # Apply sorting
    order_col = getattr(BenchmarkRun, sort_by)
    if desc:
        query = query.order_by(order_col.desc())
    else:
        query = query.order_by(order_col)
    
    runs = query.limit(limit).all()
    
    if not runs:
        print("No benchmark runs found")
        return
    
    headers = ["ID", "Timestamp", "Implementation", "Execution Time (s)", 
               "Records", "Duplicates", "Threshold", "Input"]
    rows = []
    
    for run in runs:
        # Format timestamp and truncate input file path if too long
        timestamp = run.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        input_file = run.input_file
        if input_file and len(input_file) > 30:
            input_file = "..." + input_file[-27:]
        
        rows.append([
            run.id,
            timestamp,
            run.implementation,
            f"{run.execution_time:.2f}" if run.execution_time else "-",
            run.record_count if run.record_count else "-",
            run.duplicate_count if run.duplicate_count else "-",
            run.threshold if run.threshold else "-",
            input_file
        ])
    
    print("\nBenchmark Runs")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print(f"\nShowing {len(runs)} of {session.query(BenchmarkRun).count()} total runs")

if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Initialize database connection
        engine = init_db(args.db_path)
        session = get_session(engine)
        
        if args.id:
            # Show details for a specific benchmark run
            show_benchmark_run(session, args.id)
        elif args.compare:
            # Compare multiple benchmark runs
            compare_benchmark_runs(session, args.compare)
        else:
            # List benchmark runs with optional filtering
            list_benchmark_runs(
                session, 
                limit=args.limit, 
                implementation=args.implementation,
                sort_by=args.sort,
                desc=args.desc
            )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 