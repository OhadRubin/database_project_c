# /database_project/src/run_workflows.py
import os

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.10"

import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
from itertools import tee
# from logging import Logger
from typing import Iterable
from typing import List
from typing import Tuple
import json # For config serialization

from scipy.integrate import quad as integrate
import glob
import time
import logging
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ray_minhash import run_nd_step_for_workflow # Returns (ray_dataset, dupe_count, time)
from ray_tfidf_vec import run_cl_step_for_workflow # Returns (ds, dupe_count, train_time, infer_time, stage2_time, dist_json)



from ml_collections import config_dict
import yaml



def read_config(path):
    with open(path) as f:
        config_data = yaml.safe_load(f)
        cfg = config_dict.ConfigDict(config_data)
    return cfg

# --- Logging Setup ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# --- Utility Function ---
def get_total_size_gb(files):
    try:
        total_bytes = sum(os.path.getsize(f) for f in files)
        return total_bytes / (1024 * 1024 * 1024)
    except Exception as e:
        print(f"Could not calculate total size: {e}")
        return None

# --- Argument Parser ---
def create_parser():
    parser = argparse.ArgumentParser(
        description="Run Deduplication and Clustering Workflows"
    )
    parser.add_argument(
        "--workflow", type=str, required=True, choices=["nd_cl", "cl_nd"],
        help="Workflow to execute: 'nd_cl' (ND then CL) or 'cl_nd' (CL then ND within clusters)"
    )

    # --- Input/Output ---
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input file pattern (e.g., 'data/*.json.gz')"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Final output directory for this specific run"
    )
    parser.add_argument(
        "--limit_files", type=int, default=None, help="Limit the number of input files"
    )

    # --- ND Parameters (used in both workflows) ---
    parser.add_argument("--threshold", type=float, default=0.7, help="MinHash Similarity threshold")
    parser.add_argument("--min_ngram_size", type=int, default=5, help="Min N-gram size for ND")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size for ND")
    parser.add_argument("--num_perm", type=int, default=256, help="Number of permutations for MinHash")
    parser.add_argument("--column", "-c", type=str, default="text", help="Column name for text data")

    # --- CL Parameters ---
    parser.add_argument(
        "--config_file", type=str, default="database_project/src/configs/base.yml",
        help="Path to clustering config YAML"
    )


    # --- Benchmarking ---
    parser.add_argument("--notes", type=str, default=None, help="Notes for benchmark DB entry")
    # --implementation argument is now replaced by --workflow
    parser.add_argument(
        "--mock", type=bool, default=False, help="Mock the execution"
    )
    parser.add_argument(
        "--max_docs", type=int, default=50000, help="Number of documents to learn the clustering on"
    )
    parser.add_argument(
        "--mock_stage1", type=bool, default=False, help="Mock the execution"
    )
    # --- Ray MinHash Parameters ---
    parser.add_argument(
        "--union_find_parallel_num", type=int, default=400, 
        help="Number of parallel workers for union-find algorithm"
    )
    parser.add_argument(
        "--union_threshold", type=int, default=256,
        help="Threshold for minhash values group to perform union-find algorithm"
    )
    parser.add_argument(
        "--max_pending_edge_buffer_task", type=int, default=20,
        help="Max number of pending edge buffer ray tasks"
    )
    parser.add_argument(
        "--num_edge_buffer_task_returns", type=int, default=10,
        help="Number of edge buffer tasks for ray.wait to return"
    )
    parser.add_argument(
        "--max_pending_filter_tasks", type=int, default=20,
        help="Max number of pending filter ray tasks"
    )
    parser.add_argument(
        "--num_filter_task_returns", type=int, default=10,
        help="Number of filter tasks for ray.wait to return"
    )
    parser.add_argument(
        "--merge_batch_size", type=int, default=100,
        help="Batch size for merging operations"
    )
    parser.add_argument(
        "--dedup_mode", type=str, default="filter",choices=["filter", "tag"],
        help="Deduplication mode: 'filter' or 'tag'"
    )

    return parser

# --- Main Execution Logic ---
if __name__ == "__main__":
    args = create_parser().parse_args()
    workflow_start_time = time.time()
    print(f"Starting workflow: {args.workflow} with args: {args}")

    # --- Variables for storing results across stages/workflows ---
    final_output_path = args.output
    final_record_count = 0
    total_duplicate_count = 0
    num_nodes_used = 1  # Default, will be updated
    actual_workflow_time = 0
    nd_step_time = 0.0
    nd_output_record_count = None # Specific for ND->CL
    cl_train_time = 0.0
    cl_inference_time = 0.0
    cl_stage2_time = 0.0
    config_details_json = "{}"

    # --- Initialize Ray ---
    import ray
    ray.init(address='auto',
                dashboard_host="0.0.0.0",
                # log_to_driver=False # Keep logs separate per node if needed
                ignore_reinit_error=True # Allow re-initialization if already connected
                )
    num_nodes_used = len([x for x in ray.nodes() if x["alive"]])


    # --- Load Configs ---
    cfg = read_config(args.config_file)
    cfg.args = args # Make args accessible within config if needed by downstream funcs

    # Prepare full config details for logging
    config_details = {
        "args": vars(args),
        "clustering_config": cfg.to_dict() # Convert ConfigDict to dict for JSON
    }
    config_details_json = json.dumps(config_details, indent=2, default=str) # Use default=str for non-serializable types



    # --- Load Data ---
    try:
        data_load_start = time.time()
        input_files = glob.glob(args.input_file)
        if not input_files:
            print(f"No files found matching input pattern: {args.input_file}")
            sys.exit(1)

        if args.limit_files is not None and args.limit_files > 0:
            input_files = input_files[:args.limit_files]
            print(f"Limited input to {len(input_files)} files.")
        elif args.limit_files is not None and args.limit_files <= 0:
             print(f"limit_files is {args.limit_files}, processing all found files.")

        total_size_gb = get_total_size_gb(input_files)
        print(f"Reading {len(input_files)} files (Total size: {total_size_gb:.2f} GB)...")

        ray_df = ray.data.read_json(input_files, override_num_blocks=cfg.num_blocks)
        # It's good practice to materialize early if memory allows, or before complex ops
        # ray_df = ray_df.materialize()
        print(f"Data loaded into Ray Dataset in {time.time() - data_load_start:.2f} seconds.")

    except Exception as e:
        print(f"Error loading data from {args.input_file}: {e}", exc_info=True)
        sys.exit(1)


    # --- Execute Workflow ---
    try:
        if args.workflow == "nd_cl":
            print("Executing ND -> CL workflow...")
            # === Stage 1: ND ===
            print("Running ND step...")
            nd_start_time = time.time()
            intermediate_ray_ds, nd_duplicates, nd_step_time = run_nd_step_for_workflow(ray_df, args)
            nd_end_time = time.time()
            nd_step_time = nd_end_time - nd_start_time # More accurate timing
            print(f"ND step completed in {nd_step_time:.2f}s. Found {nd_duplicates} duplicates.")

            nd_output_record_count = intermediate_ray_ds.count() # Capture count after ND
            total_duplicate_count = nd_duplicates
            print(f"Record count after ND: {nd_output_record_count}")

            # Prepare for CL step
            intermediate_ray_ds = intermediate_ray_ds.repartition(cfg.num_blocks).materialize()
            cfg.base_stage.should_dedup = False # Ensure CL step doesn't dedup again

            # === Stage 2: CL ===
            print("Running CL step...")
            cl_start_time = time.time()
            # CL step now returns: ds, dupe_count(0), train_t, infer_t, stage2_t(0), dist_json
            clustered_ds, metric_list = run_cl_step_for_workflow(intermediate_ray_ds, cfg)
            cl_end_time = time.time()
            
            cl_train_time = metric_list[0]["train_time"]
            cl_inference_time = metric_list[0]["inference_time"]
            cl_stage2_time = metric_list[1]["total_time"]
            
            print(f"CL step completed in {cl_end_time - cl_start_time:.2f}s.")
            print(f"  CL Train Time: {cl_train_time:.2f}s")
            print(f"  CL Inference Time: {cl_inference_time:.2f}s")
            # final_record_count is the count after ND in this workflow
            final_record_count = nd_output_record_count


        elif args.workflow == "cl_nd":
            print("Executing CL -> ND workflow...")
            # Set the deduplication flag for this workflow
            cfg.base_stage.should_dedup = True

            # === Stage 1+2: CL+ND ===
            print("Running CL -> ND step...")
            cl_nd_start_time = time.time()
            # CL step now returns: ds, dupe_count, train_t, infer_t, stage2_t, dist_json

            clustered_ds, metric_list = run_cl_step_for_workflow(ray_df, cfg)
            cl_nd_end_time = time.time()
            
            cl_train_time = metric_list[0]["train_time"]
            cl_inference_time = metric_list[0]["inference_time"]
            cl_stage2_time = metric_list[1]["total_time"]
            total_duplicate_count = metric_list[-1]["n_duplicates"]

            final_record_count = clustered_ds.count()  # Calculate final count *after* the step
            print(f"CL->ND workflow completed in {cl_nd_end_time - cl_nd_start_time:.2f}s.")
            print(f"  Total duplicates found across clusters: {total_duplicate_count}")
            print(f"  Final record count: {final_record_count}")
            print(f"  CL Train Time (Agg): {cl_train_time:.2f}s")
            print(f"  CL Inference Time (Agg): {cl_inference_time:.2f}s")
            print(f"  CL Stage2 Time: {cl_stage2_time:.2f}s")


        else:
            # Should not happen due to argparse choices
            raise ValueError(f"Invalid workflow specified: {args.workflow}")

        # --- Workflow Complete - Final Benchmarking ---
        actual_workflow_time = time.time() - workflow_start_time
        print(f"Workflow '{args.workflow}' finished. Total wall clock time: {actual_workflow_time:.2f} seconds.")


        benchmark_notes = args.notes if args.notes else f"Workflow: {args.workflow}"
        benchmark_notes += f" | CL Cfg: {os.path.basename(args.config_file)}" # Always add config file

        from db import init_db, get_session, BenchmarkRun

        engine = init_db()
        session = get_session(engine)

        # Create the main benchmark entry
        benchmark_run = BenchmarkRun(
            input_file=args.input_file, # Log the pattern
            output_dir=args.output,
            notes=benchmark_notes,
            duplicate_count=total_duplicate_count, # Meaning depends on workflow
            record_count=final_record_count,       # Final count after all steps
            implementation=args.workflow,          # Use workflow name as implementation
            num_nodes=num_nodes_used,              # Max nodes used during the workflow
            threshold=args.threshold,
            ngram_size=args.ngram_size,
            min_ngram_size=args.min_ngram_size,
            num_perm=args.num_perm,
            execution_time=actual_workflow_time,   # Total wall clock time
            limit_files=args.limit_files,          # Log the limit used
            total_size_gb=total_size_gb,           # Log calculated size
            # New fields
            nd_time_sec=nd_step_time,
            nd_output_count=nd_output_record_count, # Will be None for CL->ND
            config_file_path=args.config_file,
            cl_train_time_sec=cl_train_time,
            cl_inference_time_sec=cl_inference_time,
            cl_stage2_time_sec=cl_stage2_time,
            config_details_json=config_details_json,
        )
        session.add(benchmark_run)

        # Add resource/accuracy metrics here if they were collected
        # e.g., benchmark_run.add_resource_metrics(...)
        # e.g., benchmark_run.add_accuracy_metrics(...)

        # Commit everything related to this run
        session.commit()
        print(f"Benchmark data saved with ID: {benchmark_run.id}")
        session.close()

    except Exception as e:
        print(f"Workflow '{args.workflow}' failed: {e}", exc_info=True)
        # Consider logging a failed run marker to the DB if desired
        sys.exit(1)

    finally:
        # Optional: Shutdown Ray explicitly if needed, otherwise it might persist
        # if ray.is_initialized():
        #     print("Shutting down Ray...")
        #     ray.shutdown()
        pass

    print(f"Workflow {args.workflow} completed successfully.")