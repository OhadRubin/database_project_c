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

from scipy.integrate import quad as integrate
import glob
import time
import logging
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ray_minhash import run_nd_step_for_workflow # Returns (ray_dataset, dupe_count, nodes, time)
from ray_tfidf_vec import run_cl_step_for_workflow

# --- Import Modified Core Logic Functions ---
# These functions are assumed to be modified to handle in-memory Ray Datasets
# and avoid intermediate disk writes between ND and CL steps.




from ml_collections import config_dict
import yaml



def read_config(path):
    with open(path) as f:
        config_data = yaml.safe_load(f)
        cfg = config_dict.ConfigDict(config_data)
    return cfg

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Function ---
def get_total_size_gb(files):
    try:
        total_bytes = sum(os.path.getsize(f) for f in files)
        return total_bytes / (1024 * 1024 * 1024)
    except Exception as e:
        logger.warning(f"Could not calculate total size: {e}")
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
        "--output", "-o", type=str, required=True, help="Final output directory"
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

    # --- Execution Environment ---
    parser.add_argument(
        "--use_ray", type=bool, default=True,
        help="Use RayDP for Spark ND step in 'nd_cl' workflow (Recommended: True)"
    ) # CL->ND workflow inherently uses Ray.

    # --- Benchmarking ---
    parser.add_argument("--notes", type=str, default=None, help="Notes for benchmark DB entry")
    parser.add_argument("--db_uri", type=str, default=None, help="Database URI (uses DB_URI env var or sqlite default if None)")
    # --implementation argument is now replaced by --workflow
    parser.add_argument(
        "--mock", type=bool, default=False, help="Mock the execution"
    )

    return parser

# --- Main Execution Logic ---
if __name__ == "__main__":
    args = create_parser().parse_args()
    workflow_start_time = time.time()
    logger.info(f"Starting workflow: {args.workflow}")

    # --- Variables for storing results across stages/workflows ---
    final_output_path = args.output
    final_record_count = 0
    total_duplicate_count = 0
    num_nodes_used = 1  # Default, will be updated
    actual_workflow_time = 0

    # --- Initialize Ray ---
    # Needs careful handling if Spark also uses RayDP
    import ray
    ray.init(address='auto', 
             dashboard_host="0.0.0.0"
            #  log_to_driver=False
             )
    num_nodes_used = len([x for x in ray.nodes() if x["alive"]])
    
    cfg = read_config(args.config_file)
    cfg.args = args

    start_time = time.time()
    input_file = glob.glob(args.input_file)[:args.limit_files]    
    ray_df = ray.data.read_json(input_file, override_num_blocks=cfg.num_blocks)



    try:
        # --- Execute Selected Workflow ---
        if args.workflow == "nd_cl":
            logger.info("Executing ND -> CL workflow...")
            # === Stage 1: ND ===
            logger.info("Running ND step...")
            
            
            intermediate_ray_ds, nd_duplicates, nd_time = run_nd_step_for_workflow(ray_df, args)
            final_record_count = intermediate_ray_ds.count()
            total_duplicate_count = nd_duplicates
            
            intermediate_ray_ds = intermediate_ray_ds.repartition(1000)


                
            # === Stage 2: CL ===
            logger.info("Running CL step...")

            # final_output_path, cl_time = run_cl_step_for_workflow(intermediate_ray_ds, cfg, args.output)
            start_time = time.time()
            clustered_ds = run_cl_step_for_workflow(intermediate_ray_ds, cfg)
            cl_time = time.time() - start_time
            logger.info(f"CL step completed in {cl_time:.2f}s. Final output: {final_output_path}")
            workflow_total_time = nd_time + cl_time # This is approximate, wall clock is better

        elif args.workflow == "cl_nd":
            clustered_ds = run_cl_step_for_workflow(ray_df, cfg)
            assert False, "Not implemented"

        else:
            # Should not happen due to argparse choices
            raise ValueError(f"Invalid workflow specified: {args.workflow}")

        # --- Workflow Complete - Final Benchmarking ---
        actual_workflow_time = time.time() - workflow_start_time
        logger.info(f"Workflow '{args.workflow}' finished. Total wall clock time: {actual_workflow_time:.2f} seconds.")


        benchmark_notes = args.notes if args.notes else f"Workflow: {args.workflow}"
        if args.workflow == 'cl_nd':
             benchmark_notes += f" (CL Cfg: {os.path.basename(args.config_file)})"

        from db import init_db, get_session, BenchmarkRun
        
        engine = init_db()
        session = get_session(engine)
        
        benchmark_run = BenchmarkRun.create_from_args(
            session=session,
            args=args,
            duplicate_count=total_duplicate_count, # Meaning depends on workflow
            record_count=final_record_count,       # Final count after all steps
            execution_time=actual_workflow_time,   # Total wall clock time
            implementation=args.workflow,          # Use workflow name as implementation
            num_nodes=num_nodes_used,              # Max nodes used during the workflow
            notes=benchmark_notes,
            limit_files=args.limit_files,          # Log the limit used
            total_size_gb=0            # Log calculated size
        )
        logger.info(f"Benchmark data saved with ID: {benchmark_run.id}")

    except Exception as e:
        logger.error(f"Workflow '{args.workflow}' failed: {e}", exc_info=True)
        # Consider logging a failed run marker to the DB if desired
        sys.exit(1)

    finally:
        # Optional: Shutdown Ray if this script was the main driver
        # Be cautious if running within a larger Ray application/cluster
        # if ray.is_initialized():
        #     logger.info("Shutting down Ray...")
        #     ray.shutdown()
        pass

    logger.info(f"Workflow {args.workflow} completed successfully.")