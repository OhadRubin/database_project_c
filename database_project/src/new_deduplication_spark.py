import os
import sys
import time
import glob
import argparse
import logging
import yaml
from ml_collections import config_dict

# --- Environment Setup (Keep as needed) ---
# os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.10"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.10"

# --- Add project root to Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# --- Import Modified Core Logic Functions ---
# These functions are assumed to be modified to handle in-memory Ray Datasets
# and avoid intermediate disk writes between ND and CL steps.
try:
    from database_project.src.minhash import run_nd_step_for_workflow # Returns (ray_dataset, dupe_count, nodes, time)
    from database_project.src.ray_tfidf_vec import (
        run_cl_step_for_workflow, # Takes (ray_dataset, cfg, out_path), returns (out_path, time)
        run_cl_nd_integrated_workflow, # Takes (args, cfg), returns (out_path, time, final_count, dupe_count, nodes)
        read_config
    )
except ImportError as e:
    print(f"Error importing core workflow functions: {e}")
    print("Ensure modified functions exist in minhash.py and ray_tfidf_vec.py")
    sys.exit(1)

# --- Import Benchmarking DB Logic ---
try:
    from database_project.src.db import init_db, get_session, BenchmarkRun
except ImportError as e:
    print(f"Error importing DB functions: {e}")
    sys.exit(1)

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
    try:
        import ray
        # Use ignore_reinit_error=True if Ray might be initialized elsewhere (e.g., by RayDP)
        if not ray.is_initialized():
             ray.init(address='auto', ignore_reinit_error=True)
             logger.info(f"Ray initialized: {ray.cluster_resources()}")
        else:
             logger.info(f"Ray already initialized: {ray.cluster_resources()}")
             num_nodes_used = len([x for x in ray.nodes() if x["alive"]])
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}", exc_info=True)
        sys.exit(1)


    try:
        # --- Execute Selected Workflow ---
        if args.workflow == "nd_cl":
            logger.info("Executing ND -> CL workflow...")

            # === Stage 1: ND ===
            logger.info("Running ND step...")
            intermediate_ray_ds, nd_duplicates, num_nodes_nd, nd_time = run_nd_step_for_workflow(args)
            if intermediate_ray_ds is None:
                raise RuntimeError("ND step failed to produce a Ray Dataset.")
            # Record count *after* ND
            final_record_count = intermediate_ray_ds.count()
            total_duplicate_count = nd_duplicates
            num_nodes_used = num_nodes_nd # Or max if CL uses different nodes
            logger.info(f"ND step completed in {nd_time:.2f}s. Records after ND: {final_record_count}, Duplicates found: {total_duplicate_count}")

            # === Stage 2: CL ===
            logger.info("Running CL step...")
            cfg = read_config(args.config_file)
            # Pass potentially modified args if CL config needs them
            # cfg.args = args
            final_output_path, cl_time = run_cl_step_for_workflow(intermediate_ray_ds, cfg, args.output)
            logger.info(f"CL step completed in {cl_time:.2f}s. Final output: {final_output_path}")
            # workflow_total_time = nd_time + cl_time # This is approximate, wall clock is better

        elif args.workflow == "cl_nd":
            logger.info("Executing CL -> ND workflow...")

            # === Integrated CL + ND ===
            cfg = read_config(args.config_file)
            # Pass args needed by the integrated function (input, output, ND params etc.)
            
            # cfg.args = args
            
            final_output_path, wf_time, final_record_count, total_duplicate_count, num_nodes_clnd = run_cl_nd_integrated_workflow(args, cfg)
            num_nodes_used = num_nodes_clnd
            # workflow_total_time = wf_time # Use time reported by function
            logger.info(f"CL->ND integrated workflow completed in {wf_time:.2f}s.")
            logger.info(f"Final Records: {final_record_count}, Intra-Cluster Duplicates Removed: {total_duplicate_count}")
            logger.info(f"Final output: {final_output_path}")

        else:
            # Should not happen due to argparse choices
            raise ValueError(f"Invalid workflow specified: {args.workflow}")

        # --- Workflow Complete - Final Benchmarking ---
        actual_workflow_time = time.time() - workflow_start_time
        logger.info(f"Workflow '{args.workflow}' finished. Total wall clock time: {actual_workflow_time:.2f} seconds.")

        # --- Log to Database ---
        logger.info("Logging benchmark results to database...")
        engine = init_db(args.db_uri) # Uses arg or default logic
        session = get_session(engine)

        # Calculate total input size for logging
        total_size_gb = None
        input_file_for_log = args.input_file
        if args.input_file and args.limit_files is not None:
            try:
                input_files = glob.glob(args.input_file)[:args.limit_files]
                total_size_gb = get_total_size_gb(input_files)
                logger.info(f"Calculated input size: {total_size_gb:.2f} GB for {len(input_files)} files")
                # Optionally shorten input_file string for DB if it's a long pattern
                if len(input_files) != 1:
                     input_file_for_log = f"{args.input_file} (Limit: {args.limit_files})"
                else:
                     input_file_for_log = input_files[0] # Log specific file if only one
            except Exception as e:
                 logger.warning(f"Could not glob or calculate size for {args.input_file}: {e}")

        # Prepare args for DB logging function (ensure it matches BenchmarkRun.create_from_args expectations)
        log_args = argparse.Namespace(
             # Essential args expected by create_from_args
             input_file=input_file_for_log,
             table=None, # Assuming file input
             output=final_output_path, # Log the final output path
             threshold=args.threshold,
             ngram_size=args.ngram_size,
             min_ngram_size=args.min_ngram_size,
             num_perm=args.num_perm,
             limit_files=args.limit_files
             # Add any other args needed by create_from_args
        )

        benchmark_notes = args.notes if args.notes else f"Workflow: {args.workflow}"
        if args.workflow == 'cl_nd':
             benchmark_notes += f" (CL Cfg: {os.path.basename(args.config_file)})"


        benchmark_run = BenchmarkRun.create_from_args(
            session=session,
            args=log_args,
            duplicate_count=total_duplicate_count, # Meaning depends on workflow
            record_count=final_record_count,       # Final count after all steps
            execution_time=actual_workflow_time,   # Total wall clock time
            implementation=args.workflow,          # Use workflow name as implementation
            num_nodes=num_nodes_used,              # Max nodes used during the workflow
            notes=benchmark_notes,
            limit_files=args.limit_files,          # Log the limit used
            total_size_gb=total_size_gb            # Log calculated size
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