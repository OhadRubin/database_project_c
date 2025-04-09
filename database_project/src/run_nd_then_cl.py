import argparse
import os
import sys
import time
import glob
import logging
from ml_collections import config_dict
import yaml

# Assume core project structure allows these imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from database_project.src.minhash import modified_minhash_lsh # <--- ASSUMED MODIFIED FUNCTION
from database_project.src.ray_tfidf_vec import modified_run_clustering_on_ray_dataset, read_config # <--- ASSUMED MODIFIED FUNCTION
from database_project.src.db import init_db, get_session, BenchmarkRun

# A helper function (could be in a utils file)
def get_total_size_gb(files):
    try:
        total_bytes = sum(os.path.getsize(f) for f in files)
        return total_bytes / (1024 * 1024 * 1024)  # Convert bytes to GB
    except Exception as e:
        logging.warning(f"Could not calculate total size: {e}")
        return None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run Near-Deduplication (ND) then Clustering (CL)")
    # --- Input/Output ---
    parser.add_argument("--input_file", type=str, required=True, help="Input file pattern (e.g., 'data/*.json.gz')")
    parser.add_argument("--output", "-o", type=str, required=True, help="Final output directory for clustered data")
    parser.add_argument("--limit_files", type=int, default=None, help="Limit the number of input files to process")

    # --- ND Parameters (from minhash.py) ---
    parser.add_argument("--threshold", type=float, default=0.7, help="MinHash Similarity threshold")
    parser.add_argument("--min_ngram_size", type=int, default=5, help="Min N-gram size for ND")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size for ND")
    parser.add_argument("--num_perm", type=int, default=256, help="Number of permutations for MinHash")
    parser.add_argument("--column", "-c", type=str, default="text", help="Column to deduplicate")
    parser.add_argument("--use_ray", type=bool, default=True, help="Use RayDP for Spark ND (recommended)") # Assume True for in-memory passing

    # --- CL Parameters ---
    parser.add_argument("--config_file", type=str, default="database_project/src/configs/base.yml", help="Path to clustering config YAML")

    # --- Benchmarking ---
    parser.add_argument("--notes", type=str, default="ND->CL Workflow", help="Notes for benchmark DB")
    parser.add_argument("--db_uri", type=str, default=None, help="Database URI (uses DB_URI env var or sqlite default if None)")

    # Mock is not really applicable here as we assume modified functions
    # parser.add_argument("--mock", action="store_true", help="Mock execution")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    workflow_start_time = time.time()

    logging.info("Starting ND -> CL Workflow")

    # --- Stage 1: Near-Deduplication (ND) ---
    logging.info("Running Near-Deduplication stage...")
    try:
        # Call the modified minhash function which returns a Ray Dataset
        deduplicated_ray_ds, duplicate_count, num_nodes_nd, nd_time = modified_minhash_lsh(args)
        logging.info(f"ND stage completed in {nd_time:.2f} seconds.")
        logging.info(f"Duplicates found by ND: {duplicate_count}")
        if deduplicated_ray_ds:
            record_count = deduplicated_ray_ds.count() # Get count AFTER deduplication
            logging.info(f"Records after ND: {record_count}")
        else:
             raise ValueError("ND stage did not return a valid Ray Dataset.")

    except Exception as e:
        logging.error(f"Error during ND stage: {e}", exc_info=True)
        sys.exit(1)

    # --- Stage 2: Clustering (CL) ---
    logging.info("Running Clustering stage...")
    try:
        # Load clustering config
        cfg = read_config(args.config_file)
        cfg.args = args # Pass args if needed by config resolution

        # Call the modified clustering function that accepts a Ray Dataset
        final_output_path, cl_time = modified_run_clustering_on_ray_dataset(
            input_ray_dataset=deduplicated_ray_ds,
            cfg=cfg,
            final_output_path=args.output # Pass the *final* desired output path
        )
        logging.info(f"CL stage completed in {cl_time:.2f} seconds.")
        logging.info(f"Final clustered data saved to: {final_output_path}")

    except Exception as e:
        logging.error(f"Error during CL stage: {e}", exc_info=True)
        sys.exit(1)

    # --- Workflow Completion & Benchmarking ---
    workflow_total_time = time.time() - workflow_start_time
    logging.info(f"ND -> CL workflow finished in {workflow_total_time:.2f} seconds.")

    # Log to Database
    try:
        engine = init_db(args.db_uri) # Use arg or default logic
        session = get_session(engine)

        # Calculate total input size
        total_size_gb = None
        if args.input_file and args.limit_files is not None:
            try:
                input_files = glob.glob(args.input_file)[:args.limit_files]
                total_size_gb = get_total_size_gb(input_files)
                logging.info(f"Calculated input size: {total_size_gb:.2f} GB for {len(input_files)} files")
            except Exception as e:
                 logging.warning(f"Could not glob or calculate size for {args.input_file}: {e}")

        # Create a combined args-like object for logging if needed, or pass necessary fields
        # Note: We need to capture params from BOTH stages if relevant to the benchmark run entry
        log_args = argparse.Namespace(
             input_file=args.input_file,
             table=None, # Assuming file input for this script
             output=args.output,
             threshold=args.threshold,
             ngram_size=args.ngram_size,
             min_ngram_size=args.min_ngram_size,
             num_perm=args.num_perm,
             limit_files=args.limit_files
             # Add clustering parameters if needed in DB schema/notes
        )

        benchmark_run = BenchmarkRun.create_from_args(
            session=session,
            args=log_args,
            duplicate_count=duplicate_count, # From ND stage
            record_count=record_count,       # Count *after* ND
            execution_time=workflow_total_time, # Total workflow time
            implementation="nd_then_cl",      # Specific workflow name
            num_nodes=num_nodes_nd,           # Or max(num_nodes_nd, num_nodes_cl) if CL uses different nodes
            notes=args.notes,
            limit_files=args.limit_files,
            total_size_gb=total_size_gb
            # Add clustering specific params to notes if desired
        )
        logging.info(f"Benchmark data saved with ID: {benchmark_run.id}")

    except Exception as e:
        logging.error(f"Error saving benchmark data: {e}", exc_info=True)