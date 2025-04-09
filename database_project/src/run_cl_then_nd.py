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
from database_project.src.ray_tfidf_vec import modified_run_clustering_and_nd_pipeline, read_config # <--- ASSUMED MODIFIED FUNCTION
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
    parser = argparse.ArgumentParser(description="Run Clustering (CL) then Near-Deduplication (ND) within clusters")
    # --- Input/Output ---
    parser.add_argument("--input_file", type=str, required=True, help="Input file pattern (e.g., 'data/*.json.gz')")
    parser.add_argument("--output", "-o", type=str, required=True, help="Final output directory for clustered & deduplicated data")
    parser.add_argument("--limit_files", type=int, default=None, help="Limit the number of input files to process")

    # --- CL Parameters ---
    parser.add_argument("--config_file", type=str, default="database_project/src/configs/base.yml", help="Path to clustering config YAML")

    # --- ND Parameters (needed by the integrated ND step) ---
    parser.add_argument("--threshold", type=float, default=0.7, help="MinHash Similarity threshold for intra-cluster ND")
    parser.add_argument("--min_ngram_size", type=int, default=5, help="Min N-gram size for intra-cluster ND")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size for intra-cluster ND")
    parser.add_argument("--num_perm", type=int, default=256, help="Number of permutations for intra-cluster MinHash")
    parser.add_argument("--column", "-c", type=str, default="text", help="Column to deduplicate") # Should match clustering input

    # --- Benchmarking ---
    parser.add_argument("--notes", type=str, default="CL->ND Workflow", help="Notes for benchmark DB")
    parser.add_argument("--db_uri", type=str, default=None, help="Database URI (uses DB_URI env var or sqlite default if None)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    workflow_start_time = time.time() # Although the modified function might return total time

    logging.info("Starting CL -> ND Workflow")

    # --- Run Combined CL and ND Pipeline ---
    logging.info("Running combined Clustering and intra-cluster ND pipeline...")
    try:
        # Load clustering config
        cfg = read_config(args.config_file)
        # cfg.args = args # Pass args if needed by config resolution or the modified function

        # Call the modified Ray function that handles CL, internal ND, and final save
        final_output_path, total_time, final_record_count, duplicate_count_removed, num_nodes = modified_run_clustering_and_nd_pipeline(
            args=args, # Pass args containing input, output, limit, ND params
            cfg=cfg    # Pass clustering config
        )

        logging.info(f"CL -> ND pipeline completed in {total_time:.2f} seconds.")
        logging.info(f"Intra-cluster duplicates removed: {duplicate_count_removed}")
        logging.info(f"Final records after CL and ND: {final_record_count}")
        logging.info(f"Final deduplicated & clustered data saved to: {final_output_path}")

    except Exception as e:
        logging.error(f"Error during combined CL -> ND pipeline: {e}", exc_info=True)
        sys.exit(1)

    # --- Workflow Completion & Benchmarking ---
    # Using total_time returned from the function, assuming it captures the whole duration accurately
    workflow_total_time = total_time
    logging.info(f"CL -> ND workflow finished.") # Time already logged

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


        # Create args-like object for logging
        # Include params relevant to *both* CL (from config) and ND (from args)
        log_args = argparse.Namespace(
             input_file=args.input_file,
             table=None, # Assuming file input
             output=args.output,
             threshold=args.threshold, # ND param
             ngram_size=args.ngram_size, # ND param
             min_ngram_size=args.min_ngram_size, # ND param
             num_perm=args.num_perm, # ND param
             limit_files=args.limit_files
             # Add clustering parameters if needed in DB schema/notes
        )
        # Add cluster details to notes?
        cluster_notes = f"CL Config: {args.config_file}. Stages: {len(cfg.get('stages_list', []))}"
        full_notes = f"{args.notes}. {cluster_notes}" if args.notes else cluster_notes


        benchmark_run = BenchmarkRun.create_from_args(
            session=session,
            args=log_args,
            duplicate_count=duplicate_count_removed, # From the combined stage
            record_count=final_record_count,       # From the combined stage
            execution_time=workflow_total_time,    # From the combined stage
            implementation="cl_then_nd",           # Specific workflow name
            num_nodes=num_nodes,                   # From the combined stage
            notes=full_notes,
            limit_files=args.limit_files,
            total_size_gb=total_size_gb
        )
        logging.info(f"Benchmark data saved with ID: {benchmark_run.id}")

    except Exception as e:
        logging.error(f"Error saving benchmark data: {e}", exc_info=True)