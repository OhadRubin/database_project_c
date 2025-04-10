#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting Additional Experiments Script..."

# --- Configuration ---
PROJECT_DIR="$HOME/database_project_c" # Adjust if your project is elsewhere
PYTHON_EXEC="python3.10"
RUN_SCRIPT="$PROJECT_DIR/database_project/src/run_workflows.py"
BASE_INPUT_FILES="/dev/shm/c4_files/c4-train.*.json.gz"
BASE_OUTPUT_DIR="/mnt/gcs_bucket/ray_experiment_outputs" # Base directory for experiment outputs on GCS bucket
# BASE_OUTPUT_DIR="/dev/shm/ray_experiment_outputs" # Alternative: Use local /dev/shm if GCS Fuse isn't desired/reliable for many small writes
BASE_CONFIG_FILE="$PROJECT_DIR/database_project/src/configs/base.yml"
DEFAULT_LIMIT_FILES=40 # Default data size for parameter sensitivity tests
DEFAULT_THRESHOLD=0.7  # Default threshold for data/perm scaling tests
DEFAULT_NUM_PERM=256   # Default num_perm for data/threshold scaling tests

# --- Environment Setup ---
# Ensure we are in the project directory context if needed (scripts might use relative paths)
# cd "$PROJECT_DIR" # Uncomment if scripts rely on CWD being the project root

# Set the database connection string (ensure your environment/bashrc has REDIS_PASSWORD if needed)
if [[ -z "$POSTGRES_ADDRESS" ]]; then
  if [[ -z "$REDIS_PASSWORD" ]]; then
    echo "WARNING: POSTGRES_ADDRESS is not set and REDIS_PASSWORD is not set. Database logging might fail."
    # Optionally set a default or exit:
    # export POSTGRES_ADDRESS="your_default_connection_string"
    # exit 1
  else
    export POSTGRES_ADDRESS="postgresql+psycopg2://postgres:$REDIS_PASSWORD@34.141.239.167:5564/postgres"
    echo "POSTGRES_ADDRESS set using REDIS_PASSWORD."
  fi
else
    echo "POSTGRES_ADDRESS is already set."
fi

# Check if the base output directory exists
if [[ "$BASE_OUTPUT_DIR" == /mnt/gcs_bucket/* ]] && [ ! -d "/mnt/gcs_bucket" ]; then
    echo "ERROR: GCS Bucket mount point /mnt/gcs_bucket does not exist or is not mounted."
    echo "Please ensure GCS Fuse is running correctly before starting experiments."
    exit 1
fi
mkdir -p "$BASE_OUTPUT_DIR"
echo "Using base output directory: $BASE_OUTPUT_DIR"

# --- Helper Function to Run a Single Experiment Config ---
run_experiment() {
  local workflow="$1"
  local limit_files="$2"
  local threshold="$3"
  local num_perm="$4"
  local config_file="$5"
  local experiment_tag="$6" # e.g., "datasize", "threshold", "numperm"

  # Create a unique output directory name based on parameters
  local output_subdir="${workflow}_${experiment_tag}_files${limit_files}_thresh${threshold}_perm${num_perm}"
  local output_path="$BASE_OUTPUT_DIR/$output_subdir"

  # Create descriptive notes for the database
  local notes="Experiment: ${experiment_tag} | Workflow: ${workflow} | LimitFiles: ${limit_files} | Threshold: ${threshold} | NumPerm: ${num_perm} | Config: $(basename ${config_file})"

  # Construct the command
  local cmd="$PYTHON_EXEC $RUN_SCRIPT \
    --workflow $workflow \
    --input_file \"$BASE_INPUT_FILES\" \
    --output $output_path \
    --config_file $config_file \
    --limit_files $limit_files \
    --threshold $threshold \
    --num_perm $num_perm \
    --notes \"$notes\" \
    --use_ray True" # Assuming Ray is always used as per project setup

  echo "----------------------------------------------------------------------"
  echo "Running: $notes"
  echo "Command:"
  echo "$cmd"
  echo "Output Dir: $output_path"
  echo "----------------------------------------------------------------------"

  # Create output dir and run the command
  mkdir -p "$output_path"
  # Use eval to handle the quotes in input_file properly
  eval "$cmd"

  echo "Finished: $notes"
  echo "----------------------------------------------------------------------"
  sleep 10 # Small delay between runs, maybe helpful for resource cleanup/monitoring
}

# --- Experiment 1.1: Data Size Scaling ---
echo ""
echo "##################################################"
echo "# Running Experiment 1.1: Data Size Scaling      #"
echo "##################################################"
echo ""

FILE_SIZES=(10 20 40 80) # Adjust as needed/feasible (160 might be too large/slow)

for size in "${FILE_SIZES[@]}"; do
  run_experiment "nd_cl" "$size" "$DEFAULT_THRESHOLD" "$DEFAULT_NUM_PERM" "$BASE_CONFIG_FILE" "datasize"
  run_experiment "cl_nd" "$size" "$DEFAULT_THRESHOLD" "$DEFAULT_NUM_PERM" "$BASE_CONFIG_FILE" "datasize"
done

# --- Experiment 2.1: Varying Similarity Threshold ---
echo ""
echo "##################################################"
echo "# Running Experiment 2.1: Threshold Sensitivity  #"
echo "##################################################"
echo ""

THRESHOLDS=(0.6 0.7 0.8 0.9)

for thr in "${THRESHOLDS[@]}"; do
  run_experiment "nd_cl" "$DEFAULT_LIMIT_FILES" "$thr" "$DEFAULT_NUM_PERM" "$BASE_CONFIG_FILE" "threshold"
  run_experiment "cl_nd" "$DEFAULT_LIMIT_FILES" "$thr" "$DEFAULT_NUM_PERM" "$BASE_CONFIG_FILE" "threshold"
done

# --- Experiment 2.2: Varying Number of Permutations ---
echo ""
echo "##################################################"
echo "# Running Experiment 2.2: NumPerm Sensitivity    #"
echo "##################################################"
echo ""

NUM_PERMS=(128 256 512)

for perm in "${NUM_PERMS[@]}"; do
  run_experiment "nd_cl" "$DEFAULT_LIMIT_FILES" "$DEFAULT_THRESHOLD" "$perm" "$BASE_CONFIG_FILE" "numperm"
  run_experiment "cl_nd" "$DEFAULT_LIMIT_FILES" "$DEFAULT_THRESHOLD" "$perm" "$BASE_CONFIG_FILE" "numperm"
done

# --- Experiments Requiring Manual Changes ---

echo ""
echo "##################################################"
echo "# Manual Experiments                             #"
echo "##################################################"
echo ""
echo "NOTE: The following experiments require manual setup:"
echo ""
echo "1. Experiment 1.2 (Cluster Size Scaling):"
echo "   - Stop the Ray cluster."
echo "   - Modify your cluster provisioning (e.g., gcloud commands) to use a different number of nodes (e.g., 2, 5)."
echo "   - Restart the Ray cluster using run.sh or equivalent on all nodes."
echo "   - Re-run this script (or specific parts) with the desired fixed parameters (e.g., limit_files=40)."
echo "   - Remember to adjust 'num_nodes' in the database analysis if needed."
echo ""
echo "2. Experiment 3.1 (Varying Stage 1 Cluster Count 'k1'):"
echo "   - Edit the config file: $BASE_CONFIG_FILE"
echo "   - Change the value of 'n_clusters' under 'stages_list[0].kmeans' (line ~33)."
echo "   - Example values: 5, 10, 20, 50."
echo "   - Save the file."
echo "   - Re-run the relevant 'run_experiment' calls (e.g., for cl_nd workflow with fixed data size/ND params)."
echo "   - OR: Create separate config files (e.g., base_k5.yml, base_k20.yml) and modify the 'run_experiment' calls here to use them."
echo ""
echo "3. Experiment 4 (Output Quality Analysis):"
echo "   - These experiments require post-processing of the output data."
echo "   - Use notebooks like 'examine_clusters.ipynb' or 'viewer.ipynb' to load the generated Parquet files from '$BASE_OUTPUT_DIR'."
echo "   - Implement logic to sample data, compare against ground truth (if available), calculate metrics (Silhouette score), or perform qualitative analysis."
echo ""

echo "##################################################"
echo "# All Automated Experiments Completed            #"
echo "##################################################"