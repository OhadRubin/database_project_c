#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting Combined Setup and Experiments Script..."
START_TIME=$(date +%s)
SCRIPT_START_TIME=$START_TIME # For total script time

# --- Configuration ---
PROJECT_DIR="$HOME/database_project_c" # Adjust if your project is elsewhere
PYTHON_EXEC="python3.10"
SRC_DIR="$PROJECT_DIR/database_project/src"
RUN_SCRIPT="$SRC_DIR/run_workflows.py"
DOWNLOAD_SCRIPT="$SRC_DIR/download_c4.py"
DB_SCRIPT="$SRC_DIR/db.py"
RAY_EXEC="$HOME/.local/bin/ray" # Adjust if ray is installed elsewhere

# Data/Output Config
BASE_INPUT_FILES_DIR="/dev/shm/c4_files"
BASE_INPUT_FILES_PATTERN="$BASE_INPUT_FILES_DIR/c4-train.*.json.gz"
NUM_EXPECTED_DATA_FILES=40

# >>> Choose Output Location <<<
USE_GCS_OUTPUT=true # Set to false to use /dev/shm instead of GCS
GCS_BUCKET_NAME="meliad2_us2_backup" # Your GCS bucket name
GCS_MOUNT_POINT="/mnt/gcs_bucket"
GCS_CACHE_DIR="/dev/shm/gcs_cache"

if [ "$USE_GCS_OUTPUT" = true ]; then
  BASE_OUTPUT_DIR="$GCS_MOUNT_POINT/ray_experiment_outputs_$(date +%Y%m%d_%H%M%S)" # Unique dir per run on GCS
  FINAL_CLUSTER_OUTPUT_BASE="$GCS_MOUNT_POINT/ray_clustering_output" # For base.yml consistency check
else
  BASE_OUTPUT_DIR="/dev/shm/ray_experiment_outputs_$(date +%Y%m%d_%H%M%S)" # Unique dir per run on /dev/shm
  FINAL_CLUSTER_OUTPUT_BASE="/dev/shm/ray_clustering_output" # For base.yml consistency check
fi
echo "Using Base Output Directory: $BASE_OUTPUT_DIR"

# Experiment Defaults
BASE_CONFIG_FILE="$SRC_DIR/configs/base.yml"
DEFAULT_LIMIT_FILES=40 # Default data size for parameter sensitivity tests
DEFAULT_THRESHOLD=0.7  # Default threshold for data/perm scaling tests
DEFAULT_NUM_PERM=256   # Default num_perm for data/threshold scaling tests

# Ray Cluster Config
GCP_ZONE="us-central2-b" # Zone where TPUs are located
HEAD_NODE_GCP_NAME="v4-8-node-2" # Specific name of the intended head node in GCP
TPU_RESOURCE_NAME="TPU-v4-8-head" # Ray resource name matching base.yml

# --- Helper Function ---
log_elapsed_time() {
  local start_timestamp=$1
  local phase_name="$2"
  if [ -n "$start_timestamp" ]; then
    local current_time=$(date +%s)
    local elapsed_time=$((current_time - start_timestamp))
    local elapsed_minutes=$((elapsed_time / 60))
    local elapsed_seconds=$((elapsed_time % 60))
    echo "$phase_name Time elapsed: $elapsed_minutes minutes and $elapsed_seconds seconds"
  else
    echo "Warning: Start timestamp not provided for phase '$phase_name'"
  fi
}

# --- 1. Initial Setup (Run on ALL nodes) ---
echo "[Phase 1] Initial Setup on node $(hostname)..."

# Check/Install Core Tools (jq, git)
if ! command -v jq &> /dev/null; then
    echo "jq not found. Please install it (e.g., sudo apt-get update && sudo apt-get install jq)"
    exit 1
fi
if ! command -v git &> /dev/null; then
    echo "git not found. Please install it (e.g., sudo apt-get update && sudo apt-get install git)"
    exit 1
fi
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "$PYTHON_EXEC not found. Please install Python 3.10."
    exit 1
fi

# Clone or Update Project Repo
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Cloning project repository..."
    git clone https://github.com/OhadRubin/database_project_c "$PROJECT_DIR"
    cd "$PROJECT_DIR"
else
    echo "Updating project repository..."
    cd "$PROJECT_DIR"
    git pull
fi



# Set Database Connection String
if [[ -z "$POSTGRES_ADDRESS" ]]; then
  if [[ -z "$REDIS_PASSWORD" ]]; then
    echo "ERROR: POSTGRES_ADDRESS is not set and REDIS_PASSWORD is not set. Cannot connect to DB."
    exit 1
  else
    # Ensure the IP, port, user, dbname are correct here
    export POSTGRES_ADDRESS="postgresql+psycopg2://postgres:$REDIS_PASSWORD@34.141.239.167:5564/postgres"
    echo "POSTGRES_ADDRESS set using REDIS_PASSWORD."
  fi
else
    echo "POSTGRES_ADDRESS is already set."
fi
# Optional: Initialize DB schema if needed (run only once, maybe on head node)
# echo "Initializing database schema if necessary..."
# $PYTHON_EXEC $DB_SCRIPT

# Download and Verify C4 Data
echo "Checking C4 dataset..."
mkdir -p "$BASE_INPUT_FILES_DIR"
CURRENT_FILE_COUNT=$(ls "$BASE_INPUT_FILES_DIR" | wc -l)
if [ "$CURRENT_FILE_COUNT" -ne "$NUM_EXPECTED_DATA_FILES" ]; then
    echo "Expected $NUM_EXPECTED_DATA_FILES files but found $CURRENT_FILE_COUNT. Running download script..."
    $PYTHON_EXEC "$DOWNLOAD_SCRIPT"
    # Re-verify after download
    CURRENT_FILE_COUNT=$(ls "$BASE_INPUT_FILES_DIR" | wc -l)
    if [ "$CURRENT_FILE_COUNT" -ne "$NUM_EXPECTED_DATA_FILES" ]; then
        echo "ERROR: Download failed. Still found only $CURRENT_FILE_COUNT files after running download script."
        exit 1
    fi
    echo "Verified $NUM_EXPECTED_DATA_FILES files in $BASE_INPUT_FILES_DIR after download."
else
    echo "Verified $NUM_EXPECTED_DATA_FILES files already exist in $BASE_INPUT_FILES_DIR."
fi

# --- 2. Node Role Identification and GCS Mount (Run on ALL nodes) ---
echo "[Phase 2] Node Role Identification & GCS Mount..."
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not found. Needed for node discovery."
    exit 1
fi

ALL_NODE_IPS=$(gcloud compute tpus list --zone "$GCP_ZONE" --filter="name ~ v4-8" --format=json | jq -r '.[].ipAddress' || echo "ERROR_GETTING_IPS")
if [[ "$ALL_NODE_IPS" == "ERROR_GETTING_IPS" ]]; then
    echo "ERROR: Failed to get node IPs using gcloud. Check zone and filter."
    exit 1
fi
N_NODES=$(echo "$ALL_NODE_IPS" | wc -l)
echo "Detected $N_NODES potential Ray nodes."

HEAD_IP=$(gcloud compute tpus list --zone "$GCP_ZONE" --filter="name:$HEAD_NODE_GCP_NAME" --format=json | jq -r '.[0].ipAddress' || echo "ERROR_GETTING_HEAD_IP") # Use [0] to ensure single IP
if [[ "$HEAD_IP" == "ERROR_GETTING_HEAD_IP" ]] || [[ -z "$HEAD_IP" ]]; then
    echo "ERROR: Failed to get head node IP ($HEAD_NODE_GCP_NAME) using gcloud. Check name and zone."
    exit 1
fi

MY_IP=$(hostname -I | awk '{print $1}')
IS_HEAD=false
if [ "$MY_IP" = "$HEAD_IP" ]; then
    echo "This node ($(hostname)) is the HEAD node ($HEAD_IP)."
    IS_HEAD=true
else
    echo "This node ($(hostname) - $MY_IP) is a WORKER node. Head node IP is $HEAD_IP."
    IS_HEAD=false
fi

# Setup GCS Fuse Mount (Conditional)
if [ "$USE_GCS_OUTPUT" = true ]; then
    echo "Setting up GCS Fuse mount for $GCS_BUCKET_NAME at $GCS_MOUNT_POINT..."
    if ! command -v gcsfuse &> /dev/null; then
        echo "ERROR: gcsfuse command not found. Please install it."
        exit 1
    fi
    sudo mkdir -p "$GCS_MOUNT_POINT"
    sudo mkdir -p "$GCS_CACHE_DIR"
    sudo chown -R "$USER:$USER" "$GCS_MOUNT_POINT"
    sudo chown -R "$USER:$USER" "$GCS_CACHE_DIR"
    sudo chmod 777 "$GCS_CACHE_DIR" # Ensure cache is writable

    # Check if already mounted, unmount if necessary (careful with this)
    if mount | grep -q "$GCS_MOUNT_POINT"; then
        echo "Attempting to unmount existing GCS Fuse mount at $GCS_MOUNT_POINT..."
        sudo umount -l "$GCS_MOUNT_POINT" || echo "Warning: Failed to unmount $GCS_MOUNT_POINT. Trying to continue..."
        sleep 2
    fi

    echo "Mounting GCS bucket $GCS_BUCKET_NAME to $GCS_MOUNT_POINT..."
    gcsfuse \
        --implicit-dirs \
        --file-cache-enable-parallel-downloads \
        --file-cache-parallel-downloads-per-file 100 \
        --file-cache-max-parallel-downloads -1 \
        --file-cache-download-chunk-size-mb 10 \
        --file-cache-max-size-mb 200000 `# Adjust cache size as needed` \
        --dir-mode 0777 \
        -o allow_other --foreground \
        --log-file "$HOME/gcsfuse.log" --log-format "text" \
        --cache-dir "$GCS_CACHE_DIR" \
        "$GCS_BUCKET_NAME" "$GCS_MOUNT_POINT" &
    GCSFUSE_PID=$!
    sleep 5 # Give GCS Fuse time to mount or fail

    # Verify mount
    if ! mount | grep -q "$GCS_MOUNT_POINT"; then
        echo "ERROR: Failed to mount GCS bucket $GCS_BUCKET_NAME to $GCS_MOUNT_POINT. Check $HOME/gcsfuse.log"
        # Optionally kill the background process if it's stuck
        # kill $GCSFUSE_PID > /dev/null 2>&1
        exit 1
    fi
    echo "GCS Fuse mounted successfully."
else
    echo "Skipping GCS Fuse setup as USE_GCS_OUTPUT is false."
fi

# --- 3. Start Ray Cluster (Run on ALL nodes) ---
echo "[Phase 3] Starting Ray Cluster..."
if ! command -v $RAY_EXEC &> /dev/null; then
    echo "Ray executable not found at $RAY_EXEC. Trying default 'ray'..."
    RAY_EXEC="ray"
    if ! command -v $RAY_EXEC &> /dev/null; then
       echo "ERROR: Ray command not found. Please install Ray."
       exit 1
    fi
fi

# Check if Ray is already running
if $RAY_EXEC status > /dev/null 2>&1; then
    echo "Ray runtime seems to be already running. Skipping Ray start."
    # Optional: could add logic to stop and restart if needed
else
    echo "Starting Ray runtime..."
    if $IS_HEAD; then
        echo "Starting Ray HEAD node..."
        $RAY_EXEC start --head --dashboard-host 0.0.0.0 --disable-usage-stats --resources="{\"$TPU_RESOURCE_NAME\": 1}"
        echo "Ray head node started."
    else
        echo "Starting Ray WORKER node, connecting to $HEAD_IP:6379..."
        $RAY_EXEC start --address="$HEAD_IP:6379" --disable-usage-stats --block --resources="{\"$TPU_RESOURCE_NAME\": 1}"
        # Note: --block might make workers wait here. Can remove if workers should exit after joining.
        echo "Ray worker node started and joined cluster."
    fi
fi

# --- 4. Wait for Cluster Ready (Run on HEAD node only) ---
if $IS_HEAD; then
    echo "[Phase 4 - Head Node] Waiting for $N_NODES nodes to join Ray cluster..."
    while true; do
        # Ensure Ray is initialized within the python check command
        CURRENT_NODE_COUNT=$($PYTHON_EXEC -c 'import ray; import time; retries=5; delay=2; count=0; \
            for i in range(retries): \
              try: ray.init(address="auto", ignore_reinit_error=True); count=len([n for n in ray.nodes() if n["alive"]]); break; \
              except Exception as e: print(f"Retrying connection... ({e})", file=sys.stderr); time.sleep(delay); \
            print(count)' 2>/dev/null || echo 0)

        echo "Current live node count: $CURRENT_NODE_COUNT"
        if [ "$CURRENT_NODE_COUNT" -ge "$N_NODES" ]; then
            echo "All $N_NODES nodes are now available in the Ray cluster."
            $RAY_EXEC status # Print final status
            break
        fi
        echo "Waiting for more nodes to join... (sleeping 5 seconds)"
        sleep 5
    done
    log_elapsed_time $START_TIME "Setup Phase" # Log time taken for setup
    SETUP_COMPLETE_TIME=$(date +%s)

    # --- 5. Run Experiments (Run on HEAD node only) ---
    echo "[Phase 5 - Head Node] Running Experiments..."

    # Create base output directory
    mkdir -p "$BASE_OUTPUT_DIR"
    echo "Created base experiment output directory: $BASE_OUTPUT_DIR"

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
        --input_file \"$BASE_INPUT_FILES_PATTERN\" \
        --output $output_path \
        --config_file $config_file \
        --limit_files $limit_files \
        --threshold $threshold \
        --num_perm $num_perm \
        --notes \"$notes\" \
        --use_ray True" # Assuming Ray is always used

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

      # *** IMPORTANT NOTE ***
      # Verify that run_cl_step_for_workflow in ray_tfidf_vec.py uses the --output path passed here
      # (via args.output) for its *final* data writing, instead of only using base_dir from base.yml.
      # If it only uses base.yml, outputs from different experiments might overwrite each other
      # in $FINAL_CLUSTER_OUTPUT_BASE. You may need to modify ray_tfidf_vec.py.
      # Check if $FINAL_CLUSTER_OUTPUT_BASE exists, as it might be written to by run_cl_step_for_workflow
      if [ -d "$FINAL_CLUSTER_OUTPUT_BASE" ]; then
           echo "WARNING: Directory '$FINAL_CLUSTER_OUTPUT_BASE' exists. Check if ray_tfidf_vec.py is writing output there instead of '$output_path'."
      fi
      # ********************

      echo "Finished: $notes"
      echo "----------------------------------------------------------------------"
      sleep 10 # Small delay between runs
    }

    # --- Experiment 1.1: Data Size Scaling ---
    echo ""
    echo "##################################################"
    echo "# Running Experiment 1.1: Data Size Scaling      #"
    echo "##################################################"
    echo ""

    FILE_SIZES=(10 20 40) # Reduced max size for feasibility, adjust as needed (80?)

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


    # --- 6. Post-Experiment Notes (Run on HEAD node only) ---
    echo "[Phase 6 - Head Node] Post-Experiment Information"

    log_elapsed_time $SETUP_COMPLETE_TIME "Experiment Execution Phase"

    # --- Manual Experiments Section ---
    echo ""
    echo "##################################################"
    echo "# Manual Experiments Required                    #"
    echo "##################################################"
    echo ""
    echo "NOTE: The following experiments require manual setup/execution:"
    echo ""
    echo "1. Experiment 1.2 (Cluster Size Scaling):"
    echo "   - Stop the Ray cluster (ray stop)."
    echo "   - Modify your cluster provisioning to use a different number of nodes (e.g., 2, 5)."
    echo "   - Re-run this script on all nodes of the new cluster configuration."
    echo "   - Analyze results comparing runs with different 'num_nodes' in the database."
    echo ""
    echo "2. Experiment 3.1 (Varying Stage 1 Cluster Count 'k1'):"
    echo "   - Edit the config file: $BASE_CONFIG_FILE"
    echo "   - Change 'n_clusters' under 'stages_list[0].kmeans' (line ~33)."
    echo "   - OR create new config files (e.g., config_k5.yml, config_k20.yml)."
    echo "   - Re-run specific 'run_experiment' calls using the modified/new config file, e.g.:"
    echo "     # run_experiment \"cl_nd\" \"$DEFAULT_LIMIT_FILES\" \"$DEFAULT_THRESHOLD\" \"$DEFAULT_NUM_PERM\" \"path/to/config_k5.yml\" \"k1_sensitivity\""
    echo ""
    echo "3. Experiment 4 (Output Quality Analysis):"
    echo "   - Use notebooks like 'examine_clusters.ipynb' or 'viewer.ipynb' to load generated Parquet files from '$BASE_OUTPUT_DIR'."
    echo "   - Implement logic for sampling, ground truth comparison, metrics (Silhouette), or qualitative analysis."
    echo ""

    echo "##################################################"
    echo "# All Automated Experiments Completed            #"
    echo "##################################################"

    # Optionally stop Ray cluster
    # echo "Stopping Ray cluster..."
    # ray stop

    # Optionally cleanup GCS mount
    if [ "$USE_GCS_OUTPUT" = true ]; then
        echo "Unmounting GCS Fuse..."
        sudo umount -l "$GCS_MOUNT_POINT" || echo "Warning: Failed to unmount GCS Fuse."
        # kill $GCSFUSE_PID > /dev/null 2>&1 # Kill background process if foreground wasn't used or needed
    fi

else
    # Worker nodes just need to keep Ray running.
    # If --block was used in `ray start`, they will wait here.
    # If --block was not used, they might exit after joining Ray.
    # Add a sleep loop or another mechanism if workers need to stay alive independently.
    echo "[Worker Node $(hostname)] Joined Ray cluster. Waiting for tasks or termination."
    # Keep worker alive e.g., sleep infinity
    # sleep infinity
fi


log_elapsed_time $SCRIPT_START_TIME "Total Script"
echo "Script finished on node $(hostname)."