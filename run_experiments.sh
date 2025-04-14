
# python3.10 database_project/src/run_workflows.py --workflow cl_nd --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --limit_files 1
# log the start time
START_TIME=$(date +%s)


python3.10 -m pip install ray==2.43.0 numpy~=1.0
if [ ! -d "~/database_project_c" ]; then
    git clone https://github.com/OhadRubin/database_project_c
fi

(cd ~/database_project_c && git pull)
cd ~/database_project_c

source ~/.bashrc
export POSTGRES_ADDRESS="postgresql+psycopg2://postgres:$REDIS_PASSWORD@34.141.239.167:5564/postgres"


# Check if c4_files directory exists in /dev/shm
mkdir -p /dev/shm/c4_files
echo "Running C4 download script..."
while true; do
    FILE_COUNT=$(ls /dev/shm/c4_files | wc -l)
    if [ "$FILE_COUNT" -ne 40 ]; then
        echo "Expected 40 files but found $FILE_COUNT files. Running download script again..."
        python3.10 database_project/src/download_c4.py
    else
        echo "Verified 40 files in /dev/shm/c4_files"
        break
    fi
done

# export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"



RAY_EXEC="/home/$USER/.local/bin/ray"
# Check if Ray is running
if ! command -v $RAY_EXEC &> /dev/null; then
    echo "Ray not found, installing..."
    python3.10 -m pip install ray==2.43.0 numpy~=1.0
fi

export RAY_DATA_PUSH_BASED_SHUFFLE=1
NODES_IPS=$(gcloud compute tpus list --zone us-central2-b --filter="name ~ v4-8" --format=json | jq -r '.[].ipAddress')
N_NODES=$(echo "$NODES_IPS" | wc -l)

HEAD_IP=$(gcloud compute tpus list --zone us-central2-b --filter="name:v4-8-node-2" --format=json | jq -r '.[].ipAddress')
MY_IP=$(hostname -I | awk '{print $1}')

if [ "$MY_IP" = "$HEAD_IP" ]; then
    echo "This is the head node."
    IS_HEAD=true
else
    echo "This is NOT the head node. Head node IP is $HEAD_IP"
    IS_HEAD=false
fi



sudo mkdir -p /dev/shm/gcs_cache
sudo chmod 777 /dev/shm/gcs_cache

sudo chown -R $USER:$USER /dev/shm/gcs_cache
sudo umount -l /mnt/gcs_bucket
sleep 1
gcsfuse \
        --implicit-dirs \
        --file-cache-enable-parallel-downloads \
        --file-cache-parallel-downloads-per-file 100 \
        --file-cache-max-parallel-downloads -1 \
        --file-cache-download-chunk-size-mb 10 \
        --file-cache-max-size-mb 200000 \
        --dir-mode 0777 \
        -o allow_other --foreground \
        --cache-dir /dev/shm/gcs_cache  \
        meliad2_us2_backup /mnt/gcs_bucket &> ~/gcs_log.log &
sleep 1


# TPU-v4-8-head

echo "Checking Ray cluster status..."
if ! $RAY_EXEC status 2>/dev/null | grep -q "Ray runtime started"; then
    echo "Ray cluster is not running. Starting Ray cluster..."
    # Start Ray in head mode
    if $IS_HEAD; then
        $RAY_EXEC start --head --disable-usage-stats --resources='{"TPU-v4-8-head": 1}'
        echo "Ray cluster started in head mode"
    else
        $RAY_EXEC start --address="$HEAD_IP:6379" --disable-usage-stats --block --resources='{"TPU-v4-8-head": 1}'
        echo "Ray cluster joined as worker node"
    fi
else
    echo "Ray cluster is already running"
fi


# Wait until we have 10 nodes in the Ray cluster
echo "Waiting for $N_NODES nodes to join the Ray cluster..."
while true; do
    NODE_COUNT=$(python3.10 -c 'import ray; ray.init(address="auto"); print(len([x for x in ray.nodes() if x["alive"]]))')
    echo "Current node count: $NODE_COUNT"
    if [ "$NODE_COUNT" -ge "$N_NODES" ]; then
        echo "All $N_NODES nodes are now available in the Ray cluster"
        break
    fi
    echo "Waiting for more nodes to join... (sleeping 10 seconds)"
    sleep 1
done
$RAY_EXEC status
# print here how long we waited
# Calculate and log the time elapsed since START_TIME
if [ -n "$START_TIME" ]; then
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    ELAPSED_MINUTES=$((ELAPSED_TIME / 60))
    ELAPSED_SECONDS=$((ELAPSED_TIME % 60))
    echo "Time elapsed since start: $ELAPSED_MINUTES minutes and $ELAPSED_SECONDS seconds"
else
    echo "START_TIME not set, cannot calculate elapsed time"
fi

SCRIPT="python3.10 database_project/src/deduplication_spark.py --input_file \"/dev/shm/c4_files/c4-train.*.json.gz\" --output /dev/shm/c4_outputs"
SCRIPT="$SCRIPT --implementation tfidf_minhash_ray"


# WORKFLOW="nd_cl"
WORKFLOW="cl_nd"

SCRIPT="python3.10 database_project/src/run_workflows.py  --dedup_mode tag --workflow $WORKFLOW --input_file \"/dev/shm/c4_files/c4-train.*.json.gz\" --output /dev/shm/c4_outputs"





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
    --notes \"$notes\" "

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

# SCRIPT="$SCRIPT --implementation tfidf_minhash"
# SCRIPT="$SCRIPT  --num_perm 1024 --threshold 0.9"

# Run only on head node
if $IS_HEAD; then
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
    # # for NUM_FILES in 1 5 10 20 30 40; do
    # for NUM_FILES in 1; do
    #     COMMAND="$SCRIPT --limit_files $NUM_FILES"
    #     rm -rf /dev/shm/c4_outputs 
    #     mkdir -p /dev/shm/c4_outputs
    #     echo "$COMMAND"
    #     eval "$COMMAND"
    # done
else
    echo "Skipping deduplication script on worker node"
fi

sleep 6000
