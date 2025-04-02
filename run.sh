# Check if Java is installed, if not install it
if ! command -v java &> /dev/null; then
    echo "Java not found, installing..."
    sudo apt-get install default-jdk -y
    python3.10 -m pip install pyspark
    python3.10 -m pip install raydp
    python3.10 -m pip install git+https://github.com/OhadRubin/sparkit-learn
    git clone https://github.com/OhadRubin/database_project_c
else
    echo "Java is already installed"
fi

(cd ~/database_project_c && git pull)
cd ~/database_project_c

source ~/.bashrc
export POSTGRES_ADDRESS="postgresql+psycopg2://postgres:$REDIS_PASSWORD@34.141.239.167:5564/postgres"


# Check if c4_files directory exists in /dev/shm
if [ ! -d "/dev/shm/c4_files" ]; then
    echo "C4 files directory not found, creating it..."
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
else
    echo "C4 files directory already exists at /dev/shm/c4_files"
fi

# export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/tpu_key


RAY_EXEC="/home/$USER/.local/bin/ray"
# Check if Ray is running
if ! command -v $RAY_EXEC &> /dev/null; then
    echo "Ray not found, installing..."
    python3.10 -m pip install ray
fi


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
        $RAY_EXEC start --address="$HEAD_IP:6379" --disable-usage-stats --block --resources='{"TPU": 4}'
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


SCRIPT="python3.10 database_project/src/deduplication_spark.py --input_file \"/dev/shm/c4_files/c4-train.*.json.gz\" --output /dev/shm/c4_outputs --use_ray True"


# SCRIPT="$SCRIPT --implementation tfidf_minhash"
SCRIPT="$SCRIPT --implementation tfidf_minhash_ray"
# SCRIPT="$SCRIPT  --num_perm 1024 --threshold 0.9"

# Run only on head node
if $IS_HEAD; then
    # python3.10 database_project/src/deduplication_spark.py --input_file "/dev/shm/c4_files/c4-train.*.json.gz" --output /dev/shm/c4_outputs --use_ray True --limit_files 1
    # for NUM_FILES in 1 5 10 20 30 40; do
    for NUM_FILES in 40; do
        # COMMAND="$SCRIPT --limit_files $NUM_FILES"
        COMMAND="$SCRIPT --limit_files $NUM_FILES"
        rm -rf /dev/shm/c4_outputs 
        mkdir -p /dev/shm/c4_outputs
        echo "$COMMAND"
        eval "$COMMAND"
    done
else
    echo "Skipping deduplication script on worker node"
fi

sleep 600
# for NUM_FILES in 1 5 10 20 30 40; do
#     COMMAND="$SCRIPT --limit_files $NUM_FILES"
#     rm -rf /dev/shm/c4_outputs 
#     mkdir -p /dev/shm/c4_outputs
#     echo "$COMMAND"
#     eval "$COMMAND"
# done