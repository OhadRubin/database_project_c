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


eval "$(ssh-agent -s)"
ssh-add ~/.ssh/tpu_key


RAY_EXEC="/home/$USER/.local/bin/ray"
# Check if Ray is running
if ! command -v $RAY_EXEC &> /dev/null; then
    echo "Ray not found, installing..."
    python3.10 -m pip install ray
fi


HEAD_IP=$(gcloud compute tpus list --zone us-central2-b --filter="name:v4-8-node-2" --format=json | jq -r '.[].ipAddress')
MY_IP=$(hostname -I | awk '{print $1}')

if [ "$MY_IP" = "$HEAD_IP" ]; then
    echo "This is the head node."
    IS_HEAD=true
else
    echo "This is NOT the head node. Head node IP is $HEAD_IP"
    IS_HEAD=false
fi


echo "Checking Ray cluster status..."
if ! $RAY_EXEC status 2>/dev/null | grep -q "Ray runtime started"; then
    echo "Ray cluster is not running. Starting Ray cluster..."
    # Start Ray in head mode
    if $IS_HEAD; then
        $RAY_EXEC start --head --disable-usage-stats
        sleep 20
        echo "Ray cluster started in head mode"
    else
        sleep 20
        $RAY_EXEC start --address="$HEAD_IP:6379" --disable-usage-stats --block
        echo "Ray cluster joined as worker node"
    fi
else
    echo "Ray cluster is already running"
fi

sleep 10


$RAY_EXEC status

SCRIPT="python3.10 database_project/src/deduplication_spark.py --input_file \"/dev/shm/c4_files/c4-train.*.json.gz\" --output /dev/shm/c4_outputs --use_ray True"


# SCRIPT="$SCRIPT --implementation tfidf_minhash"
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

sleep 20
# for NUM_FILES in 1 5 10 20 30 40; do
#     COMMAND="$SCRIPT --limit_files $NUM_FILES"
#     rm -rf /dev/shm/c4_outputs 
#     mkdir -p /dev/shm/c4_outputs
#     echo "$COMMAND"
#     eval "$COMMAND"
# done