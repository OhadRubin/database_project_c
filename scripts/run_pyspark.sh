
# bash scripts/run_pyspark.sh
SCRIPT="python3.10 database_project/src/deduplication_spark.py --input_file \"/dev/shm/c4_files/c4-train.*.json.gz\" --output /dev/shm/c4_outputs"

for NUM_FILES in 1 5 10 20 30 40 50 60; do
    COMMAND="$SCRIPT --limit_files $NUM_FILES"
    rm -rf /dev/shm/c4_outputs 
    mkdir -p /dev/shm/c4_outputs
    echo "$COMMAND"
    eval "$COMMAND"
done