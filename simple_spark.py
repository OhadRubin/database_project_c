# 
# sudo apt-get install default-jdk -y && python3.10 -m pip install pyspark
# git clone https://github.com/OhadRubin/database_project_c
# python3.10 -m pip install raydp
# python3.10 database_project/src/download_c4.py
import os
from pyspark.sql import SparkSession
import time

# ray start --head
# curl -s https://checkip.amazonaws.com
# ray start --address='master_id:6379'


# Set environment variables to ensure consistent Python version
# This needs to be set before SparkSession is created
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.10"

from pyspark import SparkConf
import fire
def examine_cluster(use_ray=False):
    # Initialize SparkSession with specific configurations for multi-node setup
    if use_ray:
        import ray
        import raydp
        ray.init(address='auto')
        spark = raydp.init_spark(
                app_name="MinHashLSH",
                num_executors=200,
                executor_cores=1, # how many tasks the executor can run in parallel
                executor_memory="2g",
                configs = {
                        'spark.local.dir': '/dev/shm/pyspark_dir',  # TODO: move in arguements
                        'spark.debug.maxToStringFields': '100',

                        # 'spark.ray.raydp_spark_master.actor.resource.CPU': 0,
                        # 'spark.ray.raydp_spark_master.actor.resource.spark_master': 1,  # Force Spark driver related actor run on headnode
                        # 'spark.app.name': 'MinHashLSH',
                        # 'spark.driver.memory': '64g',
                        # 'spark.executor.memory': '2g',
                        # 'spark.submit.deployMode': 'client',
                    })
    else:
        conf = SparkConf()
        conf.set("spark.app.name", "MinHashLSH")
        conf.set("spark.debug.maxToStringFields", "100")
        conf.set("spark.local.dir", "/dev/shm/pyspark_dir") #TODO: move in arguements
        conf.set("spark.driver.memory", "64g")
        conf.set("spark.executor.memory", "64g")
        conf.set("spark.submit.deployMode", "client")
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        
        # Get Spark context
    sc = spark.sparkContext
    
    # Print cluster information
    print("Spark UI URL:", sc.uiWebUrl)
    print("Application ID:", sc.applicationId)
    print("Spark Version:", sc.version)
    print(f"Master: {sc.master}")
    
    # Get simple metrics about the cluster
    print("\nCluster Metrics:")
    print(f"Default parallelism: {sc.defaultParallelism}")
    print(f"Default min partitions: {sc.defaultMinPartitions}")
    
    # Create a simple RDD and perform actions to see distribution
    print("\nPerforming simple distributed computation...")
    
    # Determine number of partitions based on parallelism
    num_partitions = sc.defaultParallelism * 2  # 2 partitions per core
    print(f"Creating RDD with {num_partitions} partitions")
    
    # Create and time a computationally intensive task
    start_time = time.time()
    
    # Create a large RDD with specified partitions (smaller dataset to avoid memory issues)
    rdd = sc.parallelize(range(10000000), numSlices=num_partitions)
    
    # Execute a distributed computation (map and reduce)
    # Square each number and find the sum
    result = rdd.map(lambda x: x * x).reduce(lambda a, b: a + b)
    
    end_time = time.time()
    
    # Print results and timing
    print(f"Computation result: {result}")
    print(f"Computation time: {end_time - start_time:.2f} seconds")
    
    # Show partition information
    print("\nPartition distribution:")
    def count_by_partition(iterator):
        count = 0
        for _ in iterator:
            count += 1
        yield count
        
    partition_counts = rdd.mapPartitions(count_by_partition).collect()
    for i, count in enumerate(partition_counts):
        print(f"  - Partition {i}: {count} elements")
    
    # To test multi-node setup with a more complex operation
    print("\nPerforming a groupBy operation (tests shuffling between nodes)...")
    start_time = time.time()
    
    # Create a modulo-based key for each number and group
    mod_result = rdd.map(lambda x: (x % 100, 1)).reduceByKey(lambda a, b: a + b).collect()
    
    end_time = time.time()
    print(f"GroupBy operation time: {end_time - start_time:.2f} seconds")
    print(f"Number of groups: {len(mod_result)}")
    
    # Clean up
    spark.stop()
    print("\nSpark session stopped")

# python3.10 simple_spark.py --use_ray True
if __name__ == "__main__":
    import fire
    fire.Fire(examine_cluster)
