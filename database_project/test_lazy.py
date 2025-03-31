"""
this script is to test lazy broadcast variable with multiple nodes using Ray and RayDP.
"""
import os
import argparse
import time

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.10"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.10"

from pyspark import SparkConf
from pyspark.sql import SparkSession
import socket


# Global for lazy initialization on executors
EXECUTOR_BROADCAST_DATA = None

# Simulate a heavy model or resource initialization 
def load_heavy_resource():
    """Simulate loading a heavy resource like a large ML model"""
    time.sleep(5)  # Simulate expensive load operation
    print(f"Broadcast initialized on {socket.gethostname()}")
    return {"lookup_table": {
        "a": "apple",
        "b": "banana",
        "c": "cherry",
        "d": "date"
    }}

def map_func(item):
    """Function that uses the lazy-initialized broadcast variable"""
    global EXECUTOR_BROADCAST_DATA
    
    # Lazy initialization - only happens once per executor process
    if EXECUTOR_BROADCAST_DATA is None:
        EXECUTOR_BROADCAST_DATA = bc_data.value
        print(f"Executor node {socket.gethostname()} accessing broadcast data")
    
    # Use the initialized resource
    lookup = EXECUTOR_BROADCAST_DATA["lookup_table"]
    result = lookup.get(item, "unknown")
    return f"{item} -> {result}"


if __name__ == "__main__":
    # Parse command line arguments
    import ray
    import raydp
    ray.init(address='auto')
    num_nodes = len([x for x in ray.nodes() if x["alive"]])
    print(f"Running with Ray on {num_nodes} nodes")
    
    spark = raydp.init_spark(
        app_name="LazyBroadcastTest",
        num_executors=num_nodes,
        executor_cores=4,
        executor_memory="4g",
        configs={
            'spark.local.dir': '/dev/shm/pyspark_dir',
            'spark.debug.maxToStringFields': '100',
            'spark.driver.memory': '4g',
            'spark.dynamicAllocation.enabled': 'false',
            'spark.python.worker.reuse': 'true'
        }
    )


    # Get SparkContext
    sc = spark.sparkContext
    
    # Create the broadcast variable with heavy resource data
    heavy_data = load_heavy_resource()
    bc_data = sc.broadcast(heavy_data)
    print(f"Broadcast created on driver: {socket.gethostname()}")
    
    # Create an RDD and test the lazy broadcast variable
    start_time = time.time()
    rdd = sc.parallelize(["a", "b", "d"]*100, numSlices=4)
    result = rdd.map(map_func).collect()
    end_time = time.time()
    
    # Show only a sample of results to avoid excessive output
    print("Sample Results:", result[:5])
    print(f"Total Results: {len(result)}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Clean up if using Ray
    if args.use_ray:
        raydp.stop_spark()
        ray.shutdown()
    else:
        spark.stop()
    

"""
To test the lazy initialization of a broadcast variable on multiple nodes, you should:

### Running with Ray
```bash
python3.10 database_project/test_lazy.py --use_ray
```

### Running with standard Spark
```bash
python3.10 database_project/test_lazy.py
```

### Verification
Check the logs to verify:
- The initialization message (`Broadcast initialized...`) should only appear **once**, specifically on the driver node.
- Executor nodes (`Executor node...`) should indicate access without re-initializing the broadcast data.
"""