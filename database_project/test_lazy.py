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

class LazyBroadcast:
    def __init__(self, sc, initializer):
        self.sc = sc
        self.initializer = initializer
        self._broadcast_var = None

    def get(self):
        if self._broadcast_var is None:
            data = self.initializer()
            self._broadcast_var = self.sc.broadcast(data)
        return self._broadcast_var.value

def expensive_computation():
    hostname = socket.gethostname()
    print(f"Broadcast initialized on driver node: {hostname}")
    return {"a": 1, "b": 2, "c": 3}

def map_func(x):
    hostname = socket.gethostname()
    data = lazy_broadcast.get()
    print(f"Executor node {hostname} accessed broadcast variable.")
    return (x, data.get(x, "Not Found"))



if __name__ == "__main__":
    
    # Initialize Spark with Ray or standard configuration

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
        }
    )

    # Get SparkContext
    sc = spark.sparkContext
    
    # Create the lazy broadcast variable
    lazy_broadcast = LazyBroadcast(sc, expensive_computation)
    
    # Create an RDD and test the lazy broadcast variable
    start_time = time.time()
    rdd = sc.parallelize(["a", "b", "d"]*100, numSlices=4)
    result = rdd.map(map_func).collect()
    end_time = time.time()
    
    print("Final Result:", result)
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    

    
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