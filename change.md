Okay, I will generate the requested files with the changes applied based on the detailed plans in the `changes/*.md` files.

```python
# database_project/src/db.py
# --- BEGIN FILE ---
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, object_session as sa_object_session # Import object_session directly
from datetime import datetime
import os
import argparse # Added for type hint in create_from_args
import json # Added for Change 4 type hint (though not strictly needed for signature)

Base = declarative_base()

class BenchmarkRun(Base):
    __tablename__ = 'benchmark_runs'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_file = Column(String(255))
    output_dir = Column(String(255))
    notes = Column(Text, nullable=True)
    duplicate_count = Column(Integer)
    record_count = Column(Integer) # Final record count after all steps
    implementation = Column(String(50)) # e.g., "nd_cl", "cl_nd"
    num_nodes = Column(Integer)
    # ND Params
    threshold = Column(Float)
    ngram_size = Column(Integer)
    min_ngram_size = Column(Integer)
    num_perm = Column(Integer)
    # Timing & Size
    execution_time = Column(Float) # Total wall-clock time for the workflow
    limit_files = Column(Integer, nullable=True)
    total_size_gb = Column(Float, nullable=True)
    # Change 1: ND Step Time
    nd_time_sec = Column(Float, nullable=True) # Time for the ND step in nd_cl
    # Change 3: Intermediate Count after ND
    nd_output_count = Column(Integer, nullable=True) # Record count after ND in nd_cl
    # Change 2: Clustering Config Path
    config_file_path = Column(String(512), nullable=True) # Path to the cluster config YAML
    # Change 6: Clustering Time Breakdown
    cl_train_time_sec = Column(Float, nullable=True) # Time for CL model training
    cl_inference_time_sec = Column(Float, nullable=True) # Time for CL model inference
    # Change 7: Stage 2 Time
    cl_stage2_time_sec = Column(Float, nullable=True) # Time specifically for stage2 execution
    # Change 4: Full Config Details
    config_details_json = Column(Text, nullable=True) # JSON string of args + cfg
    # Change 7: Cluster Distribution
    cluster_size_distribution_json = Column(Text, nullable=True) # JSON string of final cluster sizes

    # Relationships
    resource_metrics = relationship("ResourceMetric", back_populates="benchmark_run", cascade="all, delete-orphan")
    accuracy_metrics = relationship("AccuracyMetric", back_populates="benchmark_run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<BenchmarkRun(id={self.id}, timestamp={self.timestamp}, implementation={self.implementation})>"

    # Note: create_from_spark_run is kept for potential backward compatibility or other uses
    @classmethod
    def create_from_spark_run(cls, session, input_file, output_dir, duplicate_count, record_count,
                              threshold, ngram_size, min_ngram_size, num_perm, execution_time,
                              num_nodes=1, notes=None, implementation="pyspark", limit_files=None, total_size_gb=None):
        """
        Create a new BenchmarkRun entry from a PySpark deduplication run
        (This method is kept but may need updates if used with new fields)
        """
        run = cls(
            input_file=input_file,
            output_dir=output_dir,
            duplicate_count=duplicate_count,
            record_count=record_count,
            implementation=implementation,
            num_nodes=num_nodes,
            threshold=threshold,
            ngram_size=ngram_size,
            min_ngram_size=min_ngram_size,
            num_perm=num_perm,
            execution_time=execution_time,
            notes=notes,
            limit_files=limit_files,
            total_size_gb=total_size_gb
            # New fields would need to be added here if this method is used
        )
        session.add(run)
        session.commit()
        return run

    @classmethod
    def create_from_args(cls, session, args: argparse.Namespace, duplicate_count: int, record_count: int, execution_time: float,
                       num_nodes: int = 1, notes: str = None, implementation: str = "pyspark", limit_files: int = None, total_size_gb: float = None,
                       nd_time_sec: float = None, # Change 1
                       config_file_path: str = None, # Change 2
                       nd_output_count: int = None, # Change 3
                       config_details_json: str = None, # Change 4
                       cl_train_time_sec: float = None, # Change 6
                       cl_inference_time_sec: float = None, # Change 6
                       cl_stage2_time_sec: float = None, # Change 7
                       cluster_size_distribution_json: str = None # Change 7
                       ):
        """
        Create a new BenchmarkRun entry from command line args and results, including added metrics.

        Parameters:
        -----------
        session : SQLAlchemy session
            Database session
        args : argparse.Namespace
            Command line arguments from create_parser()
        duplicate_count : int
            Number of duplicate sets found (interpretation depends on workflow)
        record_count : int
            Final records after all steps of the workflow
        execution_time : float
            Total wall clock runtime in seconds for the workflow
        num_nodes : int, optional
            Number of nodes detected in the Ray cluster
        notes : str, optional
            Additional notes about the run
        implementation : str, optional
            Implementation type (e.g., 'nd_cl', 'cl_nd'), defaults to "pyspark" for historical reasons
        limit_files : int, optional
            Number of files processed (if limited)
        total_size_gb : float, optional
            Total size of processed files in GB (currently needs calculation)
        nd_time_sec : float, optional (Change 1)
            Execution time of the ND step (primarily for nd_cl).
        config_file_path : str, optional (Change 2)
            Path to the clustering configuration YAML file.
        nd_output_count : int, optional (Change 3)
            Number of records output by the ND step (primarily for nd_cl).
        config_details_json : str, optional (Change 4)
            JSON string containing combined argparse args and loaded config dict.
        cl_train_time_sec : float, optional (Change 6)
            Time spent training clustering models.
        cl_inference_time_sec : float, optional (Change 6)
            Time spent applying clustering models (inference).
        cl_stage2_time_sec : float, optional (Change 7)
            Wall-clock time spent specifically in the stage2 function.
        cluster_size_distribution_json : str, optional (Change 7)
            JSON string representing the final distribution of records per cluster.

        Returns:
        --------
        BenchmarkRun
            The created BenchmarkRun instance (still attached to the session before commit).
        """
        run = cls(
            # Standard Params
            input_file=args.input_file or getattr(args, 'table', None), # Handle potential table arg
            output_dir=args.output,
            duplicate_count=duplicate_count,
            record_count=record_count, # Final count
            implementation=implementation, # Workflow name is passed here
            num_nodes=num_nodes,
            threshold=args.threshold,
            ngram_size=args.ngram_size,
            min_ngram_size=args.min_ngram_size,
            num_perm=args.num_perm,
            execution_time=execution_time, # Total workflow time
            notes=notes,
            limit_files=limit_files if limit_files is not None else args.limit_files,
            total_size_gb=total_size_gb,
            # Added Params from Changes
            nd_time_sec=nd_time_sec,                         # Change 1
            config_file_path=config_file_path,               # Change 2
            nd_output_count=nd_output_count,                 # Change 3
            config_details_json=config_details_json,         # Change 4
            cl_train_time_sec=cl_train_time_sec,             # Change 6
            cl_inference_time_sec=cl_inference_time_sec,     # Change 6
            cl_stage2_time_sec=cl_stage2_time_sec,           # Change 7
            cluster_size_distribution_json=cluster_size_distribution_json # Change 7
        )
        session.add(run)
        session.commit() # Commit the main run record
        # Return the committed object (its ID is now populated)
        # The caller might need to re-fetch or merge if adding related objects later in the same session scope.
        # For adding ResourceMetric, the object might still be usable if session remains active.
        return run

    def add_resource_metrics(self, cpu_percent_avg, cpu_percent_max, memory_usage_avg_mb,
                           memory_usage_max_mb, network_sent_mb=0, network_recv_mb=0,
                           disk_read_mb=0, disk_write_mb=0, session=None): # Added session parameter for flexibility
        """
        Add resource metrics for this benchmark run.
        Requires the BenchmarkRun object to be associated with a session.

        Parameters:
        -----------
        cpu_percent_avg : float
            Average CPU usage
        cpu_percent_max : float
            Maximum CPU usage
        memory_usage_avg_mb : float
            Average memory usage in MB
        memory_usage_max_mb : float
            Maximum memory usage in MB
        network_sent_mb : float, optional
            Network data sent in MB
        network_recv_mb : float, optional
            Network data received in MB
        disk_read_mb : float, optional
            Disk data read in MB
        disk_write_mb : float, optional
            Disk data written in MB
        session : SQLAlchemy session, optional
            Database session, if the object is detached or needs explicit session.

        Returns:
        --------
        ResourceMetric
            The created ResourceMetric instance
        """
        current_session = session or sa_object_session(self) # Use provided session or get object's session
        if not current_session:
             raise Exception("No session found for BenchmarkRun object. Cannot add resource metrics.")

        resource_metric = ResourceMetric(
            result_id=self.id, # Assumes self.id is populated (object was committed)
            cpu_percent_avg=cpu_percent_avg,
            cpu_percent_max=cpu_percent_max,
            memory_usage_avg_mb=memory_usage_avg_mb,
            memory_usage_max_mb=memory_usage_max_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb
        )
        # The relationship automatically handles appending if the BenchmarkRun object is managed
        self.resource_metrics.append(resource_metric)
        # The commit should happen *after* calling this method in the main script (as per Change 5 plan)
        # current_session.commit() # Removed commit from here, caller should commit.
        return resource_metric

    # add_accuracy_metrics remains unchanged from original, but might need session handling like add_resource_metrics if used.
    def add_accuracy_metrics(self, reference_implementation, true_positives, false_positives,
                           false_negatives, precision, recall, f1_score, session=None):
        """
        Add accuracy metrics for this benchmark run.
        Requires the BenchmarkRun object to be associated with a session.
        """
        current_session = session or sa_object_session(self)
        if not current_session:
             raise Exception("No session found for BenchmarkRun object. Cannot add accuracy metrics.")

        accuracy_metric = AccuracyMetric(
            result_id=self.id,
            reference_implementation=reference_implementation,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )
        self.accuracy_metrics.append(accuracy_metric)
        # current_session.commit() # Caller should commit
        return accuracy_metric

class ResourceMetric(Base):
    __tablename__ = 'resource_metrics'

    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('benchmark_runs.id'))
    cpu_percent_avg = Column(Float)
    cpu_percent_max = Column(Float)
    memory_usage_avg_mb = Column(Float)
    memory_usage_max_mb = Column(Float)
    network_sent_mb = Column(Float, default=0) # Added default
    network_recv_mb = Column(Float, default=0) # Added default
    disk_read_mb = Column(Float, default=0) # Added default
    disk_write_mb = Column(Float, default=0) # Corrected from 'resulte_mb', added default

    # Relationship
    benchmark_run = relationship("BenchmarkRun", back_populates="resource_metrics")

    def __repr__(self):
        return f"<ResourceMetric(id={self.id}, result_id={self.result_id})>"

class AccuracyMetric(Base):
    __tablename__ = 'accuracy_metrics'

    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('benchmark_runs.id'))
    reference_implementation = Column(String(100))
    true_positives = Column(Integer)
    false_positives = Column(Integer)
    false_negatives = Column(Integer)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)

    # Relationship
    benchmark_run = relationship("BenchmarkRun", back_populates="accuracy_metrics")

    def __repr__(self):
        return f"<AccuracyMetric(id={self.id}, result_id={self.result_id}, f1_score={self.f1_score})>"


def init_db(db_path=None):
    """Initialize the database, create tables if they don't exist"""
    if db_path is None:
        if "POSTGRES_ADDRESS" in os.environ:
            print("Using PostgreSQL connection:", os.environ["POSTGRES_ADDRESS"])
            db_path = os.environ["POSTGRES_ADDRESS"]
        else:
            default_db_file = 'benchmark_results.db'
            print(f"Using SQLite database: {default_db_file}")
            db_path = f'sqlite:///{default_db_file}'
    engine = create_engine(db_path)
    try:
        Base.metadata.create_all(engine) # This creates tables if they don't exist
        print("Database tables verified/created.")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise # Re-raise the exception to halt if DB setup fails critically
    return engine

def get_session(engine):
    """Create a session to interact with the database"""
    Session = sessionmaker(bind=engine)
    return Session()

# Helper function to get the object's session (already imported at top)
# def object_session(obj):
#     """Get the session for an object"""
#     return sa_object_session(obj)

# Example usage for monitoring resource stats while running a benchmark (kept for reference)
# Note: This function is standalone and not directly used by the main workflow script's Change 5 implementation.
# Change 5 uses an inline thread in run_workflows.py.
def monitor_resources(benchmark_run_id, session, interval=1.0):
    """
    Monitor system resources and add to database
    Requires psutil library
    """
    try:
        import psutil
        import time
        import statistics

        benchmark_run = session.query(BenchmarkRun).get(benchmark_run_id)
        if not benchmark_run:
            print(f"Benchmark run with ID {benchmark_run_id} not found")
            return

        cpu_percent = []
        memory_percent = []
        start_time = time.time()

        # Get initial disk and network counters
        initial_disk_io = psutil.disk_io_counters()
        initial_net_io = psutil.net_io_counters()

        try:
            print("Monitoring resources... Press Ctrl+C to stop.")
            while True:
                cpu_percent.append(psutil.cpu_percent())
                memory_info = psutil.virtual_memory()
                memory_percent.append(memory_info.percent)
                time.sleep(interval)
        except KeyboardInterrupt:
            # Calculate resource metrics
            run_time = time.time() - start_time

            # Calculate disk and network usage
            final_disk_io = psutil.disk_io_counters()
            final_net_io = psutil.net_io_counters()

            disk_read_mb = (final_disk_io.read_bytes - initial_disk_io.read_bytes) / (1024 * 1024) if initial_disk_io else 0
            disk_write_mb = (final_disk_io.write_bytes - initial_disk_io.write_bytes) / (1024 * 1024) if initial_disk_io else 0
            net_sent_mb = (final_net_io.bytes_sent - initial_net_io.bytes_sent) / (1024 * 1024) if initial_net_io else 0
            net_recv_mb = (final_net_io.bytes_recv - initial_net_io.bytes_recv) / (1024 * 1024) if initial_net_io else 0

            # Get system memory info to convert percent to MB
            memory_info = psutil.virtual_memory()
            total_memory_mb = memory_info.total / (1024 * 1024)

            avg_memory_percent = statistics.mean(memory_percent) if memory_percent else 0
            max_memory_percent = max(memory_percent) if memory_percent else 0

            avg_memory_mb = (avg_memory_percent / 100) * total_memory_mb
            max_memory_mb = (max_memory_percent / 100) * total_memory_mb

            avg_cpu = statistics.mean(cpu_percent) if cpu_percent else 0
            max_cpu = max(cpu_percent) if cpu_percent else 0

            # Add resource metrics to database using the method
            print("Adding resource metrics to database...")
            try:
                 # Pass the session explicitly if needed, or rely on object_session
                benchmark_run.add_resource_metrics(
                    cpu_percent_avg=avg_cpu,
                    cpu_percent_max=max_cpu,
                    memory_usage_avg_mb=avg_memory_mb,
                    memory_usage_max_mb=max_memory_mb,
                    network_sent_mb=net_sent_mb,
                    network_recv_mb=net_recv_mb,
                    disk_read_mb=disk_read_mb,
                    disk_write_mb=disk_write_mb,
                    session=session # Pass session explicitly
                )
                session.commit() # Commit after adding metrics
                print("Resource metrics added successfully.")
            except Exception as e:
                print(f"Error adding resource metrics: {e}")
                session.rollback()


            print(f"Resource monitoring completed after {run_time:.2f} seconds")

    except ImportError:
        print("psutil library required for resource monitoring. Install with: pip install psutil")
    except Exception as e:
        print(f"An error occurred during resource monitoring: {e}")

if __name__ == '__main__':
    # Example usage
    print("Initializing DB from main...")
    engine = init_db()
    session = get_session(engine)
    print("Database initialized successfully.")

    # Example: Add a dummy run and monitor (requires manual Ctrl+C)
    # from types import SimpleNamespace
    # dummy_args = SimpleNamespace(input_file='dummy', output='dummy', table=None, threshold=0.7, ngram_size=5, min_ngram_size=5, num_perm=256, limit_files=None)
    # try:
    #     print("Creating dummy BenchmarkRun...")
    #     dummy_run = BenchmarkRun.create_from_args(session, dummy_args, 0, 0, 0.0, notes='Dummy run for monitor test', implementation='test')
    #     print(f"Dummy run created with ID: {dummy_run.id}")
    #     # Start monitoring - requires manual stop (Ctrl+C)
    #     monitor_resources(dummy_run.id, session)
    # except Exception as e:
    #      print(f"Error in dummy run creation/monitoring: {e}")
    # finally:
    #     session.close()

# --- END FILE ---
```

```python
# database_project/src/ray_minhash.py
# --- BEGIN FILE ---
# Adapted from https://github.com/modelscope/data-juicer
import ray
import sys
import logging
import time
import sys
import os
import os
import time
from typing import List, Optional, Union, Set, Iterable, Dict, Tuple, Any # Added Set, Iterable, Dict, Tuple, Any
import re
import numpy as np
import pyarrow as pa
import ray
import regex # Note: imported 'regex' but used 're' - standardizing to 're'
from loguru import logger
from pydantic import Field, PositiveInt # Removed: Not used directly in modified code
from typing_extensions import Annotated # Removed: Not used directly in modified code

import hashlib
import struct
from collections import defaultdict
from typing import Optional
import scipy.integrate as integrate
from itertools import tee # Added for ngrams
from functools import partial # Added for tokenize func

# python3.10 -m pip install ray==2.43.0 numpy~=1.0 scipy loguru pyarrow

# Setup basic logging if loguru is not the primary logger elsewhere
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__) # Using standard logging


MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
    """
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from
    datasketch.

    :param threshold: float. The threshold for similarity
    :param num_perm: int. The number of permutations
    :param false_positive_weight: float. The weight of false positive
    :param false_negative_weight: float. The weight of false negative
    :return: Tuple[int, int]. The optimal `b` and `r` parameters. The number of
        bands, and the number of rows per band respectively
    """

    def false_positive_probability(th: float, band: int, rows: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s**float(rows))**float(band)

        a, _ = integrate.quad(proba, 0.0, th)
        return a

    def false_negative_probability(th: float, band: int, rows: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s**float(rows))**float(band))

        a, _ = integrate.quad(proba, th, 1.0)
        return a

    # object: minimize the weighted FP and FN ratio
    min_error = float('inf')
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        if max_r == 0: continue # Avoid r=0 range
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    if opt == (0,0):
        # Fallback if no valid params found (e.g., num_perm too small)
        log.warning(f"Could not find optimal parameters for threshold={threshold}, num_perm={num_perm}. Falling back to b=1, r=num_perm")
        opt = (1, num_perm)
    return opt


BATCH_SIZE = 1000


@ray.remote
class IdGenerator:

    def __init__(self, start_id=0):
        self.next_id = start_id

    @ray.method(num_returns=2)
    def get_next_id(self, count):
        current_id = self.next_id
        self.next_id += count
        return (current_id, self.next_id)


@ray.remote(scheduling_strategy='SPREAD')
class EdgeBuffer:

    def __init__(self):
        self.edge_dict = {}

    def clear(self):
        self.edge_dict = {}

    def set_edges(self, edge_dict):
        self.edge_dict = edge_dict

    def get_edges(self, key):
        return self.edge_dict.pop(key, [])


@ray.remote(scheduling_strategy='SPREAD')
class BTSUnionFind:
    """
    A distributed implementation of Union-Find with load balancing.

    The original paper on BTS Union-Find is available at:
    https://ieeexplore.ieee.org/document/10598116
    """

    def __init__(
        self,
        union_threshold,
        parallel_num,
        parallel_id,
        remote_edge_buffers,
        max_pending_edge_buffer_task,
        num_edge_buffer_task_returns,
    ):
        self.union_threshold = union_threshold
        self.parallel_num = parallel_num
        self.parallel_id = parallel_id
        self.hash_table = {}
        self.parent = {}
        self.old_parent = {}
        self.remote_edge_buffers = remote_edge_buffers
        self.edge_buffer = []
        self.edge_list_dict = {}
        self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
        self.num_edge_buffer_task_returns = num_edge_buffer_task_returns

    def add_key_value_pairs(self, pairs):
        for key, value in pairs:
            if key not in self.hash_table:
                self.hash_table[key] = []
            self.hash_table[key].append(value)
            if len(self.hash_table[key]) > self.union_threshold:
                self.hash_table[key] = [self.union_list(self.hash_table[key])]

    def flush_key_value_pairs(self):
        for value in self.hash_table.values():
            if len(value) > 1:
                self.union_list(value)
        # Clear the hash table to free memory after flushing
        self.hash_table = {}


    def balanced_union_find(self):
        for x, y in self.edge_buffer:
            self.union(x, y)
        self.edge_buffer = []
        result_refs = []
        for remote_edge_buffer in self.remote_edge_buffers:
            if len(result_refs) >= self.max_pending_edge_buffer_task: # Use >= for safety
                # Wait for some tasks to complete before submitting more
                num_returns = min(self.num_edge_buffer_task_returns, len(result_refs))
                ready_refs, result_refs = ray.wait(
                    result_refs, num_returns=num_returns)
                edge_list = ray.get(ready_refs)
                for edges in edge_list:
                    for x, y in edges:
                        self.union(x, y)
                del ready_refs # Free memory

            result_refs.append(
                remote_edge_buffer.get_edges.remote(self.parallel_id))

        # Process remaining tasks
        if result_refs:
            edge_list = ray.get(result_refs)
            for edges in edge_list:
                for x, y in edges:
                    self.union(x, y)
            del edge_list, result_refs # Free memory

        self.rebalancing()
        # Check if parent pointers have changed
        changed = self.old_parent != self.parent
        return changed


    def distribute_edge(self, u, v):
        # Ensure parallel_num is not zero
        if self.parallel_num == 0:
            raise ValueError("parallel_num cannot be zero.")

        hash_u = u // BATCH_SIZE % self.parallel_num
        hash_v = v // BATCH_SIZE % self.parallel_num
        if hash_u not in self.edge_list_dict:
            self.edge_list_dict[hash_u] = []
        self.edge_list_dict[hash_u].append((u, v))
        if hash_u != hash_v:
            if hash_v not in self.edge_list_dict:
                self.edge_list_dict[hash_v] = []
            self.edge_list_dict[hash_v].append((u, v))

    def set_edge_buffer(self):
        if self.parallel_id in self.edge_list_dict:
            self.edge_buffer = self.edge_list_dict.pop(self.parallel_id) # Use pop to remove
        else:
            self.edge_buffer = []
        # Send remaining edges to remote buffers
        ray.get(self.remote_edge_buffers[self.parallel_id].set_edges.remote(
            self.edge_list_dict))
        # Clear the local dict after sending
        self.edge_list_dict = {}


    def edge_redistribution(self):
        self.flush_key_value_pairs()
        self.rebalancing()
        self.edge_list_dict = {}
        # Distribute current parent relationships as edges
        for u, v in self.parent.items():
            # Ensure u and v are different to avoid self-loops if parent[u]=u
            if u != v:
                 self.distribute_edge(u, v)
        self.parent = {} # Clear local parent state after distributing
        self.set_edge_buffer()

    def communication(self):
        self.edge_list_dict = {}
        del_list = []
        # Compare current parent with old parent to find changed edges
        # Also send edges if the node is not local and the parent is not locally known
        for u, v in self.parent.items():
            hash_u = u // BATCH_SIZE % self.parallel_num
            # Check if parent changed OR if it's a remote node whose parent isn't stored locally
            # This ensures updates propagate correctly
            if v != self.old_parent.get(u, u) or (hash_u != self.parallel_id and v not in self.parent):
                if u != v: # Avoid self-loops
                    self.distribute_edge(u, v)

            # Mark remote nodes for deletion from local parent dict
            if hash_u != self.parallel_id:
                del_list.append(u)

        # Update old_parent state *before* deleting remote nodes
        self.old_parent = self.parent.copy()

        # Remove remote nodes from the current parent dictionary
        for u in del_list:
            # Need to check if key exists before deleting, as it might have been deleted indirectly
            if u in self.parent:
                 del self.parent[u]

        self.set_edge_buffer()


    def find(self, x):
        # Path compression implementation
        if x not in self.parent or self.parent[x] == x: # Base case: not in dict or points to self
             # Ensure the node exists as its own parent if queried directly
             if x not in self.parent:
                 self.parent[x] = x
             return x
        else:
            # Find the root recursively and update the parent pointer
            root = self.find(self.parent[x])
            self.parent[x] = root
            return root

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return # Already in the same set
        # Union by rank/size heuristic (using value comparison as proxy)
        if px > py:
            px, py = py, px # Ensure px is the smaller root
        self.parent[py] = px # Attach larger root (py) to smaller root (px)


    def union_list(self, x_list):
        if not x_list: return None # Handle empty list case
        px_list = [self.find(x) for x in x_list]
        p = min(px_list) # Find the overall minimum root
        for px in px_list:
            if p != px:
                self.parent[px] = p # Point all roots to the minimum root
        return p

    def rebalancing(self):
        new_px_dict = {}
        # Find representative for each (root, partition_id) pair
        for x in list(self.parent.keys()): # Iterate over keys snapshot as dict might change
            hash_x = x // BATCH_SIZE % self.parallel_num
            px = self.find(x) # Find the ultimate root with path compression
            key = (px, hash_x)
            if key not in new_px_dict or x < new_px_dict[key]:
                 new_px_dict[key] = x # Keep the minimum element as representative

        # Ensure root node itself is considered for representation
        # Using set to avoid iterating duplicates if multiple nodes share a root
        roots = set(self.parent.values()) | set(self.parent.keys()) # Consider all nodes involved
        for px in roots:
             # If a node is its own root, ensure it's represented
             if px not in self.parent or self.parent[px] == px:
                hash_px = px // BATCH_SIZE % self.parallel_num
                key = (px, hash_px)
                if key not in new_px_dict or px < new_px_dict[key]:
                    new_px_dict[key] = px

        # Update parent pointers to point to the representative
        for x in list(self.parent.keys()):
            hash_x = x // BATCH_SIZE % self.parallel_num
            px = self.find(x) # Use find to get the current root
            key = (px, hash_x)
            representative = new_px_dict.get(key)
            if representative is not None and x != representative:
                self.parent[x] = representative


    def squeeze(self):
        # Keep only parent entries where the key belongs to the current partition
        self.parent = {
            x: v
            for x, v in self.parent.items()
            if x // BATCH_SIZE % self.parallel_num == self.parallel_id
        }
        # Reset old state and buffers for the next round
        self.old_parent = {}
        self.edge_buffer = []
        # Clear the corresponding remote edge buffer as well
        ray.get(self.remote_edge_buffers[self.parallel_id].clear.remote())

    def dup_idx(self, queries):
        # Return indices of queries whose UID is found as a *non-root* element in the parent dict
        # This indicates it has been merged with another element.
        # A node is a duplicate if it exists in `parent` and `parent[x]` is not `x`.
        duplicate_indices = []
        for uid, idx in queries:
            # Ensure find is called to potentially populate parent dict for roots
            root = self.find(uid)
            # Check if the node exists in parent and its parent is different from itself
            if uid in self.parent and self.parent[uid] != uid:
                 duplicate_indices.append(idx)
            # Alternative check: If find(uid) != uid, it's potentially a duplicate.
            # But the above check is more direct based on the final parent state.
        return duplicate_indices


NON_ALPHA = re.compile(r"[^A-Za-z_0-9\s]+") # Keep spaces for splitting
WHITESPACE = re.compile(r"\s+")

def ngrams(sequence: List[str], n: int, min_length: int = 1) -> Iterable:
    """
    Generates n-grams from a sequence of items (tokens).
    Handles sequences shorter than n by returning an empty list.
    Uses itertools.tee for efficiency.

    :param sequence: List of items (e.g., words).
    :param n: The order of the n-grams.
    :param min_length: The minimum length of the sequence to generate n-grams.
                       Defaults to 1, meaning n-grams are generated if len(sequence) >= n.
                       Set higher (e.g., =n) if you require at least n tokens.
    :return: An iterable of n-gram tuples.
    """
    if len(sequence) < n or len(sequence) < min_length:
        return []

    iterables = tee(sequence, n)
    # Advance each iterator to its starting position for the n-gram window
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def tokenize(content: str, ngram_size: int, min_ngram_size: int) -> Set[bytes]:
    """
    Tokenizes content into n-grams.
    1. Lowercases the text.
    2. Removes non-alphanumeric characters (excluding spaces).
    3. Splits into words based on whitespace.
    4. Generates n-grams from the words.
    5. Encodes each n-gram tuple into a space-separated UTF-8 byte string.

    :param content: The input string.
    :param ngram_size: The size of the n-grams (e.g., 5).
    :param min_ngram_size: The minimum number of words required in the document
                           to generate n-grams.
    :return: A set of byte strings representing the n-grams.
    """
    if not isinstance(content, str):
        log.warning(f"Received non-string content: {type(content)}. Skipping tokenization.")
        return set()

    text = content.lower()
    text = NON_ALPHA.sub("", text) # Remove non-alphanumeric, keeping spaces
    words = WHITESPACE.split(text.strip()) # Split by whitespace
    words = [word for word in words if word] # Remove empty strings resulting from multiple spaces

    # Generate n-grams only if the number of words meets the minimum requirement
    if len(words) < min_ngram_size:
         return set()

    tokens = {
        " ".join(t).encode("utf-8")
        for t in ngrams(words, ngram_size, min_length=ngram_size) # Use min_length=ngram_size
    }
    return tokens


class RayBTSMinhashDeduplicator:
    """
    A MinhashLSH deduplicator based on RAY BTS Union-Find.
    """
    EMPTY_HASH_VALUE = 'EMPTY_HASH_PLACEHOLDER' # Use a distinct placeholder
    _batched_op = True # Indicates compatibility with map_batches

    def __init__(
        self,
        text_key: str = 'text', # Added text_key to signature
        ngram_size: int = 5,
        min_ngram_size: int = 5,
        num_permutations: int = 256,
        jaccard_threshold: float = 0.7,
        num_bands: Optional[int] = None,
        num_rows_per_band: Optional[int] = None,
        union_find_parallel_num: Union[int, str] = 'auto',
        union_threshold: int = 256, # Make Optional consistent with description
        max_pending_edge_buffer_task: int = 20,
        num_edge_buffer_task_returns: int = 10,
        max_pending_filter_tasks: int = 20,
        num_filter_task_returns: int = 10,
        merge_batch_size: int = 1000,
        hashing_batch_size: int = 10000, # Added hashing batch size
        **kwargs, # Allow passing extra args like batch_size
    ):
        """
        Initialization method.

        :param text_key: Key in the input data containing the text to process.
        :param ngram_size: window size of shingling (n-gram size).
        :param min_ngram_size: Minimum number of tokens required to generate n-grams.
        :param num_permutations: number of permutations in minhash computing.
        :param jaccard_threshold: the min jaccard similarity threshold.
        :param num_bands: number of bands in LSH. If None, computed optimally.
        :param num_rows_per_band: number of rows per band in LSH. If None, computed optimally.
        :param union_find_parallel_num: number of parallel workers for union-find. 'auto' uses half CPUs.
        :param union_threshold: threshold for minhash values group to perform union-find locally.
        :param max_pending_edge_buffer_task: max number of pending edge buffer ray tasks for ray.wait.
        :param num_edge_buffer_task_returns: number of edge buffer tasks for `ray.wait` to return.
        :param max_pending_filter_tasks: max number of pending filter ray tasks for ray.wait.
        :param num_filter_task_returns: number of filter tasks for `ray.wait` to return.
        :param merge_batch_size: batch size for BTS operations (ray.wait).
        :param hashing_batch_size: Batch size for the initial minhash calculation map_batches.
        """
        self.text_key = text_key
        # self.work_dir = kwargs.get('work_dir', None) # Not used
        self.batch_size = kwargs.get('batch_size', 1000) # General batch size, potentially unused if map_batches sizes are specific
        self.hashing_batch_size = hashing_batch_size
        self.min_ngram_size = min_ngram_size

        # Setup tokenization function
        self.tokenization_func = partial(tokenize, ngram_size=ngram_size, min_ngram_size=min_ngram_size)

        # Deduplication parameters
        self.num_permutation = num_permutations
        self.jaccard_threshold = jaccard_threshold

        # Initialize LSH parameters
        if num_bands is None or num_rows_per_band is None:
            log.info(f"Calculating optimal LSH params for threshold={self.jaccard_threshold}, num_perm={self.num_permutation}")
            self.num_bands, self.num_rows_per_band = optimal_param(
                self.jaccard_threshold,
                self.num_permutation,
            )
            # Ensure calculated params are valid
            if self.num_bands * self.num_rows_per_band > self.num_permutation:
                 log.warning(f"Optimal bands ({self.num_bands}) * rows ({self.num_rows_per_band}) > num_perm ({self.num_permutation}). Adjusting rows.")
                 self.num_rows_per_band = self.num_permutation // self.num_bands
            if self.num_bands <= 0 or self.num_rows_per_band <= 0:
                 log.error("Invalid LSH parameters calculated. Check optimal_param logic or inputs.")
                 # Fallback to default safe values
                 self.num_bands = max(1, self.num_bands)
                 self.num_rows_per_band = self.num_permutation // self.num_bands
                 if self.num_rows_per_band == 0: self.num_rows_per_band = 1 # Ensure at least 1 row
                 log.warning(f"Using fallback LSH params: bands={self.num_bands}, rows={self.num_rows_per_band}")

        else:
            self.num_bands = num_bands
            self.num_rows_per_band = num_rows_per_band
            if self.num_bands * self.num_rows_per_band > self.num_permutation:
                log.warning(f"Provided bands ({self.num_bands}) * rows ({self.num_rows_per_band}) > num_perm ({self.num_permutation}). Ensure this is intended.")
            if self.num_bands <= 0 or self.num_rows_per_band <= 0:
                raise ValueError("num_bands and num_rows_per_band must be positive.")

        log.info(f"Using LSH params: threshold={self.jaccard_threshold}, num_perm={self.num_permutation}, bands={self.num_bands}, rows={self.num_rows_per_band}")


        # Hash ranges for LSH bands
        self.hash_ranges = [(i * self.num_rows_per_band,
                             (i + 1) * self.num_rows_per_band)
                            for i in range(self.num_bands)]

        # Generate permutations for MinHash
        gen = np.random.RandomState(seed=42) # Consistent seed
        self.perm_a, self.perm_b = np.array(
            [(
                gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            ) for _ in range(self.num_permutation)],
            dtype=np.uint64,
        ).T

        # Determine number of Union-Find workers
        if union_find_parallel_num == 'auto':
            try:
                cpu_count = ray.cluster_resources().get('CPU', 1) # Default to 1 if CPU not found
                union_find_parallel_num = max(1, int(cpu_count / 2)) # Ensure at least 1 worker
            except Exception as e:
                 log.warning(f"Could not auto-detect CPUs for union_find_parallel_num: {e}. Defaulting to 1.")
                 union_find_parallel_num = 1
        else:
            union_find_parallel_num = max(1, int(union_find_parallel_num)) # Ensure at least 1

        log.info(f'Using union_find_parallel_num = {union_find_parallel_num}')
        self.union_find_parallel_num = union_find_parallel_num

        # Store other parameters
        self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
        self.num_edge_buffer_task_returns = num_edge_buffer_task_returns
        self.max_pending_filter_tasks = max_pending_filter_tasks
        self.num_filter_task_returns = num_filter_task_returns
        self.merge_batch_size = min(merge_batch_size, union_find_parallel_num) # Cap merge batch size
        self.union_threshold = union_threshold

        # Initialize Union-Find actors and edge buffers
        self.remote_edge_buffers = [
            EdgeBuffer.remote() for _ in range(self.union_find_parallel_num)
        ]
        self.union_find_list = [
            BTSUnionFind.remote(
                self.union_threshold,
                self.union_find_parallel_num,
                i,
                self.remote_edge_buffers, # Pass the list of actor handles
                self.max_pending_edge_buffer_task,
                self.num_edge_buffer_task_returns,
            ) for i in range(self.union_find_parallel_num)
        ]

        # Precompute representation for empty documents
        empty_hash_sig = np.full((self.num_permutation,), MAX_HASH, dtype=np.uint32)
        self.empty_hashes_per_band = {}
        for i, (start, end) in enumerate(self.hash_ranges):
            band_hash_bytes = i.to_bytes(4, 'big') + empty_hash_sig[start:end].tobytes()
            self.empty_hashes_per_band[i] = band_hash_bytes

        self.empty_hash_table_id = int(MAX_HASH % self.union_find_parallel_num)


    def calc_minhash(self, text_list: pa.Array, uid_list: List[int]):
        """
        Calculates minhashes for a batch of texts and sends hash band values
        to the appropriate BTSUnionFind actors.
        """
        pairs_to_send = defaultdict(list) # Group pairs by target actor ID

        for text, uid in zip(text_list, uid_list):
            text_py = text.as_py() # Convert PyArrow scalar to Python string
            tokens = self.tokenization_func(text_py)

            if len(tokens) > 0:
                # Compute MinHash signature
                hv = np.array([sha1_hash32(token) for token in tokens], dtype=np.uint64)
                # Apply permutations using broadcasting
                phv = ((hv[:, None] * self.perm_a[None, :] + self.perm_b[None, :]) % MERSENNE_PRIME).astype(np.uint32)
                # Get the minimum hash value for each permutation
                hash_values = phv.min(axis=0)

                # Generate LSH band hashes and assign to actors
                for i, (start, end) in enumerate(self.hash_ranges):
                    # Construct band hash value (band index + signature slice bytes)
                    band_hash_value = i.to_bytes(4, 'big') + hash_values[start:end].tobytes()
                    # Determine target actor ID based on the first hash in the band
                    target_actor_id = int(hash_values[start] % self.union_find_parallel_num)
                    # Add (band_hash, uid) pair to the list for the target actor
                    pairs_to_send[target_actor_id].append((band_hash_value, uid))
            else:
                # Handle empty documents: send a predefined empty hash for each band
                for i, empty_band_hash in self.empty_hashes_per_band.items():
                     # Assign empty docs consistently to a specific actor
                     target_actor_id = self.empty_hash_table_id
                     pairs_to_send[target_actor_id].append((empty_band_hash, uid))


        # Send pairs to respective actors using ray.wait for flow control
        result_refs = []
        for actor_id, pairs in pairs_to_send.items():
            if len(result_refs) >= self.max_pending_filter_tasks: # Use filter task limits here
                num_returns = min(self.num_filter_task_returns, len(result_refs))
                ready_refs, result_refs = ray.wait(result_refs, num_returns=num_returns)
                # We don't strictly need the results of add_key_value_pairs, just wait for completion
                try:
                    ray.get(ready_refs)
                except Exception as e:
                    log.error(f"Error getting result from add_key_value_pairs: {e}")
                del ready_refs # Free memory
            # Submit task to the target actor
            result_refs.append(
                self.union_find_list[actor_id].add_key_value_pairs.remote(pairs)
            )

        # Wait for remaining tasks to complete
        if result_refs:
             try:
                ray.get(result_refs)
             except Exception as e:
                log.error(f"Error getting final results from add_key_value_pairs: {e}")
        del result_refs # Free memory


    def merge_op_batch(self, object_refs):
        """Helper to wait for and get results from a list of object refs in batches."""
        results = []
        while object_refs:
            num_returns = min(self.merge_batch_size, len(object_refs))
            ready_refs, object_refs = ray.wait(object_refs, num_returns=num_returns)
            try:
                results.extend(ray.get(ready_refs))
            except Exception as e:
                 log.error(f"Error getting results during merge_op_batch: {e}")
            del ready_refs
        return results

    def merge(self):
        """Performs the distributed Union-Find merge process."""
        log.info("Starting edge redistribution...")
        self.merge_op_batch([
            union_find.edge_redistribution.remote()
            for union_find in self.union_find_list
        ])
        log.info("Edge redistribution complete. Starting balanced union-find iterations...")

        iteration = 0
        while True:
            iteration += 1
            log.info(f"Starting balanced union-find iteration {iteration}...")
            # Run balanced_union_find on all actors and check if any changed state
            results = self.merge_op_batch([
                union_find.balanced_union_find.remote()
                for union_find in self.union_find_list
            ])
            changed = any(results)
            log.info(f"Balanced union-find iteration {iteration} complete. Changed: {changed}")

            if not changed:
                break # Converged

            log.info(f"Starting communication step for iteration {iteration}...")
            # Run communication step to exchange updated parent pointers
            self.merge_op_batch([
                union_find.communication.remote()
                for union_find in self.union_find_list
            ])
            log.info(f"Communication step for iteration {iteration} complete.")

        log.info("Union-Find converged. Starting final squeeze step...")
        # Final squeeze step to clean up state
        self.merge_op_batch([
            union_find.squeeze.remote() for union_find in self.union_find_list
        ])
        log.info("Merge process complete.")

    def filter_with_union_find(self, samples: pa.Table) -> pa.Table:
        """Filters a batch of samples, keeping only non-duplicates based on the UF state."""
        query_dict = defaultdict(list)
        # Group queries (uid, original_index) by the target actor ID
        for idx, uid_scalar in enumerate(samples["uid"]):
            uid = uid_scalar.as_py() # Convert PyArrow scalar to Python int
            hash_id = uid // BATCH_SIZE % self.union_find_parallel_num
            query_dict[hash_id].append((uid, idx))

        # Mask to mark duplicates (False = duplicate, True = keep)
        mask = np.ones(len(samples), dtype=bool)

        result_refs = []
        # Query each actor for duplicate indices within its assigned UIDs
        for hash_id, query in query_dict.items():
            if len(result_refs) >= self.max_pending_filter_tasks:
                num_returns = min(self.num_filter_task_returns, len(result_refs))
                ready_refs, result_refs = ray.wait(result_refs, num_returns=num_returns)
                try:
                    results = ray.get(ready_refs)
                    # Mark duplicates based on indices returned by actors
                    for duplicate_indices in results:
                        if duplicate_indices: # Check if the list is not empty
                             mask[duplicate_indices] = False
                except Exception as e:
                    log.error(f"Error getting results from dup_idx: {e}")
                del ready_refs # Free memory

            result_refs.append(
                self.union_find_list[hash_id].dup_idx.remote(query)
            )

        # Process remaining results
        if result_refs:
            try:
                results = ray.get(result_refs)
                for duplicate_indices in results:
                     if duplicate_indices:
                         mask[duplicate_indices] = False
            except Exception as e:
                 log.error(f"Error getting final results from dup_idx: {e}")
        del result_refs, query_dict # Free memory

        # Apply the mask to filter the table, keeping only desired columns
        columns_to_keep = [name for name in samples.column_names if name != "uid"]
        return samples.select(columns_to_keep).filter(pa.array(mask))

    def run(self, dataset: ray.data.Dataset, **kwargs):
        """
        Runs the full deduplication pipeline on a Ray Dataset.

        :param dataset: Input Ray Dataset with a 'text' column (or as specified by text_key).
        :return: Ray Dataset containing only unique records.
        """
        log.info("Starting Ray BTS Minhash Deduplication pipeline...")
        start_time_run = time.time()

        # Initialize unique ID generator actor
        id_generator = IdGenerator.remote()

        # --- Phase 1: Calculate Minhashes and distribute to Union-Find actors ---
        log.info("Phase 1: Calculating Minhashes and distributing pairs...")
        start_time_hash = time.time()

        # Define the mapping function with UID generation
        def minhash_with_uid(table: pa.Table) -> pa.Table:
            num_rows = len(table)
            # Get a block of unique IDs
            min_id, max_id = ray.get(id_generator.get_next_id.remote(num_rows))
            uid_list = list(range(min_id, max_id)) # Create list directly

            # Calculate minhashes for the batch and send pairs to actors
            self.calc_minhash(table[self.text_key], uid_list)

            # Add the 'uid' column to the table
            # Ensure uid_list length matches table length
            if len(uid_list) != num_rows:
                 raise ValueError(f"Length mismatch: Got {len(uid_list)} UIDs for {num_rows} rows.")
            # Create PyArrow array for the new column
            uid_array = pa.array(uid_list, type=pa.int64()) # Use int64 for safety
            # Append column
            new_table = table.append_column("uid", uid_array)
            return new_table

        # Apply the mapping function using map_batches
        # Use configured hashing_batch_size
        dataset_with_uid = dataset.map_batches(
            minhash_with_uid,
            batch_format='pyarrow',
            batch_size=self.hashing_batch_size, # Control batch size here
            zero_copy_batch=True, # Enable zero-copy if possible
            # num_cpus=1, # Can adjust based on resource needs per task
        ).materialize() # Materialize to ensure hashing is complete before merge

        end_time_hash = time.time()
        log.info(f"Phase 1 (MinHashing) complete. Time: {end_time_hash - start_time_hash:.2f} seconds")

        # --- Phase 2: Run the distributed Union-Find merge process ---
        log.info("Phase 2: Running distributed Union-Find merge...")
        start_time_merge = time.time()
        self.merge()
        end_time_merge = time.time()
        log.info(f"Phase 2 (Merge) complete. Time: {end_time_merge - start_time_merge:.2f} seconds")

        # --- Phase 3: Filter the dataset based on Union-Find results ---
        log.info("Phase 3: Filtering dataset...")
        start_time_filter = time.time()
        # Apply the filtering function using map_batches
        # Can use a different batch size for filtering if needed
        result_dataset = dataset_with_uid.map_batches(
            self.filter_with_union_find,
            batch_format='pyarrow',
            # batch_size=self.batch_size, # Optional: specify filter batch size
            zero_copy_batch=True,
        )
        end_time_filter = time.time()
        log.info(f"Phase 3 (Filtering) complete. Time: {end_time_filter - start_time_filter:.2f} seconds")

        end_time_run = time.time()
        log.info(f"Deduplication pipeline finished. Total time: {end_time_run - start_time_run:.2f} seconds")

        # Materialize the final result before returning
        return result_dataset.materialize()


# --- Workflow Integration Functions ---

@ray.remote(num_returns=2) # Make dedup runnable as a remote task
def dedup_remote_task(ds: ray.data.Dataset, cfg: object):
    """
    Remote wrapper for running deduplication on a dataset partition (e.g., within a cluster).
    Uses configuration passed via the `cfg` object which should contain `args`.
    """
    log.info(f"Running remote deduplication task for cluster...")
    # Assumes cfg object has an 'args' attribute similar to the main script's args
    if not hasattr(cfg, 'args'):
        log.error("Configuration object 'cfg' lacks 'args' attribute needed for deduplication params.")
        # Return empty dataset and zero duplicates on error
        return ds.limit(0).materialize(), 0

    original_count = ds.count()
    log.info(f"Cluster deduplication: starting with {original_count} records")
    start_time = time.time()

    try:
        # Instantiate deduplicator using params from cfg.args
        deduplicator = RayBTSMinhashDeduplicator(
            text_key=cfg.args.column,
            ngram_size=cfg.args.ngram_size,
            min_ngram_size=cfg.args.min_ngram_size,
            num_permutations=cfg.args.num_perm,
            jaccard_threshold=cfg.args.threshold,
            # Use smaller scale params for per-cluster dedup? Or same as global?
            # Using global params for now, adjust if needed.
            union_find_parallel_num=cfg.args.get('dedup_parallel_num', 10), # Example: allow overriding parallel num via args
            union_threshold=cfg.args.get('dedup_union_threshold', 256),
            # Pass other relevant params if needed
        )
        deduplicated_dataset = deduplicator.run(ds) # No need to materialize inside run now

        unique_count = deduplicated_dataset.count()
        duplicate_count = original_count - unique_count
        log.info(f"Cluster deduplication: removed {duplicate_count} duplicates, remaining: {unique_count}")
        total_time = time.time() - start_time
        log.info(f"Cluster deduplication task finished in {total_time:.2f}s")

        # Materialize the final dataset for this task before returning
        return deduplicated_dataset.materialize(), duplicate_count

    except Exception as e:
        log.error(f"Error during remote deduplication task: {e}", exc_info=True)
        # Return original dataset (or empty) and zero duplicates on error
        return ds.limit(0).materialize(), 0 # Return empty on error to avoid propagating bad data

def run_nd_step_for_workflow(ray_df: ray.data.Dataset, args: object) -> Tuple[ray.data.Dataset, int, float]:
    """
    Runs the Near-Duplicate Detection (ND) step for a workflow.

    :param ray_df: Input Ray Dataset.
    :param args: Command-line arguments object containing ND parameters.
    :return: Tuple containing:
        - deduplicated_dataset: Ray Dataset after ND.
        - duplicate_count: Number of duplicates removed.
        - execution_time: Time taken for the ND step in seconds.
    """
    log.info(f"Starting ND step with args: {vars(args)}") # Log args using vars()

    original_count = ray_df.count()
    log.info(f"Original record count for ND: {original_count}")

    start_time = time.time()

    try:
        # Instantiate the deduplicator with parameters from args
        deduplicator = RayBTSMinhashDeduplicator(
            text_key=args.column,
            ngram_size=args.ngram_size,
            min_ngram_size=args.min_ngram_size,
            num_permutations=args.num_perm,
            jaccard_threshold=args.threshold,
            # Consider making these configurable via args as well
            union_find_parallel_num='auto', # Or a specific number based on cluster size/args
            union_threshold=256,
            # Add other parameters if needed, fetched from args
             max_pending_edge_buffer_task=args.max_pending_edge_buffer_task if hasattr(args, 'max_pending_edge_buffer_task') else 20,
             num_edge_buffer_task_returns=args.num_edge_buffer_task_returns if hasattr(args, 'num_edge_buffer_task_returns') else 10,
             max_pending_filter_tasks=args.max_pending_filter_tasks if hasattr(args, 'max_pending_filter_tasks') else 20,
             num_filter_task_returns=args.num_filter_task_returns if hasattr(args, 'num_filter_task_returns') else 10,
             merge_batch_size=args.merge_batch_size if hasattr(args, 'merge_batch_size') else 1000,
             hashing_batch_size=args.hashing_batch_size if hasattr(args, 'hashing_batch_size') else 10000,
        )

        # Run the deduplication pipeline
        deduplicated_dataset = deduplicator.run(ray_df) # Materialization happens inside run()

        # Calculate results
        execution_time = time.time() - start_time
        unique_count = deduplicated_dataset.count()
        duplicate_count = original_count - unique_count

        log.info(f"ND step finished. Time taken: {execution_time:.2f} seconds")
        log.info(f"ND step removed {duplicate_count} duplicates. Remaining records: {unique_count}")

        # Return the results
        return deduplicated_dataset, duplicate_count, execution_time

    except Exception as e:
        log.error(f"Error during ND step: {e}", exc_info=True)
        # Decide error handling: re-raise, return original dataset, or empty dataset?
        # Returning original dataset might be misleading. Returning empty.
        return ray_df.limit(0).materialize(), 0, time.time() - start_time # Return 0 duplicates and elapsed time

# --- Main Guard for Testing (Optional) ---
# Original main() function was for testing, removed as it's not part of the core library code.
# if __name__ == "__main__":
#     # Add test code here if needed, e.g.:
#     # ray.init()
#     # Create dummy data
#     # Instantiate deduplicator
#     # Run deduplicator.run()
#     # Print results
#     # ray.shutdown()
#     pass
# --- END FILE ---
```

```python
# database_project/src/ray_tfidf_vec.py
# --- BEGIN FILE ---
import os
import time
import ray
import ray.data
import pandas as pd
import numpy as np
import pickle
import json # Added for Change 7
from functools import partial
import time
import os
from typing import List, Dict, Tuple, Any, Iterator, Optional, Set # Added Iterator, Optional, Set
from sklearn.pipeline import Pipeline
import logging
import math
from functools import reduce
# Import scikit-learn components
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
import socket
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Filter specific warnings from sklearn
import warnings
warnings.filterwarnings('ignore', message="Your stop_words may be inconsistent with your preprocessing.*", category=UserWarning)

# Conditional imports for torch and jax
try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not found. Some functionalities like torch_pairwise_distance and balanced KMeans might be unavailable.")
    TORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from flax.jax_utils import pad_shard_unpad
    JAX_AVAILABLE = True
except ImportError:
    logger.warning("JAX or Flax not found. JAX acceleration for KMeans will be disabled.")
    JAX_AVAILABLE = False

# Import Minhash deduplication function (updated path)
from ray_minhash import dedup_remote_task # Renamed from dedup, now a remote task

# Import config dict type
try:
    from ml_collections import config_dict
except ImportError:
    logger.warning("ml_collections not found. Using simple dicts for config.")
    config_dict = dict # Fallback to basic dict

# Import YAML loader
try:
    import yaml
except ImportError:
    logger.error("PyYAML not found. Cannot read config files.")
    # Consider exiting or raising error if config loading is critical
    yaml = None

# --- Utility Functions ---

def number_normalizer(tokens):
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)

class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))

def compile_nearest_cluster(kmeans, kmeans_batch_size):
    """Compiles a JAX function for efficient nearest cluster assignment."""
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for compile_nearest_cluster.")

    def _nearest_cluster(data, clusters):
        data = jnp.expand_dims(data, axis=1)
        clusters = jnp.expand_dims(clusters, axis=0)
        dis = (data - clusters) ** 2.0
        dis = jnp.sum(dis, axis=-1)
        dis = jnp.squeeze(dis) # Shape becomes (batch, n_clusters)
        return dis.argmin(axis=1) # Find index of minimum distance

    n_local_devices = jax.local_device_count()
    codebook = np.array(kmeans.cluster_centers_) # Use kmeans.cluster_centers_
    codebook = jax.device_put(codebook)

    # Pmap the function across devices
    nearest_cluster_p = jax.pmap(_nearest_cluster, in_axes=(0, None))

    def nearest_cluster_bound(element):
        # The input 'element' will be sharded across devices by pmap
        return nearest_cluster_p(element, codebook)

    # Use pad_shard_unpad for efficient batch padding on TPUs/GPUs
    nearest_cluster_padded = pad_shard_unpad(nearest_cluster_bound,
                                             static_return=False, static_argnums=())

    def nearest_cluster(batch: np.ndarray) -> List[int]:
        # Ensure batch is numpy array
        if not isinstance(batch, np.ndarray):
            batch = np.array(batch)

        # Pad, shard, run pmapped function, unshard, unpad
        # Calculate min device batch size, ensure it's at least 1
        min_device_batch = max(1, kmeans_batch_size // n_local_devices)
        batch_preds = nearest_cluster_padded(batch, min_device_batch=min_device_batch)

        # Get results back to host and reshape
        batch_preds = jax.device_get(batch_preds).reshape(-1).tolist()
        return batch_preds

    return nearest_cluster

def torch_pairwise_distance(data1, data2):
    """Calculates pairwise distances and returns nearest cluster indices using PyTorch."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for torch_pairwise_distance.")
    # Ensure inputs are tensors on the correct device
    if not isinstance(data1, torch.Tensor): data1 = torch.tensor(data1)
    if not isinstance(data2, torch.Tensor): data2 = torch.tensor(data2)
    device = data2.device # Assume cluster centers define the device
    data1 = data1.to(device)

    # Calculate squared Euclidean distances using broadcasting
    A = torch.unsqueeze(data1, dim=1) # Shape (N, 1, D)
    B = torch.unsqueeze(data2, dim=0) # Shape (1, M, D)
    dis = (A - B) ** 2.0 # Shape (N, M, D)
    dis = torch.sum(dis, dim=-1) # Shape (N, M)
    # Note: squeeze is not needed if M > 1
    # dis = torch.squeeze(dis)
    return torch.argmin(dis, dim=1) # Find index of minimum distance for each row in data1

def np_pairwise_distance(data1, data2):
    """Calculates pairwise distances and returns nearest cluster indices using NumPy."""
    # Ensure inputs are numpy arrays
    if not isinstance(data1, np.ndarray): data1 = np.array(data1)
    if not isinstance(data2, np.ndarray): data2 = np.array(data2)

    A = np.expand_dims(data1, axis=1) # Shape (N, 1, D)
    B = np.expand_dims(data2, axis=0) # Shape (1, M, D)
    dis = (A - B) ** 2.0 # Shape (N, M, D)
    dis = np.sum(dis, axis=-1) # Shape (N, M)
    # dis = np.squeeze(dis) # Not needed if M > 1
    return dis.argmin(axis=1) # Find index of minimum distance

def auction_lap(job_and_worker_to_score, return_token_to_worker=True):
    """Solves balanced linear assignment using Auction Algorithm (PyTorch implementation)."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for auction_lap.")
    # Implementation remains the same as provided in the original file...
    # (Assuming the original implementation is correct)
    eps = (job_and_worker_to_score.max() - job_and_worker_to_score.min()) / 50
    eps.clamp_min_(1e-04)
    if torch.isnan(job_and_worker_to_score).any():
       raise Exception("NaN distance found in auction_lap input")
    worker_and_job_to_score = job_and_worker_to_score.detach().transpose(0,1).contiguous()
    num_workers, num_jobs = worker_and_job_to_score.size()
    if num_workers == 0 or num_jobs == 0: return torch.tensor([], dtype=torch.long) # Handle empty input
    if num_jobs < num_workers:
         logger.warning(f"Auction LAP Warning: Fewer jobs ({num_jobs}) than workers ({num_workers}). Assignment might not be balanced.")
         # Adjust jobs_per_worker logic or handle this case specifically if strict balance is needed.
         # For now, proceeding might lead to some workers getting no jobs.
         jobs_per_worker = 1 # Minimum
    else:
        jobs_per_worker = num_jobs // num_workers


    value = torch.clone(worker_and_job_to_score)
    bids = torch.zeros((num_workers, num_jobs), dtype=worker_and_job_to_score.dtype, device=worker_and_job_to_score.device, requires_grad=False)
    counter = 0
    index = None
    cost = torch.zeros((1,num_jobs,), dtype=worker_and_job_to_score.dtype, device=worker_and_job_to_score.device, requires_grad=False)

    while True:
        # Ensure k for topk is not greater than the number of jobs
        k_topk = min(jobs_per_worker + 1, num_jobs)
        if k_topk <= 0: break # Cannot proceed if k is zero or negative

        top_values, top_index = value.topk(k_topk, dim=1)

        # Check if we have enough values for bidding
        if top_values.size(1) < 2:
            # Cannot calculate bid increments, potentially means jobs_per_worker is 0 or only 1 job exists
            # Assign based on current best value if possible
            if top_values.size(1) == 1:
                 bids.zero_()
                 bids.scatter_(dim=1, index=top_index[:,0:1], src=eps) # Minimal bid
            else: # No jobs, cannot bid
                 break
        else:
            bid_increments = top_values[:,:-1] - top_values[:,-1:] + eps
            assert bid_increments.size(1) == k_topk -1 # Check size consistency
            bids.zero_()
            bids.scatter_(dim=1, index=top_index[:,:-1], src=bid_increments)

        # Retain job logic (consider refining the condition)
        if counter < 100 and index is not None and index < bids.numel():
            bids.view(-1)[index] = eps

        # Handle jobs without bidders after many iterations
        if counter > 1000:
            jobs_without_bidder = (bids == 0).all(0).nonzero(as_tuple=False).squeeze(1)
            if jobs_without_bidder.numel() > 0:
                 bids[0, jobs_without_bidder] = eps # Assign to worker 0 arbitrarily

        # Find jobs with bidders and highest bidder per job
        jobs_with_bidder = (bids > 0).any(0).nonzero(as_tuple=False).squeeze(1)
        if jobs_with_bidder.numel() == 0: break # No bids placed, exit

        high_bids, high_bidders = bids[:, jobs_with_bidder].max(dim=0)

        if high_bidders.size(0) == num_jobs:
            break # All jobs assigned

        # Update costs and values
        cost[:, jobs_with_bidder] += high_bids
        value = worker_and_job_to_score - cost

        # Ensure index calculation is safe
        if high_bidders.numel() > 0 and jobs_with_bidder.numel() > 0:
             index = (high_bidders * num_jobs) + jobs_with_bidder
             if index.max() < value.numel():
                 value.view(-1)[index] = worker_and_job_to_score.view(-1)[index] # Retain original value for winner

        counter += 1
        if counter > 2000: # Add a hard limit to prevent infinite loops
             logger.warning("Auction LAP exceeded max iterations.")
             break

    # Final assignment based on the last high bidders
    if 'high_bidders' not in locals() or high_bidders.numel() == 0:
         logger.warning("Auction LAP finished with no assignments.")
         return torch.tensor([], dtype=torch.long)

    if return_token_to_worker:
        # Ensure output matches number of jobs with bidders
        if jobs_with_bidder.numel() != high_bidders.numel():
             logger.error("Mismatch in job/bidder counts during final assignment.")
             # Handle error case appropriately, maybe return empty or partial
             return torch.tensor([], dtype=torch.long)
        # Need to return assignments for *all* original jobs if possible
        # This part needs clarification on desired output for unbalanced cases
        # Returning assignments only for jobs that received bids:
        final_assignment = torch.full((num_jobs,), -1, dtype=torch.long, device=job_and_worker_to_score.device) # -1 for unassigned
        final_assignment[jobs_with_bidder] = high_bidders
        return final_assignment

    else: # Original logic for balanced return (assumes num_jobs == num_workers * jobs_per_worker)
        _, sorting = torch.sort(high_bidders)
        assignment = jobs_with_bidder[sorting]
        if len(assignment.unique()) != num_jobs:
             logger.warning("Auction LAP result is not a unique assignment for all jobs.")
        return assignment.view(-1)


class KMeans:
    """Custom KMeans implementation supporting online updates and JAX/Torch acceleration."""
    def __init__(self, n_clusters=None, cluster_centers_=None, device=None, balanced=False, use_jax=True):
        self.n_clusters = n_clusters
        self.cluster_centers_ = cluster_centers_ # Use sklearn naming convention
        # Auto-detect device if not provided
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.balanced = balanced
        # Only use JAX if available AND requested
        self.use_jax = use_jax and JAX_AVAILABLE
        if use_jax and not JAX_AVAILABLE:
            logger.warning("JAX requested but not available. Falling back.")
            self.use_jax = False
        # Ensure Torch is available if needed (balanced or not JAX)
        if (balanced or not self.use_jax) and not TORCH_AVAILABLE:
             raise RuntimeError("PyTorch is required for balanced KMeans or non-JAX KMeans.")

        self.jax_pairwise_distance = None # Will be created if use_jax is true

    @classmethod
    def load(cls, path_to_file):
        try:
            with open(path_to_file, 'rb') as f:
                saved_state = pickle.load(f)
            # Recreate instance with saved parameters
            return cls(
                n_clusters=saved_state.get('n_clusters'),
                cluster_centers_=saved_state.get('cluster_centers_'),
                device=saved_state.get('device'),
                balanced=saved_state.get('balanced', False),
                use_jax=saved_state.get('use_jax', True) # Default to True if field missing
            )
        except Exception as e:
            logger.error(f"Failed to load KMeans model from {path_to_file}: {e}")
            raise

    def save(self, path_to_file):
        try:
            # Ensure cluster centers are numpy arrays for pickling if they are tensors
            state_to_save = self.__dict__.copy()
            if TORCH_AVAILABLE and isinstance(self.cluster_centers_, torch.Tensor):
                state_to_save['cluster_centers_'] = self.cluster_centers_.cpu().numpy()
            elif isinstance(self.cluster_centers_, np.ndarray):
                 state_to_save['cluster_centers_'] = self.cluster_centers_ # Already numpy
            else:
                 # Handle case where centers are None or other type
                 state_to_save['cluster_centers_'] = None if self.cluster_centers_ is None else np.array(self.cluster_centers_)


            # Convert device object to string for pickling
            if TORCH_AVAILABLE and isinstance(self.device, torch.device):
                 state_to_save['device'] = str(self.device)


            # Remove non-serializable JAX function if present
            if 'jax_pairwise_distance' in state_to_save:
                del state_to_save['jax_pairwise_distance']


            with open(path_to_file, 'wb+') as f :
                pickle.dump(state_to_save, f)
            logger.info(f"KMeans model saved to {path_to_file}")
        except Exception as e:
            logger.error(f"Failed to save KMeans model to {path_to_file}: {e}")
            raise


    def initialize(self, X: np.ndarray):
        """Initializes cluster centers using random sampling from data X."""
        num_samples = len(X)
        if num_samples < self.n_clusters:
            raise ValueError(f"Number of samples ({num_samples}) is less than number of clusters ({self.n_clusters}). Cannot initialize.")
        indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        initial_state = X[indices]
        return initial_state


    def fit(
            self,
            X: Any, # Can be np.ndarray or torch.Tensor initially
            tol: float = 1e-3,
            tqdm_flag: bool = True,
            iter_limit: int = 0,
            online: bool = False,
            iter_k: Optional[int] = None, # Tracks batch number for online mode
    ):
        """Fits the KMeans model to the data X."""
        if self.use_jax and self.jax_pairwise_distance is None:
             # Create JAX distance function on first fit if needed
             self.jax_pairwise_distance = create_jax_pairwise_distance()


        # Convert input to appropriate type (NumPy for JAX, Tensor for Torch)
        if self.use_jax:
            if not isinstance(X, np.ndarray): X = np.array(X)
        else: # Using PyTorch
            if not isinstance(X, torch.Tensor): X = torch.tensor(X, dtype=torch.float32)
            X = X.to(self.device)

        if tqdm_flag: logger.info(f'Running k-means on {self.device if not self.use_jax else "JAX"}...')

        # Initialization logic
        if not online or (online and iter_k == 0):
            if self.cluster_centers_ is None:
                 logger.info("Initializing KMeans centers...")
                 self.cluster_centers_ = self.initialize(X) # Initialize returns NumPy array
                 # Convert centers to appropriate type after initialization
                 if not self.use_jax: # Convert to Torch tensor if using Torch
                     self.cluster_centers_ = torch.tensor(self.cluster_centers_, dtype=torch.float32).to(self.device)
            # If online and not first iter, centers should already exist in correct format
            elif self.use_jax and TORCH_AVAILABLE and isinstance(self.cluster_centers_, torch.Tensor):
                 self.cluster_centers_ = self.cluster_centers_.cpu().numpy() # Convert existing torch centers to numpy for JAX
            elif not self.use_jax and isinstance(self.cluster_centers_, np.ndarray):
                 self.cluster_centers_ = torch.tensor(self.cluster_centers_, dtype=torch.float32).to(self.device) # Convert numpy to torch


        iteration = 0
        tqdm_meter = tqdm(desc='[running kmeans]', disable=not tqdm_flag, dynamic_ncols=True)

        while True:
            # --- Assignment Step ---
            if self.balanced:
                if not TORCH_AVAILABLE: raise RuntimeError("Balanced KMeans requires PyTorch.")
                # Calculate full distance matrix (negative similarity for auction_lap)
                # Ensure centers are torch tensor for balanced mode
                if isinstance(self.cluster_centers_, np.ndarray):
                     centers_torch = torch.tensor(self.cluster_centers_, dtype=torch.float32).to(self.device)
                else: centers_torch = self.cluster_centers_


                # Calculate distances using torch (needed for auction_lap compatibility)
                A = torch.unsqueeze(X if isinstance(X, torch.Tensor) else torch.tensor(X).to(self.device), dim=1)
                B = torch.unsqueeze(centers_torch, dim=0)
                distance_matrix = torch.sum((A - B) ** 2.0, dim=-1) # Shape (N, M)
                cluster_assignments = auction_lap(-distance_matrix) # Use negative distance

            else: # Not balanced
                if self.use_jax:
                    # Ensure centers are numpy for JAX distance function
                    centers_np = self.cluster_centers_ if isinstance(self.cluster_centers_, np.ndarray) else self.cluster_centers_.cpu().numpy()
                    # JAX function returns indices directly
                    indices = self.jax_pairwise_distance(X, centers_np)
                    cluster_assignments = np.array(indices) # Ensure numpy array
                else: # Use PyTorch
                    # Ensure centers are torch tensor
                    centers_torch = self.cluster_centers_ if isinstance(self.cluster_centers_, torch.Tensor) else torch.tensor(self.cluster_centers_).to(self.device)
                    # PyTorch function returns indices directly
                    indices = torch_pairwise_distance(X, centers_torch)
                    cluster_assignments = indices # Already a tensor

            # --- Update Step ---
            # Use NumPy for updates if using JAX, Torch otherwise
            backend = np if self.use_jax else torch
            initial_state_pre = backend.array(self.cluster_centers_) # Store previous state in correct format

            new_centers = []
            for index in range(self.n_clusters):
                 # Find points assigned to this cluster
                 if backend == np:
                     selected_indices = backend.where(cluster_assignments == index)[0]
                 else: # Torch
                     selected_indices = backend.nonzero(cluster_assignments == index).squeeze()


                 # Handle empty clusters
                 if selected_indices.shape[0] == 0:
                     logger.warning(f"Cluster {index} became empty. Re-initializing with random point.")
                     # Re-initialize with a random point from X
                     random_idx = np.random.randint(len(X))
                     new_center = X[random_idx]
                 else:
                     # Select assigned points and calculate mean
                     if backend == np:
                         selected = X[selected_indices]
                         new_center = selected.mean(axis=0)
                     else: # Torch
                         # Ensure index_select works correctly for 0-dim tensor (single point)
                         if selected_indices.dim() == 0: selected_indices = selected_indices.unsqueeze(0)
                         selected = backend.index_select(X, 0, selected_indices)
                         new_center = selected.mean(dim=0)
                 new_centers.append(new_center)

            # Update cluster centers (convert list back to array/tensor)
            if backend == np:
                 self.cluster_centers_ = backend.stack(new_centers)
            else: # Torch
                 self.cluster_centers_ = backend.stack(new_centers)


            # --- Check Convergence ---
            # Calculate shift using appropriate backend norm
            if backend == np:
                 center_shift = backend.sum(backend.sqrt(backend.sum((self.cluster_centers_ - initial_state_pre) ** 2, axis=1)))
            else: # Torch
                 center_shift = backend.sum(backend.sqrt(backend.sum((self.cluster_centers_ - initial_state_pre) ** 2, dim=1)))


            iteration += 1
            shift_squared = center_shift.item() ** 2 # Use .item() for scalar tensor

            if tqdm_flag:
                tqdm_meter.set_postfix(
                    iteration=f'{iteration}',
                    center_shift_sq=f'{shift_squared:.6f}',
                    tol=f'{tol:.6f}'
                )
                tqdm_meter.update()

            if shift_squared < tol:
                logger.info(f"KMeans converged after {iteration} iterations.")
                break
            if iter_limit != 0 and iteration >= iter_limit:
                logger.info(f"KMeans reached iteration limit ({iter_limit}).")
                break

        tqdm_meter.close()
        # Return assignments in CPU NumPy format for consistency
        if isinstance(cluster_assignments, torch.Tensor):
             return cluster_assignments.cpu().numpy()
        else:
             return cluster_assignments # Already NumPy array


def get_sklearn_feature_pipeline(tfidf_cfg):
    """Creates the Scikit-learn TF-IDF -> SVD -> Normalizer pipeline."""
    n_components = tfidf_cfg.train.get('n_components', 128) # Default if missing
    random_seed = tfidf_cfg.train.get('random_seed', 42)
    logger.info(f"Creating sklearn pipeline with SVD n_components={n_components}, random_seed={random_seed}")

    # Define stop words including the custom number token
    stop_words = list(ENGLISH_STOP_WORDS.union(["#NUMBER"]))

    # Create the pipeline
    vectorizer = Pipeline([
        ('tfidf', NumberNormalizingVectorizer(stop_words=stop_words)),
        ('svd', TruncatedSVD(n_components=n_components, random_state=random_seed)),
        ('normalizer', Normalizer(copy=False)) # Normalize in-place
    ], verbose=True) # Enable verbosity for debugging
    return vectorizer


# --- Ray Remote Functions and Classes ---

import ray.cloudpickle as cloudpickle

# Serialization helpers (no changes needed)
def serialize_objectref_dict(objectref_dict):
    return {k: cloudpickle.dumps(v) for k, v in objectref_dict.items()}

def deserialize_objectref_dict(objectref_dict):
    return {k: cloudpickle.loads(v) for k, v in objectref_dict.items()}

def create_jax_pairwise_distance():
    """Creates the JAX pairwise distance function using pmap."""
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX required for create_jax_pairwise_distance")

    def reshape_for_jax(data1, data2):
        batch_size = data1.shape[0]
        n_clusters = data2.shape[0]
        n_devices = jax.local_device_count()
        # Ensure batch size is divisible by number of devices for simple sharding
        if batch_size % n_devices != 0:
             # Pad data1 if necessary - simple padding for now
             pad_width = n_devices - (batch_size % n_devices)
             data1 = np.pad(data1, ((0, pad_width), (0, 0)), mode='constant')
             padded_batch_size = data1.shape[0]
             logger.debug(f"Padded JAX input from {batch_size} to {padded_batch_size}")
        else:
             padded_batch_size = batch_size

        data1 = data1.reshape([n_devices, padded_batch_size // n_devices, -1])
        return data1, data2, batch_size, n_clusters # Return original batch_size

    def _jax_pairwise_distance(data1_shard, data2_all):
        A = jnp.expand_dims(data1_shard, axis=1)
        B = jnp.expand_dims(data2_all, axis=0)
        dis = (A - B) ** 2.0
        dis = jnp.sum(dis, axis=-1)
        # dis = jnp.squeeze(dis) # Not needed if n_clusters > 1
        return dis # Return full distance matrix shard

    dist_func = jax.pmap(_jax_pairwise_distance, in_axes=(0, None))

    def jax_pairwise_distance(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        data1_reshaped, data2_put, original_batch_size, n_clusters = reshape_for_jax(data1, data2)
        # Put data2 on all devices once
        data2_put = jax.device_put(data2)
        dis = dist_func(data1_reshaped, data2_put) # Pass sharded data1 and replicated data2
        # Get result back, reshape, and truncate padding
        dis_np = jax.device_get(dis).reshape(-1, n_clusters)
        return dis_np[:original_batch_size] # Return only results for original data

    return jax_pairwise_distance


def fit_kmeans(embeddings: Any, kmeans_cfg: object, **kwargs) -> KMeans:
    """Fits the custom KMeans model, handling data loading and online updates."""
    logger.info(f"Fitting KMeans with config: {kmeans_cfg}")
    # Determine if JAX should be used based on KMeans instance setting
    use_jax = kwargs.get('use_jax', True) and JAX_AVAILABLE
    device = kwargs.get('device', None)

    # Create KMeans instance
    kmeans = KMeans(n_clusters=kmeans_cfg.n_clusters, balanced=kmeans_cfg.get('balanced', False), use_jax=use_jax, device=device)


    # Handle input type - assume it's NumPy array or convert
    if not isinstance(embeddings, np.ndarray):
         try:
             embeddings_np = np.array(embeddings)
         except Exception as e:
             logger.error(f"Could not convert embeddings to NumPy array: {e}")
             raise
    else:
         embeddings_np = embeddings


    # Use DataLoader for batching if PyTorch is available and needed (e.g., for Torch backend)
    # For JAX or simple NumPy updates, direct iteration might be simpler if memory allows
    if TORCH_AVAILABLE and (kmeans.balanced or not kmeans.use_jax):
         # Use DataLoader for PyTorch backend
         batch_size = kmeans_cfg.train.get('batch_size', 2048)
         logger.info(f"Using PyTorch DataLoader with batch size {batch_size}")
         embeddings_loader = DataLoader(embeddings_np, batch_size=batch_size, shuffle=True, drop_last=True) # Shuffle batches
         iterable_batches = embeddings_loader
    else:
         # For JAX/NumPy, just use the array directly if fitting in one go,
         # or handle batching manually if needed (e.g., for very large samples)
         # Assuming online fit iterates through external batches passed via loop
         iterable_batches = [embeddings_np] # Treat as single batch if not using DataLoader logic here


    with tqdm(dynamic_ncols=True, desc="fit_kmeans batches") as pbar:
        for i, batch in enumerate(iterable_batches):
            if TORCH_AVAILABLE and isinstance(batch, list): # DataLoader might yield list of tensors
                 batch = batch[0] # Assuming single tensor per batch item
            pbar.set_postfix({"batch": i})
            pbar.update(1) # Update per batch
            # Fit the batch (online=True implies multiple calls if iterable_batches has > 1 item)
            kmeans.fit(
                 batch,
                 iter_limit=kmeans_cfg.train.get('iter_limit', 5),
                 online=(len(iterable_batches) > 1), # Set online=True if looping through batches
                 iter_k=i, # Pass batch index
                 tqdm_flag=False # Disable inner tqdm loop
             )
    logger.info("KMeans fitting complete.")
    return kmeans


@ray.remote(num_cpus=1) # Keep low CPU request for coordinator task
def _fit_models_remote(
    cfg: object,
    ds: ray.data.Dataset, # Pass dataset handle directly
) -> Tuple[object, object, float]: # Return vectorizer, kmeans, train_time
    """Fits TF-IDF/SVD and KMeans on a sample, runs as a Ray remote task."""
    stage_label = cfg.cluster_col_name # Use cluster_col_name as label
    logger.info(f"[{stage_label}] Starting model fitting...")
    fit_start_time = time.time()

    # --- Sampling ---
    max_docs = cfg.get('max_docs', 5000) # Get max_docs from config
    logger.info(f"[{stage_label}] Sampling up to {max_docs} documents...")
    # Use sample if dataset is large, otherwise take all
    # This requires knowing the dataset size, which can be expensive.
    # Alternative: directly use limit.
    sample_ds = ds.limit(max_docs)
    logger.info(f"[{stage_label}] Collecting sample...")
    try:
        # Ensure sample fits in memory of the actor
        sample_df = sample_ds.to_pandas()
    except Exception as e:
         logger.error(f"[{stage_label}] Failed to collect sample to pandas (maybe too large?): {e}")
         raise # Re-raise exception

    sample_size = len(sample_df)
    logger.info(f"[{stage_label}] Sample size: {sample_size}")
    if sample_size == 0:
        raise ValueError(f"[{stage_label}] Sample size is 0. Cannot fit models.")

    texts = sample_df["text"].tolist()
    del sample_df # Free memory

    # --- TF-IDF/SVD Fitting ---
    logger.info(f"[{stage_label}] Fitting vectorizer on {len(texts)} samples...")
    vectorizer = get_sklearn_feature_pipeline(cfg.tfidf)
    try:
        # Embeddings are needed for K-Means fitting
        embeddings = vectorizer.fit_transform(texts)
    except Exception as e:
        logger.error(f"[{stage_label}] Error during vectorizer fitting: {e}")
        raise
    logger.info(f"[{stage_label}] Vectorizer fitting done. Embedding shape: {embeddings.shape}")
    del texts # Free memory

    # --- K-Means Fitting ---
    n_clusters = cfg.kmeans.get('n_clusters', 10) # Get n_clusters from config
    logger.info(f"[{stage_label}] Fitting K-means with {n_clusters} clusters...")
    try:
        kmeans = fit_kmeans(embeddings, cfg.kmeans, use_jax=True) # Pass use_jax=True (or config based)
    except Exception as e:
        logger.error(f"[{stage_label}] Error during KMeans fitting: {e}")
        raise
    logger.info(f"[{stage_label}] K-means fitting done.")
    del embeddings # Free memory

    fit_end_time = time.time()
    train_time = fit_end_time - fit_start_time
    logger.info(f"[{stage_label}] Total model fitting time: {train_time:.2f}s")

    # Return models and the calculated training time
    return vectorizer, kmeans, train_time

# --- Model Fitting Orchestration ---
# Use a separate function to launch the remote task with resource requests
def fit_models_remote(cfg, ds) -> Tuple[ray.ObjectRef, ray.ObjectRef, ray.ObjectRef]:
     """Launches the _fit_models_remote task with specified resources."""
     logger.info(f"Launching remote model fitting task with {cfg.tfidf.train.num_cpus} CPUs and TPU resource.")
     # Schedule the remote fitting task
     # Returns separate ObjectRefs for vectorizer, kmeans, and train_time
     vectorizer_ref, kmeans_ref, train_time_ref = _fit_models_remote.options(
         num_cpus=cfg.tfidf.train.num_cpus,
         resources={"TPU-v4-8-head": 1}, # Request TPU resource
     ).remote(cfg, ds)
     return vectorizer_ref, kmeans_ref, train_time_ref


# --- Inference Classes ---

class TFIDFInferenceModel:
    """Applies the fitted TF-IDF/SVD pipeline."""
    def __init__(self, vectorizer_ref: ray.ObjectRef):
        # Get the vectorizer model from the ObjectRef
        logger.info("TFIDFInferenceModel: Initializing...")
        try:
             # vectorizer, _, _ = ray.get(vectorizer_ref) # Original assumed combined ref
             # Assuming vectorizer_ref ONLY contains the vectorizer now:
             self.vectorizer = ray.get(vectorizer_ref)
             logger.info("TFIDFInferenceModel: Vectorizer loaded.")
        except Exception as e:
             logger.error(f"TFIDFInferenceModel: Failed to get vectorizer from ref: {e}")
             self.vectorizer = None # Mark as failed

    def __call__(self, batch: Dict[str, list]) -> Dict[str, list]:
        if self.vectorizer is None:
             logger.error("TFIDFInferenceModel: Vectorizer not loaded, cannot process batch.")
             # Return empty batch or raise error? Returning empty dict.
             return {'text': [], 'embeddings': []} # Adjust schema as needed

        try:
            texts = batch["text"]
            embeddings = self.vectorizer.transform(texts)
            batch["embeddings"] = list(embeddings) # Store as list of numpy arrays
            return batch
        except Exception as e:
            logger.error(f"TFIDFInferenceModel: Error processing batch: {e}", exc_info=True)
            # Return empty or partially processed batch? Returning empty.
            return {'text': [], 'embeddings': []}


class KMeansInferenceModel:
    """Applies the fitted KMeans model using compiled JAX function."""
    def __init__(self, kmeans_ref: ray.ObjectRef, cfg: object):
        logger.info("KMeansInferenceModel: Initializing...")
        try:
            # _, self.kmeans, _ = ray.get(kmeans_ref) # Original assumed combined ref
            # Assuming kmeans_ref ONLY contains the kmeans model now:
            self.kmeans = ray.get(kmeans_ref)

            # Ensure kmeans object is valid and has cluster centers
            if not hasattr(self.kmeans, 'cluster_centers_') or self.kmeans.cluster_centers_ is None:
                 raise ValueError("Loaded KMeans object has no cluster_centers_.")

            # Compile the JAX nearest cluster function
            # Check if JAX is actually used by the loaded KMeans instance
            if self.kmeans.use_jax and JAX_AVAILABLE:
                logger.info("KMeansInferenceModel: Compiling JAX tagging function...")
                self.tagging_func = compile_nearest_cluster(self.kmeans, kmeans_batch_size=cfg.kmeans.inference.batch_size)
                logger.info("KMeansInferenceModel: JAX tagging function compiled.")
            elif TORCH_AVAILABLE: # Fallback to PyTorch if JAX not used/available but Torch is
                 logger.info("KMeansInferenceModel: Using PyTorch for tagging.")
                 self.tagging_func = partial(torch_pairwise_distance, data2=torch.tensor(self.kmeans.cluster_centers_).to(self.kmeans.device))
            else: # Fallback to NumPy if neither JAX nor Torch available/used
                 logger.info("KMeansInferenceModel: Using NumPy for tagging.")
                 self.tagging_func = partial(np_pairwise_distance, data2=np.array(self.kmeans.cluster_centers_))

            self.cluster_col_name = cfg.cluster_col_name
            logger.info(f"KMeansInferenceModel: Ready to tag column '{self.cluster_col_name}'.")

        except Exception as e:
            logger.error(f"KMeansInferenceModel: Failed initialization: {e}", exc_info=True)
            self.kmeans = None
            self.tagging_func = None

    def __call__(self, batch: Dict[str, list]) -> Dict[str, list]:
        if self.kmeans is None or self.tagging_func is None:
            logger.error("KMeansInferenceModel: Not initialized correctly, cannot process batch.")
            # Return empty batch without cluster column
            batch.pop("embeddings", None) # Remove embeddings if present
            return batch

        try:
            # Extract embeddings - ensure they are numpy arrays
            embeddings = np.array([emb for emb in batch["embeddings"]])
            if embeddings.size == 0: # Handle empty batch after potential TFIDF errors
                 logger.warning("KMeansInferenceModel: Received empty embeddings batch.")
                 batch[self.cluster_col_name] = []
            else:
                 # Apply the compiled tagging function
                 cluster_labels = self.tagging_func(embeddings)
                 # Add cluster labels as integer column
                 batch[self.cluster_col_name] = np.array(cluster_labels, dtype=np.int32)

            # Remove embeddings column
            batch.pop("embeddings", None)
            return batch
        except Exception as e:
            logger.error(f"KMeansInferenceModel: Error processing batch: {e}", exc_info=True)
            batch.pop("embeddings", None)
            # Add empty cluster column or skip? Adding empty for schema consistency.
            batch[self.cluster_col_name] = []
            return batch


# --- Core Fit/Predict Logic ---

def fit_predict(ds: ray.data.Dataset, cfg: object) -> Tuple[ray.data.Dataset, float, float]:
    """Fits models on sample, predicts on full dataset, returns tagged dataset and timings."""
    logger.info(f"--- {cfg.pretty_name} Fit/Predict Starting ---")
    stage_start_time = time.time()

    # 1. Fit models remotely, get ObjectRefs for models and train time
    logger.info(f"[{cfg.pretty_name}] Launching remote model fitting...")
    vectorizer_ref, kmeans_ref, train_time_ref = fit_models_remote(cfg, ds)
    # We retrieve the train_time later, after inference starts

    # 2. Start Inference Pipeline (TF-IDF -> KMeans)
    logger.info(f"[{cfg.pretty_name}] Starting inference pipeline...")
    inf_start_time = time.time() # Start timing inference part

    # Map Batches for TF-IDF Inference
    emb_tagged_ds = ds.map_batches(
        TFIDFInferenceModel,
        batch_format="pyarrow", # Use PyArrow for potentially better memory efficiency
        batch_size=cfg.tfidf.inference.get('batch_size', 1024), # Default batch size
        num_cpus=cfg.tfidf.inference.get('num_cpus', 1), # Default CPUs per actor
        concurrency=cfg.tfidf.inference.get('concurrency', 400), # Control parallelism
        fn_constructor_kwargs={"vectorizer_ref": vectorizer_ref},
    )

    # Map Batches for KMeans Inference
    # Ensure compute resources match hardware used (CPU/TPU)
    kmeans_compute = {"num_cpus": cfg.kmeans.inference.get('num_cpus', 1)}
    if cfg.kmeans.inference.get('use_tpu', True): # Check config if TPU should be requested
        kmeans_compute["resources"] = {"TPU-v4-8-head": 1} # Request TPU resource

    tagged_ds = emb_tagged_ds.map_batches(
        KMeansInferenceModel,
        batch_format="pyarrow",
        batch_size=cfg.kmeans.inference.get('batch_size', 8192),
        compute=ray.data.ActorPoolStrategy(min_size=1, max_size=int(cfg.kmeans.inference.get('concurrency', 10))), # Use ActorPoolStrategy
        # num_cpus=cfg.kmeans.inference.num_cpus, # Defined in compute
        concurrency=int(cfg.kmeans.inference.get('concurrency', 10)), # Concurrency for task submission
        fn_constructor_kwargs={"kmeans_ref": kmeans_ref, "cfg": cfg},
        **kmeans_compute # Pass resource requests here
    )

    # 3. Retrieve Training Time (while inference is running)
    try:
        stage_train_time = ray.get(train_time_ref)
        logger.info(f"[{cfg.pretty_name}] Retrieved training time: {stage_train_time:.2f}s")
    except Exception as e:
         logger.error(f"[{cfg.pretty_name}] Failed to get training time: {e}. Setting to 0.")
         stage_train_time = 0.0

    # 4. Materialize final tagged dataset to ensure inference completes
    logger.info(f"[{cfg.pretty_name}] Materializing final tagged dataset...")
    final_tagged_ds = tagged_ds.materialize()
    inf_end_time = time.time()
    inference_time = inf_end_time - inf_start_time
    logger.info(f"[{cfg.pretty_name}] Inference pipeline complete. Time: {inference_time:.2f}s")

    # Clean up object refs if needed (Ray might do this automatically)
    del vectorizer_ref, kmeans_ref, train_time_ref, emb_tagged_ds, tagged_ds

    stage_end_time = time.time()
    logger.info(f"--- {cfg.pretty_name} Fit/Predict Complete. Total Stage Time: {stage_end_time - stage_start_time:.2f}s ---")

    return final_tagged_ds, stage_train_time, inference_time


# --- Stage Wrappers ---

# Stage 1: Global Clustering
def stage1(ds: ray.data.Dataset, cfg: object) -> Tuple[ray.data.Dataset, int, float, float]:
    """Runs the first stage of clustering."""
    logger.info(f"===== Starting {cfg.pretty_name} =====")
    stage_start_time = time.time()
    # Fit models and predict cluster assignments for stage 1
    tagged_ds, stage1_train_time, stage1_inference_time = fit_predict(ds, cfg)
    stage_end_time = time.time()
    logger.info(f"===== {cfg.pretty_name} Complete. Time: {stage_end_time - stage_start_time:.2f}s =====")
    # Stage 1 does not perform deduplication, so duplicate count is 0
    return tagged_ds, 0, stage1_train_time, stage1_inference_time


# Stage 2: Per-Cluster Processing (Clustering + Optional Dedup)
# Define fit_predict_remote task wrapper
@ray.remote
def fit_predict_remote(ds: ray.data.Dataset, cfg: object) -> Tuple[ray.data.Dataset, float, float]:
    """Remote task wrapper for fit_predict, returning dataset and timings."""
    # Materialize input dataset within the remote task if needed
    # ds = ds.materialize() # Consider if materialization is needed here
    clustered_ds, train_time, inference_time = fit_predict(ds, cfg)
    # Materialize the result before returning the handle
    return clustered_ds.materialize(), train_time, inference_time


def stage2(ds: ray.data.Dataset, cfg: object) -> Tuple[ray.data.Dataset, int, float, float]:
    """Runs the second stage: clustering within Stage 1 clusters and optional deduplication."""
    logger.info(f"===== Starting {cfg.pretty_name} =====")
    stage_start_time = time.time()

    if not cfg.partition_cols:
        raise ValueError("Stage 2 requires partition_cols (Stage 1 cluster column) in config.")
    if not cfg.cluster_spec:
        raise ValueError("Stage 2 requires cluster_spec (Stage 1 cluster count) in config.")

    stage1_cluster_col = cfg.partition_cols[0]
    stage1_num_clusters = cfg.cluster_spec[0]
    logger.info(f"Stage 2 processing based on Stage 1 column '{stage1_cluster_col}' with {stage1_num_clusters} clusters.")

    # List to hold results from each Stage 1 cluster processing
    processed_refs = [] # Stores tuples: (final_ds_ref, dupe_count_ref, train_time_ref, inf_time_ref)

    # --- Launch parallel tasks for each Stage 1 cluster ---
    logger.info("Splitting dataset and launching Stage 2 tasks...")
    # Use split() for potentially better performance than filter() N times
    try:
        split_datasets = ds.split(n=stage1_num_clusters, locality_hints=[str(i) for i in range(stage1_num_clusters)])
        # Note: This split is arbitrary, not by cluster ID. We need filter.
        # Reverting to filter approach:
        # stage1_datasets = [ds.filter(lambda row: row[stage1_cluster_col] == cluster_id)
        #                    for cluster_id in range(stage1_num_clusters)]
        # Optimization: Group by cluster first, then process groups? Less map-reduce like.
        # Sticking with filter for clarity, accepting potential inefficiency.

        for cluster_id in range(stage1_num_clusters):
             logger.debug(f"Launching task for Stage 1 Cluster ID: {cluster_id}")
             # Filter data for the current cluster
             ds_cluster_data = ds.filter(lambda row: row[stage1_cluster_col] == cluster_id)
             # ds_cluster_data = ds_cluster_data.materialize() # Materialize subset? Might be costly.

             # Check if cluster is empty
             # count = ds_cluster_data.count() # count() is expensive
             # if count == 0:
             #      logger.info(f"Skipping empty cluster {cluster_id}")
             #      continue # Skip empty clusters


             # 1. Run Stage 2 Clustering (fit_predict) remotely
             s2_clustered_ds_ref, train_time_ref, inf_time_ref = fit_predict_remote.remote(ds_cluster_data, cfg)

             # 2. Conditionally run Deduplication remotely
             if cfg.get('should_dedup', False): # Check config flag
                 logger.debug(f"Launching deduplication task for cluster {cluster_id}")
                 # Pass the *result* of clustering to deduplication
                 final_ds_ref, dupe_count_ref = dedup_remote_task.remote(s2_clustered_ds_ref, cfg)
             else:
                 # If not deduping, the final dataset is the clustered one
                 final_ds_ref = s2_clustered_ds_ref
                 # Put 0 into the object store for duplicate count
                 dupe_count_ref = ray.put(0)

             # Store refs for later retrieval
             processed_refs.append((final_ds_ref, dupe_count_ref, train_time_ref, inf_time_ref))

             # Optional delay if needed for resource management (consider removing)
             # time.sleep(5)

    except Exception as e:
         logger.error(f"Error launching stage 2 tasks: {e}", exc_info=True)
         # Return empty dataset and zero counts/times?
         return ray.data.from_items([]), 0, 0.0, 0.0

    # --- Retrieve and Aggregate Results ---
    logger.info(f"Retrieving results for {len(processed_refs)} Stage 2 tasks...")
    if not processed_refs:
        logger.warning("No Stage 2 tasks were processed (maybe all clusters were empty?). Returning empty dataset.")
        return ray.data.from_items([]), 0, 0.0, 0.0

    try:
        # Get all results using ray.get
        final_ds_list = ray.get([ref_pair[0] for ref_pair in processed_refs])
        count_results = ray.get([ref_pair[1] for ref_pair in processed_refs])
        train_times = ray.get([ref_pair[2] for ref_pair in processed_refs])
        inf_times = ray.get([ref_pair[3] for ref_pair in processed_refs])
    except Exception as e:
        logger.error(f"Error retrieving results from Stage 2 tasks: {e}", exc_info=True)
        # Return empty dataset and zero counts/times?
        return ray.data.from_items([]), 0, 0.0, 0.0

    # Aggregate timings and counts
    total_stage2_duplicates = sum(count_results)
    total_stage2_train_time = sum(train_times)
    total_stage2_inference_time = sum(inf_times) # Sum inference times across clusters

    logger.info(f"Total duplicates found in Stage 2 (if enabled): {total_stage2_duplicates}")
    logger.info(f"Aggregated Stage 2 Train Time: {total_stage2_train_time:.2f}s")
    logger.info(f"Aggregated Stage 2 Inference Time: {total_stage2_inference_time:.2f}s")


    # Union the datasets from all processed clusters
    logger.info("Unioning datasets from Stage 2...")
    # Filter out potential None results if a task failed and returned None gracefully
    valid_ds_list = [ds for ds in final_ds_list if ds is not None and ds.count() > 0] # Check count > 0

    if not valid_ds_list:
        logger.warning("No valid datasets returned from Stage 2 processing. Returning empty dataset.")
        return ray.data.from_items([]), 0, total_stage2_train_time, total_stage2_inference_time


    # Use tree reduce for union if many datasets, otherwise simple union
    if len(valid_ds_list) > 50: # Arbitrary threshold
         logger.info(f"Using tree_reduce for unioning {len(valid_ds_list)} datasets...")
         final_ds = ray.data.context.DataContext.get_current().tree_reduce(lambda ds1, ds2: ds1.union(ds2), valid_ds_list)
    elif len(valid_ds_list) > 1:
        final_ds = valid_ds_list[0].union(*valid_ds_list[1:])
    else: # Only one valid dataset
        final_ds = valid_ds_list[0]

    # Optional: Sort the final combined dataset
    try:
        sort_cols = cfg.partition_cols[:2] # Sort by first two cluster levels if available
        if sort_cols:
             logger.info(f"Sorting final dataset by {sort_cols}...")
             final_ds = final_ds.sort(key=sort_cols)
    except Exception as e:
         logger.warning(f"Could not sort final dataset: {e}")


    # Materialize the final result
    logger.info("Materializing final Stage 2 dataset...")
    final_materialized_ds = final_ds.materialize()

    stage_end_time = time.time()
    logger.info(f"===== {cfg.pretty_name} Complete. Time: {stage_end_time - stage_start_time:.2f}s =====")

    # Return final dataset, total duplicates, and aggregated timings
    return final_materialized_ds, total_stage2_duplicates, total_stage2_train_time, total_stage2_inference_time


# --- Workflow Runner ---

def run_cl_step_for_workflow(ds: ray.data.Dataset, cfg: object) -> Tuple[ray.data.Dataset, int, float, float, float, str]:
    """
    Runs the full clustering workflow (potentially multi-stage).

    Returns:
        - final_ds: The final clustered (and optionally deduplicated) Ray Dataset.
        - workflow_duplicate_count: Total duplicates removed during the CL workflow (primarily from stage2 dedup).
        - total_train_time: Aggregated training time across all stages.
        - total_inference_time: Aggregated inference time across all stages.
        - stage2_time: Wall-clock time specifically for the stage2 function execution.
        - cluster_distribution_json: JSON string of the final cluster size distribution.
    """
    logger.info("Starting Clustering Workflow Step...")
    workflow_start_time = time.time()

    # --- Configuration Setup ---
    output_base_path = cfg.get("base_dir", "/tmp/ray_clustering_output") # Use config base_dir or default
    final_output_path = f"{output_base_path}/ray_output_final_clustered" # Specific subfolder
    os.makedirs(final_output_path, exist_ok=True)
    logger.info(f"Clustering output base directory: {final_output_path}")


    # Initial repartitioning
    initial_blocks = cfg.get("num_blocks", 1000)
    logger.info(f"Repartitioning input dataset to {initial_blocks} blocks.")
    ds = ds.repartition(initial_blocks)


    # --- Workflow Variables ---
    workflow_duplicate_count = 0 # Total duplicates removed in this CL step
    total_train_time = 0.0
    total_inference_time = 0.0
    stage2_exec_time = 0.0 # Specific timing for stage2 function call

    # --- Dynamic Stage Execution ---
    partition_cols = [x.get("cluster_col_name", f"cluster_{i}") for i, x in enumerate(cfg.stages_list)]
    cluster_spec = [x.get("kmeans", {}).get("n_clusters", 10) for x in cfg.stages_list] # Default 10 clusters if not specified

    # Prepare base config to be updated per stage
    base_cfg = config_dict.ConfigDict(cfg.base_stage)
    base_cfg.cluster_spec = cluster_spec
    base_cfg.partition_cols = partition_cols
    # Propagate command-line args if available in main cfg
    if hasattr(cfg, 'args'):
        base_cfg.args = cfg.args
    # Propagate should_dedup flag
    base_cfg.should_dedup = cfg.get('should_dedup', False)


    # Define stage functions mapping
    stage_functions = {
        "stage1": stage1,
        "stage2": stage2,
        # Add other stages here if needed
    }


    # Execute stages sequentially
    current_ds = ds
    for stage_config in cfg.stages_list:
        stage_name = stage_config.get("name")
        if stage_name not in stage_functions:
            logger.error(f"Unknown stage name '{stage_name}' in config. Skipping.")
            continue

        func = stage_functions[stage_name]
        stage_cfg = base_cfg.copy_and_resolve_references()
        stage_cfg.update(stage_config) # Merge stage-specific config

        logger.info(f"--- Preparing to run {stage_name} ---")
        logger.debug(f"Stage Config: {stage_cfg}")


        stage_start_exec_time = time.time()
        # Execute the stage function
        try:
            # Stage functions now return: ds, stage_duplicates, stage_train_time, stage_inference_time
            current_ds, stage_duplicates, stage_train_time, stage_inference_time = func(current_ds, stage_cfg)
            logger.info(f"Stage '{stage_name}' completed.")
            logger.info(f"  - Duplicates added: {stage_duplicates}")
            logger.info(f"  - Train time added: {stage_train_time:.2f}s")
            logger.info(f"  - Inference time added: {stage_inference_time:.2f}s")


            # Accumulate results
            workflow_duplicate_count += stage_duplicates
            total_train_time += stage_train_time
            total_inference_time += stage_inference_time


            # Record stage2 execution time separately
            if func == stage2:
                stage2_exec_time = time.time() - stage_start_exec_time
                logger.info(f"  - Stage 2 function execution time: {stage2_exec_time:.2f}s")


        except Exception as e:
            logger.error(f"Error executing stage '{stage_name}': {e}", exc_info=True)
            # Decide how to handle stage failure: stop workflow, return partial results?
            # Returning current state before failure for now.
            logger.warning("Workflow terminated prematurely due to stage failure.")
            # Calculate distribution on potentially partial dataset? Or return None? Returning None.
            return current_ds, workflow_duplicate_count, total_train_time, total_inference_time, stage2_exec_time, None


    # --- Final Processing ---
    final_ds = current_ds.materialize() # Ensure all ops are done
    logger.info("Clustering stages complete.")


    # Final repartitioning (optional)
    final_repartition_blocks = cfg.get("final_repartition", 40)
    if final_repartition_blocks > 0:
         logger.info(f"Repartitioning final dataset to {final_repartition_blocks} blocks.")
         final_ds = final_ds.repartition(final_repartition_blocks)


    # --- Calculate Cluster Distribution (Change 7) ---
    cluster_distribution_json = None
    try:
        if partition_cols and final_ds.count() > 0: # Ensure columns exist and ds not empty
            logger.info(f"Calculating final cluster distribution based on: {partition_cols}")
            distribution_ds = final_ds.groupby(partition_cols).count()
            distribution_list = distribution_ds.take_all() # Fetch results [{col1: val, col2: val, 'count()': count}, ...]
            # Rename 'count()' column for clarity if needed
            for item in distribution_list:
                 if 'count()' in item:
                      item['count'] = item.pop('count()')
            cluster_distribution_json = json.dumps(distribution_list)
            logger.info("Cluster distribution calculated.")
            # logger.debug(f"Distribution JSON: {cluster_distribution_json}") # Potentially very long log
        else:
             logger.warning("Cannot calculate cluster distribution: No partition columns defined or dataset is empty.")
    except Exception as e:
         logger.error(f"Failed to calculate cluster distribution: {e}", exc_info=True)


    # --- Return Results ---
    workflow_end_time = time.time()
    logger.info(f"Clustering Workflow Step finished. Total time: {workflow_end_time - workflow_start_time:.2f}s")

    # Return final dataset, total duplicates, aggregated timings, stage2 time, and distribution JSON
    return final_ds.materialize(), workflow_duplicate_count, total_train_time, total_inference_time, stage2_exec_time, cluster_distribution_json


# --- Helper to read config (if needed standalone) ---
def read_config(path):
    if not yaml:
        raise RuntimeError("PyYAML is required to read config files.")
    try:
        with open(path) as f:
            config_data = yaml.safe_load(f)
            cfg = config_dict.ConfigDict(config_data)
        return cfg
    except Exception as e:
        logger.error(f"Error reading config file {path}: {e}")
        raise

# --- END FILE ---

```

```python
# database_project/src/run_workflows.py
# --- BEGIN FILE ---
import os

# --- Environment Setup (Consider if needed for your execution environment) ---
# os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.10"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.10"

import sys
import time
import logging
import argparse
import glob
import json # Change 4
import threading # Change 5
import psutil # Change 5
import statistics # Change 5
from queue import Queue, Empty # Change 5 (import Empty)

# Add project root to Python path if needed
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import Core Logic Functions ---
from database_project.src.ray_minhash import run_nd_step_for_workflow # Path adjusted for clarity
from database_project.src.ray_tfidf_vec import run_cl_step_for_workflow # Path adjusted

# --- Import DB and Config Handling ---
from database_project.src.db import init_db, get_session, BenchmarkRun # Path adjusted
try:
    from ml_collections import config_dict
    import yaml
    CONFIG_TOOLS_AVAILABLE = True
except ImportError:
    config_dict = dict # Fallback
    yaml = None
    CONFIG_TOOLS_AVAILABLE = False

# --- Import Ray ---
import ray

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def get_total_size_gb(files: List[str]) -> Optional[float]:
    """Calculates the total size of a list of files in GiB."""
    try:
        total_bytes = sum(os.path.getsize(f) for f in files)
        return total_bytes / (1024 * 1024 * 1024)
    except Exception as e:
        logger.warning(f"Could not calculate total size: {e}")
        return None

def read_config(path: str) -> config_dict.ConfigDict:
    """Reads YAML config file into a ConfigDict."""
    if not CONFIG_TOOLS_AVAILABLE or not yaml:
        raise RuntimeError("ml_collections and PyYAML are required to read config files.")
    try:
        with open(path) as f:
            config_data = yaml.safe_load(f)
            cfg = config_dict.ConfigDict(config_data)
        logger.info(f"Successfully read config file: {path}")
        return cfg
    except Exception as e:
        logger.error(f"Error reading config file {path}: {e}")
        raise

# --- Resource Monitoring Thread Function (Change 5) ---
def resource_monitor_thread(stop_event: threading.Event, results_queue: Queue, interval: float = 1.0):
    """
    Monitors CPU and Memory usage periodically and puts results in a queue.
    Runs in a background thread.
    """
    logger.info("Resource monitoring thread started.")
    process = psutil.Process(os.getpid()) # Monitor current process
    cpu_percents = []
    memory_mb = [] # Store memory in MB

    while not stop_event.is_set():
        try:
            # Get CPU percent (may require interval=None or initial call to be non-blocking)
            cpu_percent = process.cpu_percent(interval=None) # Use process CPU%
            # Get Memory usage (RSS - Resident Set Size)
            memory_info = process.memory_info()
            mem_mb = memory_info.rss / (1024 * 1024)

            cpu_percents.append(cpu_percent)
            memory_mb.append(mem_mb)

        except psutil.Error as e:
            logger.warning(f"psutil error during monitoring: {e}")
        except Exception as e:
             logger.error(f"Unexpected error in monitoring thread: {e}", exc_info=True)

        # Wait for the specified interval or until stop_event is set
        stop_event.wait(interval)

    logger.info(f"Resource monitoring thread stopping. Collected {len(cpu_percents)} samples.")
    # Put results into the queue for the main thread
    results_queue.put({
        "cpu_percents": cpu_percents,
        "memory_mb": memory_mb # Send memory in MB
    })

# --- Argument Parser ---
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Deduplication and Clustering Workflows using Ray"
    )
    parser.add_argument(
        "--workflow", type=str, required=True, choices=["nd_cl", "cl_nd"],
        help="Workflow to execute: 'nd_cl' (ND then CL) or 'cl_nd' (CL then ND within clusters)"
    )

    # --- Input/Output ---
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input file pattern (e.g., 'data/*.json.gz')"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Base output directory for workflow results"
    )
    parser.add_argument(
        "--limit_files", type=int, default=None, help="Limit the number of input files"
    )

    # --- ND Parameters (used in both workflows directly or via CL->ND) ---
    parser.add_argument("--threshold", type=float, default=0.7, help="MinHash Similarity threshold")
    parser.add_argument("--min_ngram_size", type=int, default=5, help="Min N-gram size for ND")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size for ND")
    parser.add_argument("--num_perm", type=int, default=256, help="Number of permutations for MinHash")
    parser.add_argument("--column", "-c", type=str, default="text", help="Column name for text data")

    # --- CL Parameters ---
    parser.add_argument(
        "--config_file", type=str, default=os.path.join(project_root, "database_project/src/configs/base.yml"), # Default path relative to project root
        help="Path to clustering config YAML"
    )

    # --- Execution Environment ---
    parser.add_argument(
        "--use_ray", type=bool, default=True, help="Flag indicating Ray usage (currently always True)"
    ) # Kept for potential future non-Ray paths, but logic assumes Ray.

    # --- Benchmarking ---
    parser.add_argument("--notes", type=str, default=None, help="Notes for benchmark DB entry")
    parser.add_argument("--mock", type=bool, default=False, help="Mock the execution (not fully implemented)")

    return parser

# --- Main Execution Logic ---
if __name__ == "__main__":
    args = create_parser().parse_args()
    workflow_start_time = time.time()
    logger.info(f"Starting workflow: {args.workflow}")
    logger.info(f"Running with arguments: {vars(args)}")

    # --- Variables for storing results across stages/workflows ---
    final_output_path = args.output # Base path for workflow output
    final_record_count = 0
    total_duplicate_count = 0 # Interpretation depends on workflow
    num_nodes_used = 1 # Default, will be updated after Ray init
    actual_workflow_time = 0
    nd_step_time = None             # Change 1: Time for ND step
    nd_output_count_for_log = None  # Change 3: Count after ND step
    cl_train_time = 0.0             # Change 6: Clustering train time
    cl_inference_time = 0.0         # Change 6: Clustering inference time
    cl_stage2_time = 0.0            # Change 7: Time for stage2 function
    cl_cluster_dist_json = None     # Change 7: Cluster distribution JSON

    # --- Initialize Ray ---
    try:
        # Connect to existing cluster started by run.sh/run_experiments.sh
        ray.init(address='auto', ignore_reinit_error=True)
        nodes = ray.nodes()
        num_nodes_used = len([n for n in nodes if n["alive"]])
        logger.info(f"Successfully connected to Ray cluster. Detected {num_nodes_used} live nodes.")
        logger.debug(f"Ray Nodes: {nodes}")
    except Exception as e:
        logger.error(f"Failed to initialize Ray or connect to cluster: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Clustering Config ---
    if not CONFIG_TOOLS_AVAILABLE:
        logger.error("Cannot proceed without ml_collections and PyYAML for config loading.")
        sys.exit(1)
    try:
        cfg = read_config(args.config_file)
        cfg.args = args # Inject command-line args into config for easy access in CL step
        # Pass should_dedup flag based on workflow for cl_nd case
        cfg.should_dedup = (args.workflow == "cl_nd")
        logger.info(f"Clustering config loaded. Deduplication within CL step: {cfg.should_dedup}")
    except Exception as e:
        logger.error(f"Failed to load clustering config file '{args.config_file}': {e}")
        sys.exit(1)


    # --- Start Resource Monitoring (Change 5) ---
    logger.info("Setting up resource monitoring...")
    stop_monitoring_event = threading.Event()
    resource_results_queue = Queue()
    monitor_thread = threading.Thread(
        target=resource_monitor_thread,
        args=(stop_monitoring_event, resource_results_queue, 1.0), # Check every 1 second
        daemon=True # Allows main program to exit even if thread is blocked
    )
    logger.info("Starting resource monitoring thread...")
    monitor_thread.start()
    # --- End Resource Monitoring Setup ---

    # --- Load Initial Data ---
    try:
        logger.info(f"Reading input data pattern: {args.input_file}")
        # Use glob to find files matching the pattern
        input_files = glob.glob(args.input_file)
        if not input_files:
            raise FileNotFoundError(f"No files found matching pattern: {args.input_file}")

        # Apply file limit if specified
        if args.limit_files is not None and args.limit_files > 0:
            logger.info(f"Limiting input to {args.limit_files} files.")
            input_files = input_files[:args.limit_files]
        else:
            logger.info(f"Using all {len(input_files)} found files.")

        # Calculate total size for logging
        calculated_size_gb = get_total_size_gb(input_files)
        logger.info(f"Calculated input size: {calculated_size_gb:.2f} GB" if calculated_size_gb is not None else "Input size calculation failed.")

        # Read data using Ray Data
        # Consider adding override_num_blocks based on config or cluster size
        num_blocks = cfg.get("num_blocks", max(100, num_nodes_used * 2)) # Example heuristic
        logger.info(f"Reading JSON files into Ray Dataset with {num_blocks} blocks...")
        ray_df = ray.data.read_json(input_files, override_num_blocks=num_blocks)
        initial_count = ray_df.count()
        logger.info(f"Initial dataset loaded with {initial_count} records.")

    except Exception as e:
        logger.error(f"Failed to load input data: {e}", exc_info=True)
        stop_monitoring_event.set() # Stop monitor thread on error
        sys.exit(1)

    # --- Main Workflow Execution ---
    clustered_ds = None # Placeholder for the final dataset handle
    try:
        # --- Execute Selected Workflow ---
        if args.workflow == "nd_cl":
            logger.info("Executing ND -> CL workflow...")
            # === Stage 1: ND ===
            logger.info("Running ND step...")
            nd_start = time.time()
            intermediate_ray_ds, nd_duplicates, nd_step_time = run_nd_step_for_workflow(ray_df, args) # Capture time (Change 1)
            nd_duration = time.time() - nd_start
            logger.info(f"ND step finished in {nd_duration:.2f}s (reported internal time: {nd_step_time:.2f}s).")

            # Log intermediate count (Change 3)
            nd_output_count_for_log = intermediate_ray_ds.count()
            logger.info(f"Record count after ND step: {nd_output_count_for_log}")
            total_duplicate_count = nd_duplicates # ND is the only source of duplicates here

            # Repartition before CL (consider making this configurable)
            cl_input_blocks = cfg.get("cl_input_repartition", 1000)
            logger.info(f"Repartitioning dataset to {cl_input_blocks} blocks before CL step...")
            intermediate_ray_ds = intermediate_ray_ds.repartition(cl_input_blocks).materialize()

            # === Stage 2: CL ===
            logger.info("Running CL step...")
            cl_start = time.time()
            # Update CL call to capture new return values (Change 6 & 7)
            clustered_ds, _, cl_train_time, cl_inference_time, cl_stage2_time, cl_cluster_dist_json = \
                run_cl_step_for_workflow(intermediate_ray_ds, cfg) # Ignore duplicate count from CL step in nd_cl
            cl_duration = time.time() - cl_start
            logger.info(f"CL step completed in {cl_duration:.2f}s.")

            # Final record count should be count of the result from CL
            final_record_count = clustered_ds.count()


        elif args.workflow == "cl_nd":
            logger.info("Executing CL -> ND workflow...")
            # CL step inherently includes ND via stage2's conditional call
            logger.info("Running CL step (including potential ND within clusters)...")
            cl_nd_start = time.time()
            # Update CL call to capture new return values (Change 6 & 7)
            clustered_ds, cl_nd_duplicates, cl_train_time, cl_inference_time, cl_stage2_time, cl_cluster_dist_json = \
                run_cl_step_for_workflow(ray_df, cfg)
            cl_nd_duration = time.time() - cl_nd_start

            total_duplicate_count = cl_nd_duplicates # Assign the duplicates found within clusters
            final_record_count = clustered_ds.count()  # Calculate final count *after* the combined step
            logger.info(f"CL->ND workflow completed in {cl_nd_duration:.2f}s.")
            logger.info(f"Total duplicates found across clusters: {total_duplicate_count}")
            logger.info(f"CL->ND workflow final record count: {final_record_count}")


        else:
            # Should not happen due to argparse choices
            raise ValueError(f"Invalid workflow specified: {args.workflow}")

        # --- Workflow Complete ---
        actual_workflow_time = time.time() - workflow_start_time
        logger.info(f"Workflow '{args.workflow}' finished. Total wall clock time: {actual_workflow_time:.2f} seconds.")

        # --- Optional: Write final output ---
        # final_output_path is passed to run_cl_step_for_workflow via cfg.args.output
        # Ensure that function actually writes the final 'clustered_ds' there if needed.
        # Example write:
        # logger.info(f"Writing final output dataset to: {args.output}")
        # clustered_ds.write_parquet(args.output)

    except Exception as e:
        logger.error(f"Workflow '{args.workflow}' failed during execution: {e}", exc_info=True)
        stop_monitoring_event.set() # Ensure monitor thread stops on error
        # Optionally log a failed run marker to DB
        sys.exit(1) # Exit with error status

    finally:
        # --- Stop Resource Monitoring and Get Results (Change 5) ---
        logger.info("Stopping resource monitoring thread...")
        stop_monitoring_event.set()
        monitor_thread.join(timeout=5) # Wait for thread to finish (with timeout)
        if monitor_thread.is_alive():
            logger.warning("Monitoring thread did not exit cleanly.")

        cpu_avg, cpu_max, mem_avg_mb, mem_max_mb = 0.0, 0.0, 0.0, 0.0
        resource_metrics_collected = False
        try:
            # Use get with timeout in case queue remains empty
            collected_metrics = resource_results_queue.get(timeout=1.0)
            cpu_percents = collected_metrics.get("cpu_percents", [])
            memory_mb_samples = collected_metrics.get("memory_mb", [])
            resource_metrics_collected = bool(cpu_percents or memory_mb_samples) # Mark if data exists

            if cpu_percents:
                cpu_avg = statistics.mean(cpu_percents)
                cpu_max = max(cpu_percents)
            if memory_mb_samples:
                mem_avg_mb = statistics.mean(memory_mb_samples)
                mem_max_mb = max(memory_mb_samples)

            if resource_metrics_collected:
                 logger.info(f"Resource Stats: CPU Avg={cpu_avg:.2f}%, CPU Max={cpu_max:.2f}%, Mem Avg={mem_avg_mb:.2f}MB, Mem Max={mem_max_mb:.2f}MB")
            else:
                 logger.info("No resource metrics were collected by the monitoring thread.")

        except Empty: # Changed from queue.Empty
            logger.warning("No resource metrics received from monitoring thread queue (queue empty).")
        except Exception as e:
            logger.error(f"Error processing resource metrics: {e}", exc_info=True)
        # --- End Resource Monitoring Processing ---

        # --- Prepare Full Configuration JSON (Change 4) ---
        full_config_details = {}
        try:
            # Convert args namespace to dict
            args_dict = vars(args)
            # Convert ConfigDict cfg to dict (handle potential nested ConfigDicts)
            def configdict_to_dict(cd):
                if isinstance(cd, config_dict.ConfigDict):
                     return {k: configdict_to_dict(v) for k, v in cd.items()}
                elif isinstance(cd, list):
                     return [configdict_to_dict(i) for i in cd]
                else:
                     return cd # Assume primitive type

            cfg_dict = configdict_to_dict(cfg)
            # Combine them (handle potential key collisions if necessary, e.g., args override cfg)
            full_config_details = {"args": args_dict, "config": cfg_dict}
            full_config_json = json.dumps(full_config_details, indent=2, default=str) # Use default=str for non-serializable
            logger.info("Full configuration details prepared for logging.")
            # logger.debug(f"Full Config JSON: {full_config_json}") # Can be very long
        except Exception as e:
            logger.error(f"Failed to serialize full configuration details: {e}")
            full_config_json = json.dumps({"error": "Failed to serialize config"})


        # --- Database Logging ---
        logger.info("Logging benchmark results to database...")
        db_engine = None
        db_session = None
        try:
            db_engine = init_db() # Initialize DB connection and tables if needed
            db_session = get_session(db_engine)

            # Prepare notes
            benchmark_notes = args.notes if args.notes else f"Workflow: {args.workflow}"
            benchmark_notes += f" | Config: {os.path.basename(args.config_file)}"
            benchmark_notes += f" | Files: {args.limit_files if args.limit_files else 'All'}"

            # Create BenchmarkRun entry using the extended create_from_args
            benchmark_run = BenchmarkRun.create_from_args(
                session=db_session,
                args=args,
                duplicate_count=total_duplicate_count, # Interpretation depends on workflow
                record_count=final_record_count,       # Final count after all steps
                execution_time=actual_workflow_time,   # Total wall clock time
                implementation=args.workflow,          # Use workflow name as implementation
                num_nodes=num_nodes_used,              # Max nodes used during the workflow
                notes=benchmark_notes,
                limit_files=args.limit_files,          # Log the limit used
                total_size_gb=calculated_size_gb,      # Log calculated size
                # --- Add new fields ---
                nd_time_sec=nd_step_time,                   # Change 1
                config_file_path=args.config_file,          # Change 2
                nd_output_count=nd_output_count_for_log,    # Change 3
                config_details_json=full_config_json,       # Change 4
                cl_train_time_sec=cl_train_time,            # Change 6
                cl_inference_time_sec=cl_inference_time,    # Change 6
                cl_stage2_time_sec=cl_stage2_time,          # Change 7
                cluster_size_distribution_json=cl_cluster_dist_json # Change 7
            )
            logger.info(f"Benchmark data saved with Run ID: {benchmark_run.id}")

            # --- Add Resource Metrics to DB (Change 5) ---
            if benchmark_run and benchmark_run.id is not None and resource_metrics_collected:
                try:
                    logger.info(f"Adding resource metrics to BenchmarkRun ID: {benchmark_run.id}")
                    # Ensure benchmark_run is still associated with the session
                    # If create_from_args committed and closed, might need re-fetching/merging
                    # Assuming benchmark_run is still valid within db_session:
                    benchmark_run.add_resource_metrics( # Pass session explicitly if needed
                        cpu_percent_avg=cpu_avg,
                        cpu_percent_max=cpu_max,
                        memory_usage_avg_mb=mem_avg_mb,
                        memory_usage_max_mb=mem_max_mb,
                        session=db_session # Pass session for safety
                    )
                    db_session.commit() # Commit the added resource metric
                    logger.info(f"Resource metrics successfully added to BenchmarkRun ID: {benchmark_run.id}")
                except Exception as e:
                    logger.error(f"Failed to add resource metrics to DB for Run ID {benchmark_run.id}: {e}", exc_info=True)
                    try: db_session.rollback() # Rollback on error
                    except: logger.error("Rollback failed after resource metrics error.")
            elif not resource_metrics_collected:
                 logger.info("Skipping resource metrics logging as no data was collected.")
            else:
               logger.error("Failed to create BenchmarkRun or get ID, cannot add resource metrics.")
            # --- End Add Resource Metrics ---

        except Exception as e:
            logger.error(f"Database logging failed: {e}", exc_info=True)
        finally:
            if db_session:
                db_session.close()
            # Consider stopping engine if it's not needed anymore
            # if db_engine:
            #     db_engine.dispose()

        # --- Ray Shutdown ---
        # Optional: Shut down Ray connection from the script's perspective
        # ray.shutdown() # Might disconnect other processes if not desired
        logger.info("Ray script finished interaction with the cluster.")

    logger.info(f"Workflow {args.workflow} completed successfully.")
    final_elapsed_time = time.time() - workflow_start_time
    logger.info(f"Total script execution time: {final_elapsed_time:.2f} seconds.")

# --- END FILE ---
```