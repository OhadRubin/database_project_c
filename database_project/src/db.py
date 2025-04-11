# /database_project/src/db.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, object_session as sa_object_session
from datetime import datetime
import os
import json # For config details serialization

Base = declarative_base()

class BenchmarkRun(Base):
    __tablename__ = 'benchmark_runs'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_file = Column(String(255))
    output_dir = Column(String(255))
    notes = Column(Text, nullable=True)
    duplicate_count = Column(Integer) # Meaning depends on workflow (ND total or CL->ND sum)
    record_count = Column(Integer) # Final record count after workflow
    implementation = Column(String(50)) # Stores workflow name ('nd_cl' or 'cl_nd')
    num_nodes = Column(Integer)
    threshold = Column(Float) # ND threshold
    ngram_size = Column(Integer) # ND ngram size
    min_ngram_size = Column(Integer) # ND min ngram size
    num_perm = Column(Integer) # ND num permutations
    execution_time = Column(Float) # Total wall clock time for the workflow
    limit_files = Column(Integer, nullable=True)
    total_size_gb = Column(Float, nullable=True)

    # --- NEW COLUMNS based on change.md ---
    nd_time_sec = Column(Float, nullable=True) # Time for the ND step (if applicable)
    nd_output_count = Column(Integer, nullable=True) # Records after ND step (for ND->CL)
    config_file_path = Column(String(255), nullable=True) # Path to clustering YAML config
    cl_train_time_sec = Column(Float, nullable=True) # Aggregated CL training time
    cl_inference_time_sec = Column(Float, nullable=True) # Aggregated CL inference time
    cl_stage2_time_sec = Column(Float, nullable=True) # Time for stage2 logic (if applicable)
    config_details_json = Column(Text, nullable=True) # JSON string of args + clustering config
    # cluster_size_distribution_json = Column(Text, nullable=True) # JSON string of final cluster counts

    # Relationships
    # resource_metrics = relationship("ResourceMetric", back_populates="benchmark_run", cascade="all, delete-orphan")
    # accuracy_metrics = relationship("AccuracyMetric", back_populates="benchmark_run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<BenchmarkRun(id={self.id}, timestamp={self.timestamp}, implementation={self.implementation})>"

    @classmethod
    def create_from_args(cls, session, args, duplicate_count, record_count, execution_time,
                       num_nodes=1, notes=None, implementation="pyspark", limit_files=None, total_size_gb=None,
                       # Add new parameters corresponding to new columns
                       nd_time_sec=None, nd_output_count=None, config_file_path=None,
                       cl_train_time_sec=None, cl_inference_time_sec=None, cl_stage2_time_sec=None,
                       config_details_json=None, 
                    #    cluster_size_distribution_json=None
                       ):
        """
        Create a new BenchmarkRun entry from command line args and results, including detailed metrics.
        """
        run = cls(
            input_file=args.input_file, # Use input_file pattern from args
            output_dir=args.output,
            duplicate_count=duplicate_count,
            record_count=record_count,
            implementation=implementation, # This should be args.workflow now
            num_nodes=num_nodes,
            threshold=args.threshold,
            ngram_size=args.ngram_size,
            min_ngram_size=args.min_ngram_size,
            num_perm=args.num_perm,
            execution_time=execution_time,
            notes=notes,
            limit_files=limit_files if limit_files is not None else args.limit_files,
            total_size_gb=total_size_gb,
            # Assign new fields
            nd_time_sec=nd_time_sec,
            nd_output_count=nd_output_count,
            config_file_path=config_file_path if config_file_path is not None else args.config_file,
            cl_train_time_sec=cl_train_time_sec,
            cl_inference_time_sec=cl_inference_time_sec,
            cl_stage2_time_sec=cl_stage2_time_sec,
            config_details_json=config_details_json,
            # cluster_size_distribution_json=cluster_size_distribution_json
        )
        session.add(run)
        # Commit is handled by the caller (run_workflows.py) after potentially adding metrics
        return run


def init_db(db_path=None):
    """Initialize the database, create tables if they don't exist"""
    if db_path is None:
        if "POSTGRES_ADDRESS" in os.environ:
            print("Using PostgreSQL")
            db_path = os.environ["POSTGRES_ADDRESS"]
        else:
            print("Using SQLite")
            db_path = 'sqlite:///benchmark_results.db'
            print(f"SQLite DB will be created at: {os.path.abspath(db_path)}")
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    """Create a session to interact with the database"""
    Session = sessionmaker(bind=engine)
    return Session()

# Helper function to get the object's session
def object_session(obj):
    """Get the session for an object"""
    return sa_object_session(obj)

# # Example usage for monitoring resource stats while running a benchmark
# def monitor_resources(benchmark_run_id, session, interval=1.0):
#     """
#     Monitor system resources and add to database
#     Requires psutil library

#     Parameters:
#     -----------
#     benchmark_run_id : int
#         ID of the benchmark run
#     session : SQLAlchemy session
#         Database session
#     interval : float
#         Monitoring interval in seconds
#     """
#     try:
#         import psutil
#         import time
#         import statistics

#         benchmark_run = session.query(BenchmarkRun).get(benchmark_run_id)
#         if not benchmark_run:
#             print(f"Benchmark run with ID {benchmark_run_id} not found")
#             return

#         cpu_percent = []
#         memory_percent = []
#         start_time = time.time()

#         # Get initial disk and network counters
#         initial_disk_io = psutil.disk_io_counters()
#         initial_net_io = psutil.net_io_counters()

#         try:
#             while True:
#                 cpu_percent.append(psutil.cpu_percent())
#                 memory_info = psutil.virtual_memory()
#                 memory_percent.append(memory_info.percent)
#                 time.sleep(interval)
#         except KeyboardInterrupt:
#             # Calculate resource metrics
#             run_time = time.time() - start_time

#             # Calculate disk and network usage
#             final_disk_io = psutil.disk_io_counters()
#             final_net_io = psutil.net_io_counters()

#             disk_read_mb = (final_disk_io.read_bytes - initial_disk_io.read_bytes) / (1024 * 1024)
#             disk_write_mb = (final_disk_io.write_bytes - initial_disk_io.write_bytes) / (1024 * 1024)
#             net_sent_mb = (final_net_io.bytes_sent - initial_net_io.bytes_sent) / (1024 * 1024)
#             net_recv_mb = (final_net_io.bytes_recv - initial_net_io.bytes_recv) / (1024 * 1024)

#             # Get system memory info to convert percent to MB
#             memory_info = psutil.virtual_memory()
#             total_memory_mb = memory_info.total / (1024 * 1024)

#             avg_memory_percent = statistics.mean(memory_percent) if memory_percent else 0
#             max_memory_percent = max(memory_percent) if memory_percent else 0

#             avg_memory_mb = (avg_memory_percent / 100) * total_memory_mb
#             max_memory_mb = (max_memory_percent / 100) * total_memory_mb

#             # Add resource metrics to database
#             # Note: add_resource_metrics no longer commits internally
#             # benchmark_run.add_resource_metrics(
#             #     cpu_percent_avg=statistics.mean(cpu_percent) if cpu_percent else 0,
#             #     cpu_percent_max=max(cpu_percent) if cpu_percent else 0,
#             #     memory_usage_avg_mb=avg_memory_mb,
#             #     memory_usage_max_mb=max_memory_mb,
#             #     network_sent_mb=net_sent_mb,
#             #     network_recv_mb=net_recv_mb,
#             #     disk_read_mb=disk_read_mb,
#             #     disk_write_mb=disk_write_mb
#             # )
#             # Commit is now handled by the caller (e.g., after the main benchmark run is added)
#             session.commit() # Commit here after adding the metrics for this monitoring session

#             print(f"Resource monitoring completed after {run_time:.2f} seconds. Metrics added and committed.")

#     except ImportError:
#         print("psutil library required for resource monitoring. Install with: pip install psutil")
#     except Exception as e:
#         print(f"Error during resource monitoring: {e}")
#         session.rollback() # Rollback if error occurs during metric addition/commit

if __name__ == '__main__':
    # Example usage
    engine = init_db()
    session = get_session(engine)
    print("Database initialized successfully")

    # Example of creating a run and adding metrics (caller commits)
    # try:
    #     # Create a dummy args object for testing
    #     class DummyArgs:
    #         input_file = "test_input"
    #         output = "test_output"
    #         threshold = 0.7
    #         ngram_size = 5
    #         min_ngram_size = 5
    #         num_perm = 256
    #         limit_files = 10
    #         config_file = "test_config.yml"

    #     dummy_args = DummyArgs()
    #     dummy_config = {"key": "value"}
    #     dummy_config_json = json.dumps(dummy_config)
    #     dummy_dist_json = json.dumps([{"cluster_A": 0, "count": 100}])

    #     new_run = BenchmarkRun.create_from_args(
    #         session=session,
    #         args=dummy_args,
    #         duplicate_count=10,
    #         record_count=90,
    #         execution_time=123.45,
    #         implementation="nd_cl",
    #         num_nodes=4,
    #         notes="Test run with new fields",
    #         total_size_gb=1.2,
    #         nd_time_sec=30.5,
    #         nd_output_count=95,
    #         cl_train_time_sec=20.1,
    #         cl_inference_time_sec=50.2,
    #         cl_stage2_time_sec=0.0,
    #         config_details_json=dummy_config_json,
    #         cluster_size_distribution_json=dummy_dist_json
    #     )
    #     print(f"Created run (before commit): {new_run}")

    #     # Add metrics (optional)
    #     new_run.add_resource_metrics(
    #         cpu_percent_avg=50.0, cpu_percent_max=90.0,
    #         memory_usage_avg_mb=1024, memory_usage_max_mb=2048
    #     )
    #     print(f"Added resource metrics (before commit)")

    #     # Commit the run and its metrics
    #     session.commit()
    #     print(f"Committed run with ID: {new_run.id}")

    # except Exception as e:
    #     print(f"Error during example usage: {e}")
    #     session.rollback()
    # finally:
    #     session.close()