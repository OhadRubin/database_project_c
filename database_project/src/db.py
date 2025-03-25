from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class BenchmarkRun(Base):
    __tablename__ = 'benchmark_runs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_file = Column(String(255))
    output_dir = Column(String(255))
    notes = Column(Text, nullable=True)
    duplicate_count = Column(Integer)
    record_count = Column(Integer)
    implementation = Column(String(50))
    num_nodes = Column(Integer)
    threshold = Column(Float)
    ngram_size = Column(Integer)
    min_ngram_size = Column(Integer)
    num_perm = Column(Integer)
    execution_time = Column(Float)
    limit_files = Column(Integer, nullable=True)
    total_size_gb = Column(Float, nullable=True)
    
    # Relationships
    resource_metrics = relationship("ResourceMetric", back_populates="benchmark_run", cascade="all, delete-orphan")
    accuracy_metrics = relationship("AccuracyMetric", back_populates="benchmark_run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<BenchmarkRun(id={self.id}, timestamp={self.timestamp}, implementation={self.implementation})>"
    
    @classmethod
    def create_from_spark_run(cls, session, input_file, output_dir, duplicate_count, record_count, 
                            threshold, ngram_size, min_ngram_size, num_perm, execution_time, 
                            num_nodes=1, notes=None, implementation="pyspark", limit_files=None, total_size_gb=None):
        """
        Create a new BenchmarkRun entry from a PySpark deduplication run
        
        Parameters:
        -----------
        session : SQLAlchemy session
            Database session
        input_file : str
            Source data file path
        output_dir : str
            Results directory
        duplicate_count : int
            Number of duplicate sets found
        record_count : int
            Records after deduplication
        threshold : float
            Similarity threshold used
        ngram_size : int
            N-gram size parameter
        min_ngram_size : int
            Minimum document size
        num_perm : int
            Number of permutations
        execution_time : float
            Runtime in seconds
        num_nodes : int, optional
            Number of nodes in the cluster
        notes : str, optional
            Additional notes about the run
        implementation : str, optional
            Implementation type, defaults to "pyspark"
        limit_files : int, optional
            Number of files processed (if limited)
        total_size_gb : float, optional
            Total size of processed files in GB
            
        Returns:
        --------
        BenchmarkRun
            The created BenchmarkRun instance
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
        )
        session.add(run)
        session.commit()
        return run
    
    @classmethod
    def create_from_args(cls, session, args, duplicate_count, record_count, execution_time, 
                       num_nodes=1, notes=None, implementation="pyspark", limit_files=None, total_size_gb=None):
        """
        Create a new BenchmarkRun entry from command line args and results
        
        Parameters:
        -----------
        session : SQLAlchemy session
            Database session
        args : argparse.Namespace
            Command line arguments from create_parser()
        duplicate_count : int
            Number of duplicate sets found
        record_count : int
            Records after deduplication
        execution_time : float
            Runtime in seconds
        num_nodes : int, optional
            Number of nodes in the cluster
        notes : str, optional
            Additional notes about the run
        implementation : str, optional
            Implementation type, defaults to "pyspark"
        limit_files : int, optional
            Number of files processed (if limited)
        total_size_gb : float, optional
            Total size of processed files in GB
            
        Returns:
        --------
        BenchmarkRun
            The created BenchmarkRun instance
        """
        run = cls(
            input_file=args.input_file or args.table,
            output_dir=args.output,
            duplicate_count=duplicate_count,
            record_count=record_count,
            implementation=implementation,
            num_nodes=num_nodes,
            threshold=args.threshold,
            ngram_size=args.ngram_size,
            min_ngram_size=args.min_ngram_size,
            num_perm=args.num_perm,
            execution_time=execution_time,
            notes=notes,
            limit_files=limit_files if limit_files is not None else args.limit_files,
            total_size_gb=total_size_gb
        )
        session.add(run)
        session.commit()
        return run
    
    def add_resource_metrics(self, cpu_percent_avg, cpu_percent_max, memory_usage_avg_mb, 
                           memory_usage_max_mb, network_sent_mb=0, network_recv_mb=0, 
                           disk_read_mb=0, disk_write_mb=0):
        """
        Add resource metrics for this benchmark run
        
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
            
        Returns:
        --------
        ResourceMetric
            The created ResourceMetric instance
        """
        session = object_session(self)
        resource_metric = ResourceMetric(
            result_id=self.id,
            cpu_percent_avg=cpu_percent_avg,
            cpu_percent_max=cpu_percent_max,
            memory_usage_avg_mb=memory_usage_avg_mb,
            memory_usage_max_mb=memory_usage_max_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb
        )
        self.resource_metrics.append(resource_metric)
        session.commit()
        return resource_metric
    
    def add_accuracy_metrics(self, reference_implementation, true_positives, false_positives, 
                           false_negatives, precision, recall, f1_score):
        """
        Add accuracy metrics for this benchmark run
        
        Parameters:
        -----------
        reference_implementation : str
            Reference implementation used for comparison
        true_positives : int
            Number of true positives
        false_positives : int
            Number of false positives
        false_negatives : int
            Number of false negatives
        precision : float
            Precision score
        recall : float
            Recall score
        f1_score : float
            F1 score
            
        Returns:
        --------
        AccuracyMetric
            The created AccuracyMetric instance
        """
        session = object_session(self)
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
        session.commit()
        return accuracy_metric

class ResourceMetric(Base):
    __tablename__ = 'resource_metrics'
    
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('benchmark_runs.id'))
    cpu_percent_avg = Column(Float)
    cpu_percent_max = Column(Float)
    memory_usage_avg_mb = Column(Float)
    memory_usage_max_mb = Column(Float)
    network_sent_mb = Column(Float)
    network_recv_mb = Column(Float)
    disk_read_mb = Column(Float)
    disk_write_mb = Column(Float)  # Corrected from 'resulte_mb'
    
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

def init_db(db_path='sqlite:///benchmark_results.db'):
    """Initialize the database, create tables if they don't exist"""
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
    from sqlalchemy.orm import object_session as sa_object_session
    return sa_object_session(obj)

# Example usage for monitoring resource stats while running a benchmark
def monitor_resources(benchmark_run_id, session, interval=1.0):
    """
    Monitor system resources and add to database
    Requires psutil library
    
    Parameters:
    -----------
    benchmark_run_id : int
        ID of the benchmark run
    session : SQLAlchemy session
        Database session
    interval : float
        Monitoring interval in seconds
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
            
            disk_read_mb = (final_disk_io.read_bytes - initial_disk_io.read_bytes) / (1024 * 1024)
            disk_write_mb = (final_disk_io.write_bytes - initial_disk_io.write_bytes) / (1024 * 1024)
            net_sent_mb = (final_net_io.bytes_sent - initial_net_io.bytes_sent) / (1024 * 1024)
            net_recv_mb = (final_net_io.bytes_recv - initial_net_io.bytes_recv) / (1024 * 1024)
            
            # Get system memory info to convert percent to MB
            memory_info = psutil.virtual_memory()
            total_memory_mb = memory_info.total / (1024 * 1024)
            
            avg_memory_percent = statistics.mean(memory_percent)
            max_memory_percent = max(memory_percent)
            
            avg_memory_mb = (avg_memory_percent / 100) * total_memory_mb
            max_memory_mb = (max_memory_percent / 100) * total_memory_mb
            
            # Add resource metrics to database
            benchmark_run.add_resource_metrics(
                cpu_percent_avg=statistics.mean(cpu_percent),
                cpu_percent_max=max(cpu_percent),
                memory_usage_avg_mb=avg_memory_mb,
                memory_usage_max_mb=max_memory_mb,
                network_sent_mb=net_sent_mb,
                network_recv_mb=net_recv_mb,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb
            )
            
            print(f"Resource monitoring completed after {run_time:.2f} seconds")
            
    except ImportError:
        print("psutil library required for resource monitoring. Install with: pip install psutil")

if __name__ == '__main__':
    # Example usage
    engine = init_db()
    session = get_session(engine)
    print("Database initialized successfully")
