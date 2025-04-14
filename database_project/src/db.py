# /database_project/src/db.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, object_session as sa_object_session
from datetime import datetime
import os
import json # For config details serialization

Base = declarative_base()

class BenchmarkRun(Base):
    __tablename__ = 'benchmark_runs_v2'

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
    
    config_details_json = Column(Text, nullable=True) # JSON string of args + clustering config
    metrics = Column(JSON, nullable=True) # JSON string of metrics


    def __repr__(self):
        return f"<BenchmarkRun(id={self.id}, timestamp={self.timestamp}, implementation={self.implementation})>"

    @classmethod
    def create_from_args(cls, session, args, duplicate_count, record_count, execution_time,
                       num_nodes=1, notes=None, implementation="pyspark", limit_files=None, total_size_gb=None,
                       metrics=None,
                       config_details_json=None, 
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
            metrics=metrics,
            config_details_json=config_details_json,
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

if __name__ == '__main__':
    # Example usage
    engine = init_db()
    session = get_session(engine)
    print("Database initialized successfully")
