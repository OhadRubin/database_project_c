# database_project_c

A benchmark system for data deduplication

## Database Schema

This project uses SQLAlchemy to track and analyze deduplication benchmark runs with the following schema:

1. **benchmark_runs** - Tracks each benchmark execution
   - Basic information (run timestamp, input/output paths)
   - Parameters (threshold, ngram_size, min_ngram_size, num_perm)
   - Results (execution_time, duplicate_count, record_count)

2. **resource_metrics** - System resource usage metrics
   - CPU and memory utilization
   - Network and disk I/O

3. **accuracy_metrics** - Comparison metrics
   - Precision, recall, F1 score
   - True/false positives, false negatives

## Usage

### Setting up the database

Initialize the database and tables:

```python
from benchmark_db import init_db, get_session

engine = init_db()  # Default: sqlite:///benchmark_results.db
session = get_session(engine)
```

### Running a benchmark with automatic tracking

The simplest way is to use the benchmark helper script which manages the database for you:

```bash
python3.10 scripts/benchmark_helper.py --monitor --notes "First benchmark run" -- \
  --input_file "/path/to/data.json" --output /path/to/output \
  --threshold 0.7 --ngram_size 5 --min_ngram_size 5 --num_perm 256
```

Arguments before `--` are for the benchmark helper, arguments after are passed directly to the deduplication script.

### Viewing benchmark results

Use the show_benchmark_results.py script to view and compare benchmark runs:

```bash
# List all benchmark runs
python3.10 scripts/show_benchmark_results.py

# Show details for a specific run
python3.10 scripts/show_benchmark_results.py --id 1

# Compare multiple runs
python3.10 scripts/show_benchmark_results.py --compare 1 2 3

# Sort by execution time (descending)
python3.10 scripts/show_benchmark_results.py --sort execution_time --desc
```

### Manually creating benchmark entries

You can also create entries programmatically:

```python
from benchmark_db import init_db, get_session, BenchmarkRun

engine = init_db()
session = get_session(engine)

# Create a benchmark run
run = BenchmarkRun.create_from_spark_run(
    session=session,
    input_file="/path/to/data.csv",
    output_dir="/path/to/output",
    duplicate_count=1000,
    record_count=9000,
    threshold=0.7,
    ngram_size=5,
    min_ngram_size=5,
    num_perm=256,
    execution_time=120.5,
    notes="Test run"
)

# Add resource metrics
run.add_resource_metrics(
    cpu_percent_avg=75.2,
    cpu_percent_max=99.8,
    memory_usage_avg_mb=4200.5,
    memory_usage_max_mb=4800.2
)

# Add accuracy metrics
run.add_accuracy_metrics(
    reference_implementation="tfidf",
    true_positives=950,
    false_positives=50,
    false_negatives=30,
    precision=0.95,
    recall=0.97,
    f1_score=0.96
)
```

## Requirements

- Python 3.10+
- SQLAlchemy
- tabulate (for displaying results)
- psutil (for resource monitoring)
