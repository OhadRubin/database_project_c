import argparse
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psutil
import subprocess
from typing import Dict, List, Any, Tuple, Set
from subprocess import Popen, PIPE
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark')


class ResourceMonitor:
    """Monitor system resources during benchmark execution."""
    
    def __init__(self, interval=1.0):
        """
        Initialize resource monitor.
        
        Parameters
        ----------
        interval : float
            Sampling interval in seconds
        """
        self.interval = interval
        self.process = None
        self.monitoring = False
        self.cpu_percent = []
        self.memory_usage = []
        self.network_io_sent = []
        self.network_io_recv = []
        self.disk_io_read = []
        self.disk_io_write = []
        self.timestamps = []
    
    def _monitor_resources(self, pid):
        """
        Monitor resources for a given process.
        
        Parameters
        ----------
        pid : int
            Process ID to monitor
        """
        try:
            process = psutil.Process(pid)
            start_time = time.time()
            
            # Get initial network and disk IO counters
            net_io_start = psutil.net_io_counters()
            disk_io_start = psutil.disk_io_counters()
            last_time = start_time
            
            while self.monitoring:
                # Record timestamp
                current_time = time.time()
                self.timestamps.append(current_time - start_time)
                
                # Record CPU and memory usage
                self.cpu_percent.append(process.cpu_percent())
                self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                
                # Record network and disk IO
                net_io_current = psutil.net_io_counters()
                disk_io_current = psutil.disk_io_counters()
                
                self.network_io_sent.append(net_io_current.bytes_sent)
                self.network_io_recv.append(net_io_current.bytes_recv)
                self.disk_io_read.append(disk_io_current.read_bytes)
                self.disk_io_write.append(disk_io_current.write_bytes)
                
                # Sleep for the specified interval
                time.sleep(self.interval)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process may have terminated
            logger.warning(f"Process {pid} no longer accessible")
    
    def start(self, pid):
        """
        Start monitoring resources.
        
        Parameters
        ----------
        pid : int
            Process ID to monitor
        """
        import threading
        
        # Reset metrics
        self.cpu_percent = []
        self.memory_usage = []
        self.network_io_sent = []
        self.network_io_recv = []
        self.disk_io_read = []
        self.disk_io_write = []
        self.timestamps = []
        
        self.monitoring = True
        self.process = threading.Thread(target=self._monitor_resources, args=(pid,))
        self.process.daemon = True
        self.process.start()
        
        logger.info(f"Started monitoring resources for process {pid}")
    
    def stop(self):
        """Stop monitoring resources."""
        self.monitoring = False
        if self.process:
            self.process.join(timeout=5)
        logger.info("Stopped resource monitoring")
    
    def get_summary(self):
        """
        Get summary of resource usage.
        
        Returns
        -------
        Dict[str, Any]
            Resource usage summary
        """
        if not self.cpu_percent:
            return {
                "cpu_percent_avg": None,
                "cpu_percent_max": None,
                "memory_usage_avg_mb": None,
                "memory_usage_max_mb": None,
                "network_sent_mb": None,
                "network_recv_mb": None,
                "disk_read_mb": None,
                "disk_write_mb": None
            }
        
        # Calculate network and disk IO transferred during monitoring
        network_sent = max(self.network_io_sent) - min(self.network_io_sent)
        network_recv = max(self.network_io_recv) - min(self.network_io_recv)
        disk_read = max(self.disk_io_read) - min(self.disk_io_read)
        disk_write = max(self.disk_io_write) - min(self.disk_io_write)
        
        return {
            "cpu_percent_avg": np.mean(self.cpu_percent) if self.cpu_percent else None,
            "cpu_percent_max": max(self.cpu_percent) if self.cpu_percent else None,
            "memory_usage_avg_mb": np.mean(self.memory_usage) if self.memory_usage else None,
            "memory_usage_max_mb": max(self.memory_usage) if self.memory_usage else None,
            "network_sent_mb": network_sent / (1024 * 1024),
            "network_recv_mb": network_recv / (1024 * 1024),
            "disk_read_mb": disk_read / (1024 * 1024),
            "disk_write_mb": disk_write / (1024 * 1024)
        }


class DeduplicationBenchmark:
    """Benchmark different deduplication implementations."""
    
    def __init__(self, 
                 input_file: str, 
                 threshold: float = 0.7,
                 ngram_size: int = 5,
                 min_ngram_size: int = 5,
                 num_perm: int = 256,
                 column: str = "content",
                 output_dir: str = "benchmark_results"):
        """
        Initialize the benchmark.
        
        Parameters
        ----------
        input_file : str
            Path to the input file (CSV or Parquet)
        threshold : float
            Similarity threshold
        ngram_size : int
            N-gram size
        min_ngram_size : int
            Minimum document size to be considered
        num_perm : int
            Number of permutations
        column : str
            Column name containing document content
        output_dir : str
            Directory to store benchmark results
        """
        self.input_file = input_file
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.min_ngram_size = min_ngram_size
        self.num_perm = num_perm
        self.column = column
        self.output_dir = output_dir
        
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            "pyspark_multi": {},
            "pyspark_single": {},
            "python_tfidf": {}
        }
        
        # Create resource monitor
        self.resource_monitor = ResourceMonitor()
    
    def _run_command(self, command: List[str], monitor_resources=True) -> Tuple[int, str, str, float, Dict[str, Any]]:
        """
        Run a command and time its execution.
        
        Parameters
        ----------
        command : List[str]
            Command to run
        monitor_resources : bool
            Whether to monitor resources during execution
            
        Returns
        -------
        Tuple[int, str, str, float, Dict[str, Any]]
            Return code, stdout, stderr, execution time, resource metrics
        """
        logger.info(f"Running command: {' '.join(command)}")
        start_time = time.time()
        
        process = Popen(command, stdout=PIPE, stderr=PIPE, text=True)
        
        # Start resource monitoring if requested
        if monitor_resources:
            self.resource_monitor.start(process.pid)
        
        stdout, stderr = process.communicate()
        
        # Stop resource monitoring
        if monitor_resources:
            self.resource_monitor.stop()
            resource_metrics = self.resource_monitor.get_summary()
        else:
            resource_metrics = {}
        
        execution_time = time.time() - start_time
        
        logger.info(f"Command completed in {execution_time:.2f} seconds")
        return process.returncode, stdout, stderr, execution_time, resource_metrics
    
    def _get_duplicate_sets(self, output_path: str) -> List[Set[str]]:
        """
        Extract duplicate sets from the output.
        
        Parameters
        ----------
        output_path : str
            Path to the output directory
            
        Returns
        -------
        List[Set[str]]
            List of duplicate sets, where each set contains document IDs
        """
        # Use the read_results.py script to get the data, but capture the output
        # This function is to be used for accuracy comparison between methods
        result_command = [
            "python3.10",
            "read_results.py",
            "--input", output_path,
            "--output_format", "json"  # We'll need to add this option to read_results.py
        ]
        
        try:
            # We'll try to run the command directly to get JSON output
            # If read_results.py doesn't support JSON output yet, we'll return empty list
            process = subprocess.run(result_command, capture_output=True, text=True)
            if process.returncode == 0:
                try:
                    # Try to parse the output as JSON
                    result_data = json.loads(process.stdout)
                    
                    # Extract duplicate sets
                    duplicate_sets = []
                    for cluster in result_data:
                        if len(cluster) > 1:  # Only include clusters with duplicates
                            duplicate_sets.append(set(str(doc_id) for doc_id in cluster))
                    
                    return duplicate_sets
                except json.JSONDecodeError:
                    logger.warning("Could not parse output as JSON")
                    return []
            else:
                logger.warning("Failed to get duplicate sets from output")
                return []
        except Exception as e:
            logger.error(f"Error getting duplicate sets: {e}")
            return []

    def run_pyspark_multi(self, output_subdir="pyspark_multi") -> Dict[str, Any]:
        """
        Run the PySpark multi-worker implementation.
        
        Parameters
        ----------
        output_subdir : str
            Subdirectory to store results
            
        Returns
        -------
        Dict[str, Any]
            Results of the benchmark
        """
        output_path = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        
        command = [
            "python3.10", 
            "database_project/src/deduplication_spark.py",
            "--input_file", self.input_file,
            "--threshold", str(self.threshold),
            "--ngram_size", str(self.ngram_size),
            "--min_ngram_size", str(self.min_ngram_size),
            "--num_perm", str(self.num_perm),
            "--column", self.column,
            "--output", output_path
        ]
        
        return_code, stdout, stderr, execution_time, resource_metrics = self._run_command(command)
        
        # Count number of records after deduplication
        record_count = None
        if return_code == 0:
            # Assuming we have a similar read_results.py script that can count records
            count_command = [
                "python3.10",
                "read_results.py",
                "--input", output_path
            ]
            count_return_code, count_stdout, count_stderr, _, _ = self._run_command(
                count_command, monitor_resources=False
            )
            
            # Extract record count from output
            if count_return_code == 0:
                for line in count_stdout.split('\n'):
                    if "Total records after deduplication:" in line:
                        record_count = int(line.split(":")[-1].strip())
                        break
        
        # Get duplicate sets for accuracy comparison
        duplicate_sets = self._get_duplicate_sets(output_path)
        
        results = {
            "execution_time": execution_time,
            "return_code": return_code,
            "record_count": record_count,
            "output_path": output_path,
            "success": return_code == 0,
            "resource_metrics": resource_metrics,
            "duplicate_sets": duplicate_sets,
            "duplicate_count": len(duplicate_sets)
        }
        
        self.results["pyspark_multi"] = results
        return results
    
    def run_pyspark_single(self, output_subdir="pyspark_single") -> Dict[str, Any]:
        """
        Run the PySpark single-worker implementation.
        
        Parameters
        ----------
        output_subdir : str
            Subdirectory to store results
            
        Returns
        -------
        Dict[str, Any]
            Results of the benchmark
        """
        output_path = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        
        # Set Spark to local mode
        os.environ["SPARK_MASTER_HOST"] = "local[1]"
        
        command = [
            "python3.10", 
            "database_project/src/deduplication_spark.py",
            "--input_file", self.input_file,
            "--threshold", str(self.threshold),
            "--ngram_size", str(self.ngram_size),
            "--min_ngram_size", str(self.min_ngram_size),
            "--num_perm", str(self.num_perm),
            "--column", self.column,
            "--output", output_path
        ]
        
        return_code, stdout, stderr, execution_time, resource_metrics = self._run_command(command)
        
        # Count number of records after deduplication
        record_count = None
        if return_code == 0:
            # Assuming we have a similar read_results.py script that can count records
            count_command = [
                "python",
                "read_results.py",
                "--input", output_path
            ]
            count_return_code, count_stdout, count_stderr, _, _ = self._run_command(
                count_command, monitor_resources=False
            )
            
            # Extract record count from output
            if count_return_code == 0:
                for line in count_stdout.split('\n'):
                    if "Total records after deduplication:" in line:
                        record_count = int(line.split(":")[-1].strip())
                        break
        
        # Unset environment variable
        del os.environ["SPARK_MASTER_HOST"]
        
        # Get duplicate sets for accuracy comparison
        duplicate_sets = self._get_duplicate_sets(output_path)
        
        results = {
            "execution_time": execution_time,
            "return_code": return_code,
            "record_count": record_count,
            "output_path": output_path,
            "success": return_code == 0,
            "resource_metrics": resource_metrics,
            "duplicate_sets": duplicate_sets,
            "duplicate_count": len(duplicate_sets)
        }
        
        self.results["pyspark_single"] = results
        return results
    
    def run_python_tfidf(self, output_subdir="python_tfidf") -> Dict[str, Any]:
        """
        Run the pure Python implementation with TF-IDF clustering.
        Note: This is a placeholder until the implementation is available.
        
        Parameters
        ----------
        output_subdir : str
            Subdirectory to store results
            
        Returns
        -------
        Dict[str, Any]
            Results of the benchmark
        """
        output_path = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        
        # This is a placeholder until the pure Python implementation is available
        logger.warning("Pure Python with TF-IDF clustering implementation not available yet")
        
        # When the implementation is available, uncomment and modify the following:
        # command = [
        #     "python", 
        #     "database_project/src/deduplication_python.py",
        #     "--input_file", self.input_file,
        #     "--threshold", str(self.threshold),
        #     "--ngram_size", str(self.ngram_size),
        #     "--min_ngram_size", str(self.min_ngram_size),
        #     "--num_perm", str(self.num_perm),
        #     "--column", self.column,
        #     "--output", output_path,
        #     "--use_tfidf_clustering"  # Special flag for this implementation
        # ]
        # 
        # return_code, stdout, stderr, execution_time, resource_metrics = self._run_command(command)
        # 
        # # Count number of records after deduplication
        # record_count = None
        # if return_code == 0:
        #     count_command = [
        #         "python3.10",
        #         "read_results.py",
        #         "--input", output_path
        #     ]
        #     count_return_code, count_stdout, count_stderr, _, _ = self._run_command(
        #         count_command, monitor_resources=False
        #     )
        #     
        #     # Extract record count from output
        #     if count_return_code == 0:
        #         for line in count_stdout.split('\n'):
        #             if "Total records after deduplication:" in line:
        #                 record_count = int(line.split(":")[-1].strip())
        #                 break
        #
        # # Get duplicate sets for accuracy comparison
        # duplicate_sets = self._get_duplicate_sets(output_path)
        
        results = {
            "execution_time": None,
            "return_code": None,
            "record_count": None,
            "output_path": output_path,
            "success": False,
            "message": "Implementation not available yet",
            "resource_metrics": {},
            "duplicate_sets": [],
            "duplicate_count": 0
        }
        
        self.results["python_tfidf"] = results
        return results
    
    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all implementations and return results.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Results of all benchmarks
        """
        logger.info("Running all benchmarks")
        
        self.run_pyspark_multi()
        self.run_pyspark_single()
        self.run_python_tfidf()
        
        # Save results to file
        with open(os.path.join(self.output_dir, "benchmark_results.json"), 'w') as f:
            # We need to convert the sets to lists for JSON serialization
            serializable_results = self.results.copy()
            for impl_name, impl_results in serializable_results.items():
                if "duplicate_sets" in impl_results:
                    duplicate_sets = impl_results["duplicate_sets"]
                    serializable_results[impl_name]["duplicate_sets"] = [list(s) for s in duplicate_sets]
            
            json.dump(serializable_results, f, indent=2)
        
        # Perform accuracy analysis if we have multiple implementations
        self._perform_accuracy_analysis()
        
        return self.results
    
    def _perform_accuracy_analysis(self) -> None:
        """
        Perform accuracy analysis by comparing duplicate sets across implementations.
        """
        logger.info("Performing accuracy analysis")
        
        # Create a reference implementation (we'll use pyspark_multi if available)
        reference_impl = None
        reference_sets = []
        
        for impl_name, results in self.results.items():
            if results.get("success", False) and results.get("duplicate_sets"):
                reference_impl = impl_name
                reference_sets = results["duplicate_sets"]
                break
        
        if not reference_impl:
            logger.warning("No successful implementation with duplicate sets found for accuracy analysis")
            return
        
        accuracy_results = {}
        
        for impl_name, results in self.results.items():
            if not results.get("success", False) or impl_name == reference_impl:
                continue
            
            impl_sets = results.get("duplicate_sets", [])
            
            # Calculate metrics
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            # Check each duplicate pair in the reference implementation
            for ref_set in reference_sets:
                # For each pair in the reference set, check if it's in the implementation
                pairs_found = 0
                total_pairs = len(ref_set) * (len(ref_set) - 1) // 2  # n choose 2
                
                for impl_set in impl_sets:
                    # Count the number of pairs in both sets
                    common_docs = ref_set.intersection(impl_set)
                    if len(common_docs) > 1:
                        pairs_found += len(common_docs) * (len(common_docs) - 1) // 2
                
                true_positives += pairs_found
                false_negatives += total_pairs - pairs_found
            
            # Check each duplicate pair in the implementation that's not in reference
            for impl_set in impl_sets:
                pairs_found_in_ref = 0
                total_impl_pairs = len(impl_set) * (len(impl_set) - 1) // 2
                
                for ref_set in reference_sets:
                    common_docs = impl_set.intersection(ref_set)
                    if len(common_docs) > 1:
                        pairs_found_in_ref += len(common_docs) * (len(common_docs) - 1) // 2
                
                false_positives += total_impl_pairs - pairs_found_in_ref
            
            # Calculate precision, recall, F1
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            accuracy_results[impl_name] = {
                "reference_implementation": reference_impl,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        
        # Save accuracy results
        with open(os.path.join(self.output_dir, "accuracy_results.json"), 'w') as f:
            json.dump(accuracy_results, f, indent=2)
        
        # Update main results with accuracy
        for impl_name, acc_results in accuracy_results.items():
            self.results[impl_name]["accuracy"] = acc_results
    
    def create_report(self) -> None:
        """
        Create a report of the benchmark results.
        """
        logger.info("Creating benchmark report")
        
        # Create DataFrame for comparison
        data = []
        
        for implementation, results in self.results.items():
            row = {
                "Implementation": implementation,
                "Execution Time (s)": results.get("execution_time"),
                "Record Count": results.get("record_count"),
                "Success": results.get("success", False)
            }
            
            # Add resource metrics if available
            resource_metrics = results.get("resource_metrics", {})
            for metric_name, metric_value in resource_metrics.items():
                row[f"Resource_{metric_name}"] = metric_value
            
            # Add accuracy metrics if available
            accuracy = results.get("accuracy", {})
            for metric_name, metric_value in accuracy.items():
                if metric_name in ["precision", "recall", "f1_score"]:
                    row[f"Accuracy_{metric_name}"] = metric_value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(os.path.join(self.output_dir, "benchmark_report.csv"), index=False)
        
        # Generate plots
        self._generate_plots()
    
    def _generate_plots(self) -> None:
        """Generate plots for benchmark results."""
        # Only generate plots if we have at least one successful result
        if not any(r.get("success", False) for r in self.results.values()):
            logger.warning("No successful benchmarks to plot")
            return
        
        # Filter out unsuccessful implementations
        successful_results = {
            name: results for name, results in self.results.items() 
            if results.get("success", False)
        }
        
        if not successful_results:
            return
        
        # 1. Performance comparison plot
        plt.figure(figsize=(10, 6))
        
        implementations = list(successful_results.keys())
        times = [results["execution_time"] for results in successful_results.values()]
        
        plt.bar(implementations, times)
        plt.title("Deduplication Execution Time Comparison")
        plt.xlabel("Implementation")
        plt.ylabel("Execution Time (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "execution_time_plot.png"))
        logger.info(f"Execution time plot saved to {os.path.join(self.output_dir, 'execution_time_plot.png')}")
        
        # 2. Memory usage comparison
        plt.figure(figsize=(10, 6))
        
        memory_usage = [
            results.get("resource_metrics", {}).get("memory_usage_max_mb", 0) 
            for results in successful_results.values()
        ]
        
        if any(memory_usage):  # Only plot if we have data
            plt.bar(implementations, memory_usage)
            plt.title("Maximum Memory Usage Comparison")
            plt.xlabel("Implementation")
            plt.ylabel("Memory Usage (MB)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, "memory_usage_plot.png"))
            logger.info(f"Memory usage plot saved to {os.path.join(self.output_dir, 'memory_usage_plot.png')}")
        
        # 3. Network usage comparison
        plt.figure(figsize=(10, 6))
        
        network_sent = [
            results.get("resource_metrics", {}).get("network_sent_mb", 0) 
            for results in successful_results.values()
        ]
        network_recv = [
            results.get("resource_metrics", {}).get("network_recv_mb", 0) 
            for results in successful_results.values()
        ]
        
        if any(network_sent) or any(network_recv):  # Only plot if we have data
            x = np.arange(len(implementations))
            width = 0.35
            
            plt.bar(x - width/2, network_sent, width, label='Sent')
            plt.bar(x + width/2, network_recv, width, label='Received')
            
            plt.title("Network Usage Comparison")
            plt.xlabel("Implementation")
            plt.ylabel("Network I/O (MB)")
            plt.xticks(x, implementations, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, "network_usage_plot.png"))
            logger.info(f"Network usage plot saved to {os.path.join(self.output_dir, 'network_usage_plot.png')}")
        
        # 4. Accuracy comparison (if available)
        implementations_with_accuracy = [
            name for name, results in successful_results.items() 
            if "accuracy" in results
        ]
        
        if implementations_with_accuracy:
            plt.figure(figsize=(10, 6))
            
            precision = [successful_results[impl]["accuracy"]["precision"] for impl in implementations_with_accuracy]
            recall = [successful_results[impl]["accuracy"]["recall"] for impl in implementations_with_accuracy]
            f1 = [successful_results[impl]["accuracy"]["f1_score"] for impl in implementations_with_accuracy]
            
            x = np.arange(len(implementations_with_accuracy))
            width = 0.25
            
            plt.bar(x - width, precision, width, label='Precision')
            plt.bar(x, recall, width, label='Recall')
            plt.bar(x + width, f1, width, label='F1 Score')
            
            plt.title("Accuracy Metrics Comparison")
            plt.xlabel("Implementation")
            plt.ylabel("Score")
            plt.xticks(x, implementations_with_accuracy, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, "accuracy_plot.png"))
            logger.info(f"Accuracy plot saved to {os.path.join(self.output_dir, 'accuracy_plot.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark deduplication implementations")
    parser.add_argument("--input_file", required=True, help="Input file path (CSV or Parquet)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size")
    parser.add_argument("--min_ngram_size", type=int, default=5, help="Minimum document size")
    parser.add_argument("--num_perm", type=int, default=256, help="Number of permutations")
    parser.add_argument("--column", type=str, default="content", help="Content column name")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--skip_pyspark_multi", action="store_true", help="Skip PySpark multi-worker benchmark")
    parser.add_argument("--skip_pyspark_single", action="store_true", help="Skip PySpark single-worker benchmark")
    parser.add_argument("--skip_python_tfidf", action="store_true", help="Skip Python TF-IDF benchmark")
    args = parser.parse_args()
    
    benchmark = DeduplicationBenchmark(
        input_file=args.input_file,
        threshold=args.threshold,
        ngram_size=args.ngram_size,
        min_ngram_size=args.min_ngram_size,
        num_perm=args.num_perm,
        column=args.column,
        output_dir=args.output_dir
    )
    
    # Run selected benchmarks
    results = {}
    
    if not args.skip_pyspark_multi:
        results["pyspark_multi"] = benchmark.run_pyspark_multi()
    
    if not args.skip_pyspark_single:
        results["pyspark_single"] = benchmark.run_pyspark_single()
    
    if not args.skip_python_tfidf:
        results["python_tfidf"] = benchmark.run_python_tfidf()
    
    # Perform accuracy analysis
    benchmark._perform_accuracy_analysis()
    
    # Create report
    benchmark.create_report()
    
    # Print summary
    print("\nBenchmark Summary:")
    for implementation, result in results.items():
        status = "Success" if result.get("success", False) else "Failed"
        time_str = f"{result.get('execution_time', 'N/A'):.2f}s" if result.get("execution_time") else "N/A"
        record_count = result.get("record_count", "N/A")
        
        print(f"- {implementation}: {status}, Time: {time_str}, Records: {record_count}")