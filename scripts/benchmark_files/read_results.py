import argparse
import json
from pyspark.sql import SparkSession


def read_results(input_path="output", output_format="text"):
    # Initialize Spark session
    spark = SparkSession.builder.appName("ReadResults").getOrCreate()

    # Read the deduplicated data
    df = spark.read.parquet(input_path)

    # Show the results based on output format
    if output_format == "text":
        print(f"Total records after deduplication: {df.count()}")
        df.show(truncate=False)
    elif output_format == "json":
        # Get the data as clusters in JSON format
        # Collect the data into a list of dictionaries
        results = df.collect()
        
        # Organize by cluster
        clusters = {}
        for row in results:
            cluster_id = row["cluster_id"]
            doc_id = row["id"]
            
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            
            clusters[cluster_id].append(doc_id)
        
        # Convert to list of clusters for JSON output
        cluster_list = list(clusters.values())
        
        # Output as JSON
        print(json.dumps(cluster_list))
    else:
        print(f"Unknown output format: {output_format}")

    # Stop the Spark session
    spark.stop()
    

# python read_results.py --input test_benchmark_results/pyspark_multi
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read deduplicated results")
    parser.add_argument("--input", type=str, default="output", help="Input directory with Parquet files")
    parser.add_argument("--output_format", type=str, choices=["text", "json"], default="text", 
                        help="Output format (text or json)")
    args = parser.parse_args()
    
    read_results(args.input, args.output_format)