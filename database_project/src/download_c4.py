import requests
from pyspark import SparkFiles
from pyspark import SparkContext
from pyspark.sql import SparkSession
import httpx
import asyncio
from pathlib import Path

spark = SparkSession.builder.appName("read_json") \
    .config("spark.local.dir", "/dev/shm/pyspark_dir") \
    .getOrCreate()

# Option 1: Download using requests and then read
url_tempalte = 'https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.{index:05d}-of-01024.json.gz'

files_to_download = [url_tempalte.format(index=i) for i in range(40)]

async def download_file(client, url, output_path):
    if output_path.exists():
        return
    # Configure follow_redirects=True to handle 302 redirects
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"Downloaded {output_path}")

async def download_files_in_batches(files_list, batch_size=10):
    output_dir = Path("/dev/shm/c4_files")
    output_dir.mkdir(exist_ok=True)
    
    all_output_paths = []
    
    # Process files in batches
    for i in range(0, len(files_list), batch_size):
        batch = files_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}, files {i}-{min(i+batch_size-1, len(files_list)-1)}")
        
        # Create client with follow_redirects=True
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = []
            for url in batch:
                filename = url.split('/')[-1]
                output_path = output_dir / filename
                all_output_paths.append(output_path)
                tasks.append(download_file(client, url, output_path))
            
            await asyncio.gather(*tasks)
    
    return str(output_dir / "*.json.gz")

import time

start_time = time.time()

local_file = asyncio.run(download_files_in_batches(files_to_download))

# Read from local file
df = spark.read.json(local_file)

# Example: Display first few rows
df.show(5)

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

# python3.10 spark_read_json.py