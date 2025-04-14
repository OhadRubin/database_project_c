import sys 
import numpy as np
sys.path.append('/home/ohadr/database_project_c')
from database_project.src.ray_minhash import RayBTSMinhashDeduplicator, run_nd_step_for_workflow, tokenize

import ray
ray.init(address='auto',
            dashboard_host="0.0.0.0",
            # log_to_driver=False # Keep logs separate per node if needed
            ignore_reinit_error=True # Allow re-initialization if already connected
            )
import glob
input_files = glob.glob("/dev/shm/c4_files/c4-train.*.json.gz")[:1]
ray_df = ray.data.read_json(input_files)
print(ray_df.count())

import argparse
args = argparse.Namespace(text_key="text",
                          ngram_size=5,
                          min_ngram_size=5,
                          num_perm=256,
                          threshold=0.5,
                          column="text",
                          union_find_parallel_num=2,
                          union_threshold=256,
                          max_pending_edge_buffer_task=20,
                          num_edge_buffer_task_returns=10,
                          max_pending_filter_tasks=20,
                          num_filter_task_returns=10,
                          merge_batch_size=100)
intermediate_ray_ds, nd_duplicates, nd_step_time = run_nd_step_for_workflow(ray_df, args, mode="tag")
print(intermediate_ray_ds.count())
print(nd_duplicates)
print(nd_step_time)



def jaccard(set_a, set_b):
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))




# Count false positives per duplicate set
def analyze_duplicate_set(group_df, ngram_size, min_ngram_size):
    if len(group_df) <= 1:  # Skip singleton groups
        output = -1
    else:
    
        # Calculate Jaccard similarity for all pairs in this group
        texts = group_df["text"].tolist()
        tokenized = [tokenize(text, ngram_size=ngram_size, min_ngram_size=min_ngram_size) for text in texts]
        
        false_positive_count = 0
        total_pairs = 0
        
        for i in range(len(tokenized)):
            for j in range(i+1, len(tokenized)):
                if i == j:
                    continue
                similarity = jaccard(tokenized[i], tokenized[j])
                if similarity < args.threshold:  # This pair is a false positive
                    false_positive_count += 1
                total_pairs += 1
        
        false_positive_rate = false_positive_count / total_pairs if total_pairs > 0 else 0
        
        output=float(false_positive_rate) 
        
    return {"false_positive_rate": np.array([output])}



# Map each group to its false positive rate
false_positive_stats = intermediate_ray_ds.groupby("duplicate_set_id").map_groups(
    analyze_duplicate_set, 
    batch_format="pandas"
)

false_positive_stats = false_positive_stats.filter(lambda x: x["false_positive_rate"] >= 0).mean("false_positive_rate").materialize()
