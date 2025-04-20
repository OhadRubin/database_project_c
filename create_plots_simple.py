import matplotlib.pyplot as plt

# Data
dataset_sizes = [3, 6, 12]

fp_macro_nd_cl = [0.453, 0.468, 0.471]
fp_macro_cl_nd = [0.441, 0.416, 0.431]

fp_micro_nd_cl = [0.778, 0.793, 0.823]
fp_micro_cl_nd = [0.760, 0.790, 0.818]

thresholds = [0.6, 0.7, 0.8, 0.9]
dup_thresh_nd_cl = [530000, 420000, 240000,  85000]
dup_thresh_cl_nd = [510000, 405000, 240000,  86000]

permutations = [128, 256, 512]
dup_perm_nd_cl = [402500, 421600, 392000]
dup_perm_cl_nd = [390000, 407500, 380000]

cl_inf_nd_cl = [150, 200, 295]
cl_inf_cl_nd = [175, 213, 298]

total_time_nd_cl = [570, 635, 790]
total_time_cl_nd = [530, 550, 735]

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Top-left: Macro FPR vs Dataset Size
ax = axs[0, 0]
ax.plot(dataset_sizes, fp_macro_nd_cl, '-o', label='nd_cl')
ax.plot(dataset_sizes, fp_macro_cl_nd, '-o', label='cl_nd')
ax.set_title('Macro False Positive Rate vs Dataset Size')
ax.set_xlabel('Dataset Size (GB)')
ax.set_ylabel('False Positive Rate')
ax.grid(True)
ax.legend()

# Top-middle: Micro FPR vs Dataset Size
ax = axs[0, 1]
ax.plot(dataset_sizes, fp_micro_nd_cl, '-o', label='nd_cl')
ax.plot(dataset_sizes, fp_micro_cl_nd, '-o', label='cl_nd')
ax.set_title('Micro False Positive Rate vs Dataset Size')
ax.set_xlabel('Dataset Size (GB)')
ax.set_ylabel('False Positive Rate')
ax.grid(True)
ax.legend()

# Top-right: Duplicate Count vs Similarity Threshold
ax = axs[0, 2]
ax.plot(thresholds, dup_thresh_nd_cl, '-o', label='nd_cl')
ax.plot(thresholds, dup_thresh_cl_nd, '-o', label='cl_nd')
ax.set_title('Duplicate Count vs Similarity Threshold')
ax.set_xlabel('Threshold')
ax.set_ylabel('Duplicate Count')
ax.grid(True)
ax.legend()

# Bottom-left: Duplicate Count vs Number of Permutations
ax = axs[1, 0]
ax.plot(permutations, dup_perm_nd_cl, '-o', label='nd_cl')
ax.plot(permutations, dup_perm_cl_nd, '-o', label='cl_nd')
ax.set_title('Duplicate Count vs Number of Permutations')
ax.set_xlabel('Number of Permutations')
ax.set_ylabel('Duplicate Count')
ax.grid(True)
ax.legend()

# Bottom-middle: CL Inference Time vs Dataset Size
ax = axs[1, 1]
ax.plot(dataset_sizes, cl_inf_nd_cl, '-o', label='nd_cl')
ax.plot(dataset_sizes, cl_inf_cl_nd, '-o', label='cl_nd')
ax.set_title('CL Inference Time vs Dataset Size')
ax.set_xlabel('Dataset Size (GB)')
ax.set_ylabel('Time (seconds)')
ax.grid(True)
ax.legend()

# Bottom-right: Total Execution Time vs Dataset Size
ax = axs[1, 2]
ax.plot(dataset_sizes, total_time_nd_cl, '-o', label='nd_cl')
ax.plot(dataset_sizes, total_time_cl_nd, '-o', label='cl_nd')
ax.set_title('Total Execution Time vs Dataset Size')
ax.set_xlabel('Dataset Size (GB)')
ax.set_ylabel('Time (seconds)')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
