Project story:
We have two operations
1. Deduplication
2. Clustering

- In large scale data management, it is customary to first run near-deduplication (ND) and only then perform clustering (CL).
- Historically, this is probably because not all workflows require clustering
- We are examining the advantages of doing it the other way around.
Our research question is: should one perform ND first and then CL or CL followed by ND.

Towards that end we we take existing clutsering methods and formulate them in a way that is amenable to be used with map-reduce style processing with Ray.

We use an existing Minhash-LSH-based implementation in Spark, and implement our own MapReduce-based implementation of CL.
We run our clustering implementation on 10 nodes in parallel.

