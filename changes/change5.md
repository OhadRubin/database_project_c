Okay, let's create a detailed plan for implementing Change #5: "Log Basic Head-Node Resource Usage" with minimal code modification, leveraging the existing `ResourceMetric` table in `db.py`.

**Goal:** Capture average and peak CPU/Memory usage of the head node (where `run_workflows.py` executes) during a workflow run and store it in the `resource_metrics` table, linked to the corresponding `benchmark_runs` entry.

**Strategy:**

1.  Utilize the existing `ResourceMetric` table and `BenchmarkRun.add_resource_metrics` method in `db.py`. No schema changes are needed.
2.  In `run_workflows.py`, start a background thread *before* the main workflow logic begins.
3.  This thread will periodically sample CPU and Memory usage using the `psutil` library.
4.  The thread will store these samples in memory.
5.  *After* the main workflow logic completes, signal the thread to stop.
6.  Calculate the average and maximum CPU/Memory usage from the collected samples.
7.  After creating the `BenchmarkRun` entry, call the `add_resource_metrics` method to save the calculated resource stats.

**Detailed Plan:**

**1. File: `database_project/src/db.py`**

*   **No Changes Needed.**
*   **Verification:** Confirm the `ResourceMetric` table already has the required columns:
    *   `cpu_percent_avg`: Float (Already exists)
    *   `cpu_percent_max`: Float (Already exists)
    *   `memory_usage_avg_mb`: Float (Already exists)
    *   `memory_usage_max_mb`: Float (Already exists)
*   **Verification:** Confirm the `BenchmarkRun` class has the `add_resource_metrics` method (Lines 155-199), which takes these values and creates/links a `ResourceMetric` record.

**2. File: `database_project/src/run_workflows.py`**

*   **Add Imports:**
    *   **Location:** Around lines 1-25 (Add to existing imports)
    *   **Action:** Add imports for threading, resource monitoring, and statistics calculation.
    *   **Existing:**
        ```python
        import os
        # ... other imports ...
        import time
        import logging
        import argparse
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from ray_minhash import run_nd_step_for_workflow # Returns (ray_dataset, dupe_count, nodes, time)
        from ray_tfidf_vec import run_cl_step_for_workflow
        from ml_collections import config_dict
        import yaml
        import glob
        import ray # Already imported later, consolidate here maybe
        from db import init_db, get_session, BenchmarkRun # Already imported later, consolidate here
        ```
    *   **Add:**
        ```python
        import threading
        import psutil
        import statistics
        from queue import Queue # To safely pass results from thread
        ```

*   **Define Monitoring Thread Function:**
    *   **Location:** Around line 105 (Before the `if __name__ == "__main__":` block)
    *   **Action:** Define the function that will run in the background thread to collect metrics.
    *   **New Code (Conceptual):**
        ```python
        def resource_monitor_thread(stop_event: threading.Event, results_queue: Queue, interval: float = 1.0):
            """
            Monitors CPU and Memory usage periodically and puts results in a queue.
            """
            logger.info("Resource monitoring thread started.")
            cpu_percents = []
            memory_percents = []
            while not stop_event.is_set():
                try:
                    cpu_percent = psutil.cpu_percent()
                    memory_info = psutil.virtual_memory()
                    memory_percent = memory_info.percent

                    cpu_percents.append(cpu_percent)
                    memory_percents.append(memory_percent)
                except psutil.Error as e:
                    logger.warning(f"psutil error during monitoring: {e}")
                except Exception as e:
                     logger.error(f"Unexpected error in monitoring thread: {e}", exc_info=True)

                # Wait for the specified interval or until stop_event is set
                stop_event.wait(interval)

            logger.info(f"Resource monitoring thread stopping. Collected {len(cpu_percents)} samples.")
            # Put results into the queue for the main thread
            results_queue.put({
                "cpu_percents": cpu_percents,
                "memory_percents": memory_percents
            })
        ```

*   **Integrate Monitoring into Main Execution Block:**
    *   **Location:** Inside `if __name__ == "__main__":` (Line 107 onwards)
    *   **Action:** Start the monitor thread before the workflow, stop it after, calculate stats, and save them.

    *   **Modify Around Line 109 (Start Timing):**
        *   **Existing:**
            ```python
            109 |     workflow_start_time = time.time()
            110 |     logger.info(f"Starting workflow: {args.workflow}")
            ```
        *   **Add (After Line 110):** Thread setup
            ```python
            # --- Resource Monitoring Setup ---
            stop_monitoring_event = threading.Event()
            resource_results_queue = Queue()
            monitor_thread = threading.Thread(
                target=resource_monitor_thread,
                args=(stop_monitoring_event, resource_results_queue, 1.0), # Check every 1 second
                daemon=True # Allows main program to exit even if thread is blocked
            )
            logger.info("Starting resource monitoring thread...")
            monitor_thread.start()
            # --- End Resource Monitoring Setup ---
            ```

    *   **Modify Around Line 181 (After Workflow Completion, Before DB Logging):**
        *   **Existing:**
            ```python
            181 |         actual_workflow_time = time.time() - workflow_start_time
            182 |         logger.info(f"Workflow '{args.workflow}' finished. Total wall clock time: {actual_workflow_time:.2f} seconds.")
            ```
        *   **Add (After Line 182):** Stop thread, get results, calculate stats.
            ```python
            # --- Stop Resource Monitoring and Get Results ---
            logger.info("Stopping resource monitoring thread...")
            stop_monitoring_event.set()
            monitor_thread.join(timeout=5) # Wait for thread to finish
            if monitor_thread.is_alive():
                logger.warning("Monitoring thread did not exit cleanly.")

            cpu_avg, cpu_max, mem_avg_mb, mem_max_mb = 0.0, 0.0, 0.0, 0.0
            try:
                collected_metrics = resource_results_queue.get_nowait() # Get results
                cpu_percents = collected_metrics.get("cpu_percents", [])
                memory_percents = collected_metrics.get("memory_percents", [])

                if cpu_percents:
                    cpu_avg = statistics.mean(cpu_percents)
                    cpu_max = max(cpu_percents)
                if memory_percents:
                    mem_avg_percent = statistics.mean(memory_percents)
                    mem_max_percent = max(memory_percents)
                    # Get total physical memory to convert percent to MB
                    total_memory_bytes = psutil.virtual_memory().total
                    total_memory_mb = total_memory_bytes / (1024 * 1024)
                    mem_avg_mb = (mem_avg_percent / 100.0) * total_memory_mb
                    mem_max_mb = (mem_max_percent / 100.0) * total_memory_mb
                logger.info(f"Resource Stats: CPU Avg={cpu_avg:.2f}%, CPU Max={cpu_max:.2f}%, Mem Avg={mem_avg_mb:.2f}MB, Mem Max={mem_max_mb:.2f}MB")

            except queue.Empty:
                logger.warning("No resource metrics collected from monitoring thread.")
            except Exception as e:
                logger.error(f"Error processing resource metrics: {e}", exc_info=True)
            # --- End Resource Monitoring ---
            ```

    *   **Modify Around Line 194 (Database Logging):**
        *   **Existing:**
            ```python
            194 |         benchmark_run = BenchmarkRun.create_from_args(
            195 |             session=session,
            196 |             args=args,
            197 |             duplicate_count=total_duplicate_count, # Meaning depends on workflow
            198 |             record_count=final_record_count,       # Final count after all steps
            199 |             execution_time=actual_workflow_time,   # Total wall clock time
            200 |             implementation=args.workflow,          # Use workflow name as implementation
            201 |             num_nodes=num_nodes_used,              # Max nodes used during the workflow
            202 |             notes=benchmark_notes,
            203 |             limit_files=args.limit_files,          # Log the limit used
            204 |             total_size_gb=0            # Log calculated size
            205 |         )
            206 |         logger.info(f"Benchmark data saved with ID: {benchmark_run.id}")
            ```
        *   **Modify/Add (After Line 206):** Get the created run object (or its ID) and add the metrics. The `create_from_args` method likely needs to return the created object or ID. Let's assume it returns the object.
            ```python
            # (Code from 194-206 remains the same)
            206 |         logger.info(f"Benchmark data saved with ID: {benchmark_run.id}")
            # --- Add Resource Metrics to DB ---
            207 |         if benchmark_run and benchmark_run.id is not None:
            208 |             try:
            209 |                 # Re-fetch the object within the session if necessary, or use the returned one
            210 |                 # session = get_session(engine) # May need new session if previous was closed
            211 |                 # fresh_benchmark_run = session.query(BenchmarkRun).get(benchmark_run.id)
            212 |                 # if fresh_benchmark_run:
            213 |                 # Use the object returned by create_from_args directly if session is still active
            214 |                 if cpu_percents or memory_percents: # Only add if we collected data
            215 |                      benchmark_run.add_resource_metrics(
            216 |                          session=session, # Pass session explicitly if needed by the method
            217 |                          cpu_percent_avg=cpu_avg,
            218 |                          cpu_percent_max=cpu_max,
            219 |                          memory_usage_avg_mb=mem_avg_mb,
            220 |                          memory_usage_max_mb=mem_max_mb
            221 |                          # Add other fields like network/disk if desired and collected
            222 |                      )
            223 |                      logger.info(f"Resource metrics added to BenchmarkRun ID: {benchmark_run.id}")
            224 |                 else:
            225 |                     logger.info("Skipping resource metrics logging as no data was collected.")
            226 |                 # else:
            227 |                 #     logger.error(f"Failed to retrieve BenchmarkRun ID {benchmark_run.id} to add resource metrics.")
            228 |             except Exception as e:
            229 |                 logger.error(f"Failed to add resource metrics to DB: {e}", exc_info=True)
            230 |         else:
            231 |            logger.error("Failed to create BenchmarkRun or get ID, cannot add resource metrics.")
            # --- End Add Resource Metrics ---
            ```
            *Self-correction:* The `add_resource_metrics` method in `db.py` uses `object_session(self)` (line 185), so passing the session might not be necessary if the `benchmark_run` object is still attached to the session from `create_from_args`. However, explicitly ensuring the object is session-bound or passing the session can be safer. Let's remove the explicit `session=` argument from the call for minimal change, assuming `create_from_args` leaves the object attached.

            *Refined Code for Adding Metrics:*
            ```python
            # (Code from 194-206 remains the same)
            206 |         logger.info(f"Benchmark data saved with ID: {benchmark_run.id}")
            # --- Add Resource Metrics to DB ---
            207 |         if benchmark_run and benchmark_run.id is not None:
            208 |             try:
            209 |                 if cpu_percents or memory_percents: # Only add if we collected data
            210 |                      # Ensure the object is associated with the session if necessary
            211 |                      # Depending on how create_from_args manages the session, may need merge/refresh
            212 |                      # Assuming benchmark_run is still valid within the session:
            213 |                      benchmark_run.add_resource_metrics(
            214 |                          cpu_percent_avg=cpu_avg,
            215 |                          cpu_percent_max=cpu_max,
            216 |                          memory_usage_avg_mb=mem_avg_mb,
            217 |                          memory_usage_max_mb=mem_max_mb
            218 |                      )
            219 |                      session.commit() # Commit the added resource metric
            220 |                      logger.info(f"Resource metrics added to BenchmarkRun ID: {benchmark_run.id}")
            221 |                 else:
            222 |                     logger.info("Skipping resource metrics logging as no data was collected.")
            223 |             except Exception as e:
            224 |                 logger.error(f"Failed to add resource metrics to DB: {e}", exc_info=True)
            225 |                 # Optionally rollback session changes if error occurs
            226 |                 try:
            227 |                    session.rollback()
            228 |                 except Exception as rb_e:
            229 |                    logger.error(f"Rollback failed: {rb_e}")
            230 |         else:
            231 |            logger.error("Failed to create BenchmarkRun or get ID, cannot add resource metrics.")
            # --- End Add Resource Metrics ---
            ```
            *Note:* Added `session.commit()` after calling `add_resource_metrics` as the method itself doesn't seem to commit in `db.py` (only `create_from_args` does).

This plan details the necessary additions and modifications primarily within `run_workflows.py` to implement resource monitoring using a thread and leveraging the existing database structure defined in `db.py`. It requires adding `psutil` and standard library imports, defining a monitoring thread target function, managing the thread lifecycle around the main workflow execution, calculating statistics, and adding the results to the database via the existing `add_resource_metrics` method.