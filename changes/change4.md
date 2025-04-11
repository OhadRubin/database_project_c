Okay, let's create a detailed plan for implementing **Change #4: Log Full Configuration Details (Args + YAML)** from `changes.md`.

The goal is to capture the complete command-line arguments (`args`) and the loaded YAML configuration (`cfg`) used for a specific run, serialize them into a single JSON string, and store this string in a new database column for maximum reproducibility. We will modify `db.py` to add the column and update the logging function, and `run_workflows.py` to prepare and pass this data.

---

**Plan for Change #4: Log Full Configuration Details (Args + YAML)**

**Summary:**

1.  **Database Schema Modification (`db.py`):** Add a new `Text` column named `config_details_json` to the `BenchmarkRun` table model.
2.  **Database Logging Function Modification (`db.py`):** Update the `BenchmarkRun.create_from_args` class method to accept the JSON configuration string as a new argument and save it to the new `config_details_json` column.
3.  **Orchestration Script Modification (`run_workflows.py`):**
    *   Import the `json` library.
    *   Combine the `args` namespace (converted to a dictionary) and the `cfg` ConfigDict (converted to a dictionary) into a single Python dictionary.
    *   Serialize this combined dictionary into a JSON string.
    *   Pass this JSON string as the new argument to the modified `BenchmarkRun.create_from_args` method.

---

**Detailed Steps:**

**Step 1: Modify Database Schema (`db.py`)**

*   **File:** `database_project/src/db.py`
*   **Goal:** Add the new column to the `BenchmarkRun` SQLAlchemy model.
*   **Change 1.1:**
    *   **Line Number:** Around line 28 (after `total_size_gb` or alongside other text fields like `notes`).
    *   **Existing Code Context:**
        ```python
        26 |     limit_files = Column(Integer, nullable=True)
        27 |     total_size_gb = Column(Float, nullable=True)
        28 |     
        29 |     # Relationships
        30 |     resource_metrics = relationship("ResourceMetric", back_populates="benchmark_run", cascade="all, delete-orphan")
        31 |     accuracy_metrics = relationship("AccuracyMetric", back_populates="benchmark_run", cascade="all, delete-orphan")
        ```
    *   **Modification Description:** Add a new line defining the `config_details_json` column. Use `Text` type for potentially long JSON strings. Make it `nullable=True` for backward compatibility with older database entries if necessary, although new runs should populate it.

*   **Change 1.2:**
    *   **Line Number:** Within the `create_from_args` method signature, around line 103.
    *   **Existing Code Context:**
        ```python
        101 |     @classmethod
        102 |     def create_from_args(cls, session, args, duplicate_count, record_count, execution_time, 
        103 |                        num_nodes=1, notes=None, implementation="pyspark", limit_files=None, total_size_gb=None):
        ```
    *   **Modification Description:** Add a new parameter `config_details_json=None` to the method signature to accept the serialized JSON string.

*   **Change 1.3:**
    *   **Line Number:** Within the `create_from_args` method, inside the `cls(...)` instantiation, around line 150.
    *   **Existing Code Context:**
        ```python
        148 |             limit_files=limit_files if limit_files is not None else args.limit_files,
        149 |             total_size_gb=total_size_gb
        150 |         )
        ```
    *   **Modification Description:** Add the `config_details_json` field to the `BenchmarkRun` object instantiation, assigning the value passed in the new method parameter.

**Step 2: Modify Orchestration Script (`run_workflows.py`)**

*   **File:** `database_project/src/run_workflows.py`
*   **Goal:** Prepare the combined configuration dictionary, serialize it to JSON, and pass it to the database logging function.
*   **Change 2.1:**
    *   **Line Number:** Around line 18 (near other imports).
    *   **Existing Code Context:**
        ```python
        17 | from scipy.integrate import quad as integrate
        18 | import glob
        19 | import time
        ```
    *   **Modification Description:** Add an import statement for the `json` library: `import json`.

*   **Change 2.2:**
    *   **Line Number:** Around line 193 (before the call to `BenchmarkRun.create_from_args`).
    *   **Existing Code Context:**
        ```python
        190 |         
        191 |         engine = init_db()
        192 |         session = get_session(engine)
        193 |         
        194 |         benchmark_run = BenchmarkRun.create_from_args(
        195 |             session=session,
        ```
    *   **Modification Description:** Insert code here to prepare the full configuration JSON. This involves converting the `args` Namespace to a dictionary using `vars(args)`, converting the `cfg` ConfigDict to a dictionary (assuming it behaves like one or has a `.to_dict()` method), combining them into a single dictionary, and then serializing this dictionary to a JSON string using `json.dumps`. Ensure to handle potential non-serializable items gracefully if they exist in `args` or `cfg`, although standard argparse args and YAML-loaded configs are usually fine. Store the resulting JSON string in a variable (e.g., `full_config_json`).

*   **Change 2.3:**
    *   **Line Number:** Within the `BenchmarkRun.create_from_args` call, around line 205 (after `total_size_gb`).
    *   **Existing Code Context:**
        ```python
        202 |             notes=benchmark_notes,
        203 |             limit_files=args.limit_files,          # Log the limit used
        204 |             total_size_gb=0            # Log calculated size
        205 |         )
        ```
    *   **Modification Description:** Add the new keyword argument `config_details_json` to the function call, passing the JSON string variable created in the previous step (e.g., `config_details_json=full_config_json`).

---

**Summary of Changes with Line Numbers:**

1.  **`database_project/src/db.py`**
    *   Around L28: Add `config_details_json = Column(Text, nullable=True)`
    *   Around L103: Add `, config_details_json=None` to `create_from_args` signature.
    *   Around L150: Add `config_details_json=config_details_json,` inside the `cls(...)` call.
2.  **`database_project/src/run_workflows.py`**
    *   Around L18: Add `import json`
    *   Around L193: Add code to convert `args` and `cfg` to dicts, combine them, and use `json.dumps` to create `full_config_json`.
    *   Around L205: Add `config_details_json=full_config_json` to the `BenchmarkRun.create_from_args(...)` call.

This plan adds the necessary database column and modifies the logging mechanism to store the full configuration as a JSON string, achieving the goal of Change #4 while minimizing modifications to the core workflow logic. Remember to handle potential errors during JSON serialization if `args` or `cfg` might contain non-standard objects, though this is less likely with the current structure.