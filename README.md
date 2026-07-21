# Fabric Spark Job Deployment


Scope:

1. Deploy or reuse a Fabric Environment from `env/environment.yml`.
2. Create or update a Spark Job Definition and attach local Python files from `src/`.
3. Keep the baseline notebook and local sample data used to create the scripted pipeline.

## Repository Layout

```text
fabric-spark-job-mlops/
|-- data/
|   |-- green_tripdata_2022-08.parquet
|   `-- taxi+_zone_lookup.csv
|-- deploy_to_fabric.py
|-- env/
|   `-- environment.yml
|-- notebooks/
|   `-- Notebook_e2e_ml-test-20260721.ipynb
|-- requirements.txt
`-- src/
    |-- __init__.py
    |-- config.py
    |-- main.py
    `-- pipeline.py
```

## Prerequisites

- Python 3.9+
- Azure CLI authenticated (`az login`) with the correct tenant/subscription context
- Fabric workspace ID and lakehouse ID for the target workspace
- Internal access rights to create/update workspace items and publish environments

Install Python dependency:

```bash
pip install requests
```

## Data and Notebook

- `data/green_tripdata_2022-08.parquet`:
  Local copy of the raw trip dataset used by the pipeline.

- `data/taxi+_zone_lookup.csv`:
  Local lookup file used during ingestion.

- `notebooks/Notebook_e2e_ml-test-20260721.ipynb`:
  Baseline notebook used to create and validate the scripted implementation in `src/`.

Important:

- The deployed Spark job reads data from Fabric Lakehouse paths defined in `src/config.py`:
  - `Files/green_tripdata_2022-08.parquet`
  - `Files/taxi+_zone_lookup.csv`
- The files in `data/` are local repo copies. Upload them to the target Lakehouse `Files/` area before running the job.
- The notebook in `notebooks/` is not used by the deployed pipeline or by `deploy_to_fabric.py`.

## Internal Runbook

### 1. Run deployment

```bash
python deploy_to_fabric.py --workspace-id <WORKSPACE_ID> --lakehouse-id <LAKEHOUSE_ID>
```

### 2. Optional overrides

```bash
python deploy_to_fabric.py --workspace-id <WORKSPACE_ID> --lakehouse-id <LAKEHOUSE_ID> --environment-name ts-forecasting-env --job-name fabric-e2e-demo-ml-pipeline-spark-job --src-dir src
```

### 3. What the script does

1. Gets a Fabric access token from Azure CLI.
2. Creates the environment item, or reuses an existing one with the same name.
3. Uploads `env/environment.yml` and publishes the environment.
4. Creates the Spark Job Definition, or updates an existing one on name conflict.
5. Uploads local files from `src/` into the job definition:
   - `main.py` -> `Main/main.py`
   - Other `.py` files -> `Libs/*.py`
6. Assumes the input data files are already available in the target Lakehouse `Files/` paths.

## Troubleshooting

- `main.py not found`:
  Use `--src-dir` and point to a folder that contains `main.py`.

- `EnvironmentValidationFailed` with multipart/form-data:
  Ensure you are using the current `deploy_to_fabric.py` from this repo.

- `ItemDisplayNameNotAvailableYet` or name conflict:
  Wait a short time and rerun; Fabric can take time to release item names.

## Notes for Team

- This repo intentionally keeps one deployment path: `deploy_to_fabric.py`.
- Keep source files required by deployment in `src/` and environment dependencies in `env/environment.yml`.
- Keep local reference data in `data/` and the baseline notebook in `notebooks/` for traceability.

