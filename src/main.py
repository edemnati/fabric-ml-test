"""
main.py
Entry point for Spark Job Definition.

This script initializes the Spark session and executes the full E2E ML pipeline.
Can be run as a Fabric Spark Job Definition or standalone for testing.

Usage in Fabric Spark Job Definition:
    - Main file: main.py
    - Language: PySpark
    - Files: Use "Upload local files" option to upload all src/*.py files
    - When using "Upload local files", all Python files are placed in __pyfiles__/
      directory, so imports work directly without path manipulation

Command-line arguments (optional):
    --model-version: Override model version (default: v1)
    --forecast-horizon: Override forecast horizon (default: 5)
    --raw-table-name: Override raw Delta table name
    --clean-table-name: Override cleansed Delta table name
    --daily-features-table: Override daily features Delta table name
    --registry-table: Override model registry Delta table name
    --predictions-table: Override predictions Delta table name
    --scoring-log-table: Override scoring log Delta table name
"""
import sys
import os
import argparse
from pyspark.sql import SparkSession

# Import pipeline modules
# Note: When using "Upload local files" in Spark Job Definition,
# all .py files are in the same directory (__pyfiles__), so imports work directly
from config import PipelineConfig
from pipeline import E2EMLPipeline


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="E2E ML Pipeline - Spark Job Definition")
    
    # Override configuration parameters via CLI
    parser.add_argument("--model-version", type=str, default="v1", help="Model version identifier")
    parser.add_argument("--forecast-horizon", type=int, default=5, help="Number of periods to forecast")
    parser.add_argument("--raw-parquet-path", type=str, default="Files/green_tripdata_2022-08.parquet", 
                        help="Path to raw parquet data")
    parser.add_argument("--min-aic-improvement", type=float, default=1.0, 
                        help="Minimum AIC improvement to register model")

    # Delta table overrides (align Spark Job with notebook naming)
    parser.add_argument("--raw-table-name", type=str, default="job_e2e_demo_green_tripdata_2022_08",
                        help="Delta table name for raw ingested data")
    parser.add_argument("--clean-table-name", type=str, default="job_e2e_demo_green_tripdata_2022_08_cleansed",
                        help="Delta table name for cleansed data")
    parser.add_argument("--daily-features-table", type=str, default="job_e2e_demo_average_fare_per_day",
                        help="Delta table name for daily engineered features")
    parser.add_argument("--registry-table", type=str, default="job_e2e_demo_time_series_model_registry",
                        help="Delta table name for model registry")
    parser.add_argument("--predictions-table", type=str, default="job_e2e_demo_average_fare_forecast",
                        help="Delta table name for forecast outputs")
    parser.add_argument("--scoring-log-table", type=str, default="job_e2e_demo_scoring_runs",
                        help="Delta table name for scoring audit logs")
    
    # Pipeline control flags
    parser.add_argument("--skip-inference", action="store_true", help="Skip batch inference step")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only, don't run pipeline")
    
    return parser.parse_args()


def get_or_create_spark():
    """
    Get existing Spark session or create new one.
    In Fabric Spark Job Definition, spark session is pre-created.
    """
    try:
        # Try to get the pre-existing spark session (Fabric environment)
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active Spark session found")
        print("Using existing Spark session (Fabric environment)")
        return spark
    except Exception as e:
        print(f"Warning: Could not get active session: {e}")
        print("Creating new Spark session (standalone mode)")
        # Fallback for local testing
        spark = SparkSession.builder \
            .appName("E2E-ML-Pipeline") \
            .getOrCreate()
        return spark


def main():
    """Main entry point for the pipeline."""
    print("=" * 70)
    print("E2E ML PIPELINE - SPARK JOB DEFINITION")
    print("=" * 70)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    print(f"Arguments received:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Initialize configuration with overrides
    config = PipelineConfig(
        model_version=args.model_version,
        forecast_horizon=args.forecast_horizon,
        raw_parquet_path=args.raw_parquet_path,
        min_aic_improvement=args.min_aic_improvement,
        raw_table_name=args.raw_table_name,
        clean_table_name=args.clean_table_name,
        daily_features_table=args.daily_features_table,
        registry_table=args.registry_table,
        predictions_table=args.predictions_table,
        scoring_log_table=args.scoring_log_table,
    )
    
    print(f"\nPipeline Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Get Spark session early.
    # Fabric jobs expect Spark context initialization even if we exit early.
    try:
        spark = get_or_create_spark()
        print(f"Spark Version: {spark.version}")
        print(f"Spark App Name: {spark.sparkContext.appName}")
    except Exception as e:
        print(f"ERROR: Failed to initialize Spark session: {e}")
        return 1

    # Dry run mode - validate config and exit
    if args.dry_run:
        print("\n[DRY RUN MODE] Configuration validated successfully. Exiting.")
        return 0
    
    # Create and run pipeline
    try:
        pipeline = E2EMLPipeline(spark=spark, config=config)
        
        if args.skip_inference:
            print("\n[INFO] Running pipeline WITHOUT batch inference (--skip-inference flag set)")
            pipeline.ingest_data()
            pipeline.clean_data()
            pipeline.engineer_features()
            pipeline.train_models()
            pipeline.register_model()
        else:
            # Run full pipeline
            pipeline.run_full_pipeline()
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ PIPELINE EXECUTION FAILED")
        print("=" * 70)
        print(f"Error: {repr(e)}")
        
        import traceback
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    """
    Entry point when script is executed.
    
    In Fabric Spark Job Definition:
        - This script is executed as main file
        - Spark session is pre-created
        - Exit code is captured for monitoring
    
    For local testing:
        python main.py --dry-run
        python main.py --model-version v2 --forecast-horizon 7
    """
    exit_code = main()
    sys.exit(exit_code)
