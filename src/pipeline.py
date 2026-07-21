"""
pipeline.py
End-to-end ML pipeline orchestration.
Refactored from notebook cells into modular functions.
"""
from __future__ import annotations  # Enables postponed evaluation of type hints

import json
import os
import uuid
import datetime as dt
import tempfile
from typing import Tuple, Optional

import pandas as pd
import statsmodels.api as sm
import joblib
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, hour, dayofweek
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from config import PipelineConfig


class E2EMLPipeline:
    """Orchestrates the complete E2E ML workflow."""
    
    def __init__(self, spark: SparkSession, config: PipelineConfig):
        self.spark = spark
        self.config = config
        self.logger = self._setup_logging()
        
        # Model artifacts
        self.baseline_result = None
        self.sarimax_result = None
        self.y = None
        self.X = None
        
    def _setup_logging(self):
        """Setup basic logging (can be enhanced with Fabric logging)."""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    # ========== STEP 1: DATA INGESTION ==========
    
    def ingest_data(self) -> Tuple[DataFrame, DataFrame]:
        """
        Load raw trip data and lookup data into Delta tables.
        Returns: (trips_raw_df, zones_df)
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: DATA INGESTION")
        self.logger.info("=" * 60)
        
        # 1.1 Load raw trip data
        self.logger.info(f"Loading raw trip data from: {self.config.raw_parquet_path}")
        trips_raw_df = self.spark.read.parquet(self.config.raw_parquet_path)
        
        row_count = trips_raw_df.count()
        self.logger.info(f"Raw trip row count: {row_count:,}")
        
        # 1.2 Persist as Delta table
        self.logger.info(f"Saving to Delta table: {self.config.raw_table_name}")
        trips_raw_df.write.format("delta").mode("overwrite").saveAsTable(self.config.raw_table_name)
        
        # 1.3 Load taxi zone lookup
        self.logger.info(f"Loading lookup data from: {self.config.lookup_csv_path}")
        zones_df = (
            self.spark.read.format("csv")
            .option("header", "true")
            .load(self.config.lookup_csv_path)
        )
        
        self.logger.info(f"Lookup zones loaded: {zones_df.count():,} rows")
        
        return trips_raw_df, zones_df
    
    # ========== STEP 2: DATA WRANGLING ==========
    
    def clean_data(self) -> DataFrame:
        """
        Apply data quality rules and cleansing.
        Returns: trips_clean_df
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: DATA WRANGLING / CLEANING")
        self.logger.info("=" * 60)
        
        # Load raw table
        trips_df = self.spark.read.table(self.config.raw_table_name)
        initial_count = trips_df.count()
        
        # Apply quality filters
        trips_clean_df = trips_df.filter(
            (col("trip_distance") > 0) & (col("fare_amount") > 0)
        )
        
        clean_count = trips_clean_df.count()
        removed_count = initial_count - clean_count
        
        self.logger.info(f"Initial row count: {initial_count:,}")
        self.logger.info(f"Clean row count:   {clean_count:,}")
        self.logger.info(f"Removed rows:      {removed_count:,} ({removed_count/initial_count*100:.2f}%)")
        
        # Normalize and cast columns
        trips_clean_df = (
            trips_clean_df
            .withColumn(
                "store_and_fwd_flag",
                F.when(col("store_and_fwd_flag") == "Y", F.lit(True)).otherwise(F.lit(False))
            )
            .withColumn("lpep_pickup_datetime", col("lpep_pickup_datetime").cast("timestamp"))
            .withColumn("lpep_dropoff_datetime", col("lpep_dropoff_datetime").cast("timestamp"))
        )
        
        # Persist cleansed data
        self.logger.info(f"Saving to Delta table: {self.config.clean_table_name}")
        trips_clean_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            self.config.clean_table_name
        )
        
        return trips_clean_df
    
    # ========== STEP 3: FEATURE ENGINEERING ==========
    
    def engineer_features(self) -> DataFrame:
        """
        Build daily aggregated features for time-series modeling.
        Returns: daily_features_df
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: FEATURE ENGINEERING (DAILY FEATURES)")
        self.logger.info("=" * 60)
        
        trips_clean_df = self.spark.read.table(self.config.clean_table_name)
        
        # Aggregate to daily level
        daily_features_df = (
            trips_clean_df
            .groupBy(F.to_date("lpep_pickup_datetime").alias("pickup_date"))
            .agg(
                F.avg("fare_amount").alias("average_fare"),
                F.avg("trip_distance").alias("avg_trip_distance"),
                F.count("*").alias("trip_count"),
            )
            .orderBy("pickup_date")
        )
        
        feature_count = daily_features_df.count()
        self.logger.info(f"Daily features created: {feature_count} days")
        
        # Persist
        self.logger.info(f"Saving to Delta table: {self.config.daily_features_table}")
        daily_features_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            self.config.daily_features_table
        )
        
        return daily_features_df
    
    # ========== STEP 4: MODEL TRAINING ==========
    
    def train_models(self) -> Tuple[sm.tsa.statespace.sarimax.SARIMAXResultsWrapper, sm.tsa.statespace.sarimax.SARIMAXResultsWrapper]:
        """
        Train baseline SARIMA and SARIMAX with exogenous variables.
        Returns: (baseline_result, sarimax_result)
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: MODEL EXPERIMENTS & TRAINING")
        self.logger.info("=" * 60)
        
        # Load daily features
        daily_features_df = self.spark.read.table(self.config.daily_features_table).orderBy("pickup_date")
        daily_features_pd = daily_features_df.toPandas()
        daily_features_pd["pickup_date"] = pd.to_datetime(daily_features_pd["pickup_date"])
        daily_features_pd.set_index("pickup_date", inplace=True)
        daily_features_pd.sort_index(inplace=True)
        
        # Prepare target and exogenous variables
        y = daily_features_pd["average_fare"].astype(float)
        X = daily_features_pd[self.config.exog_cols].astype(float)
        
        # Drop NaNs
        valid_mask = y.notna()
        for c in self.config.exog_cols:
            valid_mask &= X[c].notna()
        
        y = y[valid_mask]
        X = X[valid_mask]
        
        n_obs = len(y)
        self.logger.info(f"Number of observations after cleaning: {n_obs}")
        
        if n_obs < self.config.min_observations:
            raise ValueError(f"Not enough data points for robust modeling (n={n_obs}, min={self.config.min_observations})")
        
        # Store for later use
        self.y = y
        self.X = X
        
        # 4.1 Baseline SARIMA (no exogenous)
        self.logger.info("Training baseline SARIMA model (no exogenous)...")
        baseline_model = sm.tsa.statespace.SARIMAX(
            y,
            exog=None,
            order=self.config.baseline_order,
            seasonal_order=self.config.baseline_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        baseline_result = baseline_model.fit(disp=False)
        self.logger.info(f"Baseline AIC: {baseline_result.aic:.4f}")
        
        # 4.2 SARIMAX with exogenous
        # Adjust order based on data size
        if n_obs >= 30:
            order = self.config.sarimax_order
            seasonal_order = self.config.sarimax_seasonal_order
        else:
            order = (1, 1, 1)
            seasonal_order = (0, 0, 0, 0)
        
        self.logger.info(f"Training SARIMAX model with exogenous variables...")
        self.logger.info(f"  Order: {order}, Seasonal: {seasonal_order}")
        
        sarimax_model = sm.tsa.statespace.SARIMAX(
            y,
            exog=X,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarimax_result = sarimax_model.fit(disp=False)
        self.logger.info(f"SARIMAX AIC: {sarimax_result.aic:.4f}")
        
        aic_improvement = baseline_result.aic - sarimax_result.aic
        self.logger.info(f"AIC improvement: {aic_improvement:.4f}")
        
        # Store for later use
        self.baseline_result = baseline_result
        self.sarimax_result = sarimax_result
        
        # Log to MLflow (optional, with error handling)
        self._log_to_mlflow(baseline_result, sarimax_result, n_obs, order, seasonal_order)
        
        return baseline_result, sarimax_result
    
    def _log_to_mlflow(self, baseline_result, sarimax_result, n_obs, order, seasonal_order):
        """Log model to MLflow if available."""
        try:
            import mlflow
            import mlflow.statsmodels
            
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                self.logger.info(f"MLflow run_id: {run_id}")
                
                # Log params and metrics
                mlflow.log_param("order", str(order))
                mlflow.log_param("seasonal_order", str(seasonal_order))
                mlflow.log_param("exog_columns", ",".join(self.config.exog_cols))
                mlflow.log_param("n_obs", int(n_obs))
                
                mlflow.log_metric("train_aic_baseline", float(baseline_result.aic))
                mlflow.log_metric("train_aic_sarimax", float(sarimax_result.aic))
                
                # Log model
                artifact_path = "model"
                mlflow.statsmodels.log_model(sarimax_result, artifact_path=artifact_path)
                
                # Register model
                model_uri = f"runs:/{run_id}/{artifact_path}"
                registered_model = mlflow.register_model(model_uri=model_uri, name=self.config.model_name)
                
                self.logger.info(f"Registered MLflow model: models:/{registered_model.name}/{registered_model.version}")
                
        except Exception as e:
            self.logger.warning(f"MLflow logging failed (continuing): {repr(e)}")
    
    # ========== STEP 5: MODEL REGISTRY ==========
    
    def register_model(self) -> Optional[str]:
        """
        Persist model to storage and register in Delta-based registry.
        Returns: model_path
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: MODEL REGISTRY (DELTA-BASED)")
        self.logger.info("=" * 60)
        
        if self.baseline_result is None or self.sarimax_result is None:
            raise RuntimeError("Models not trained yet. Call train_models() first.")
        
        # 5.1 Save model to local storage
        model_path = self._save_model_locally()
        
        # 5.2 Register in Delta table
        aic_gain = self.baseline_result.aic - self.sarimax_result.aic
        self.logger.info(f"AIC improvement vs baseline: {aic_gain:.4f}")
        
        if aic_gain < self.config.min_aic_improvement:
            self.logger.warning(f"AIC improvement too small ({aic_gain:.4f} < {self.config.min_aic_improvement}); skipping registry update.")
            return None
        
        self.logger.info("AIC improvement sufficient; registering model in Delta table...")
        
        metrics = {
            "train_aic_baseline": float(self.baseline_result.aic),
            "train_aic_sarimax": float(self.sarimax_result.aic),
            "aic_gain": float(aic_gain),
            "n_obs": int(len(self.y)),
        }
        
        from pyspark.sql import Row
        
        model_registry_row = Row(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            model_path=model_path,
            mlflow_model_uri="",  # Can be populated if MLflow succeeded
            target_series=self.config.daily_features_table,
            created_utc=pd.Timestamp.utcnow().isoformat(),
            order=str(self.config.sarimax_order),
            seasonal_order=str(self.config.sarimax_seasonal_order),
            n_obs=int(len(self.y)),
            exog_columns=",".join(self.config.exog_cols),
            stage="Staging",
            status="active",
            metrics_json=json.dumps(metrics),
        )
        
        registry_df = self.spark.createDataFrame([model_registry_row])
        registry_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
            self.config.registry_table
        )
        
        self.logger.info(f"Model metadata registered in: {self.config.registry_table}")
        
        return model_path
    
    def _save_model_locally(self) -> str:
        """Save trained model to local storage."""
        # Spark Job Definitions do not have notebook resource mounts.
        # Use a local temp directory that is always available in the driver container.
        base_res_path = tempfile.gettempdir()

        models_dir = os.path.join(base_res_path, self.config.model_storage_base_path)
        os.makedirs(models_dir, exist_ok=True)
        
        model_filename = f"{self.config.model_name}_{self.config.model_version}.joblib"
        model_path = os.path.join(models_dir, model_filename)
        
        joblib.dump(self.sarimax_result, model_path)
        self.logger.info(f"Model saved to: {model_path}")
        
        return model_path
    
    # ========== STEP 6: BATCH INFERENCE ==========
    
    def run_batch_inference(self):
        """
        Load registered model and generate batch forecasts.
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 6: BATCH INFERENCE")
        self.logger.info("=" * 60)
        
        # 6.1 Load latest model from registry
        self.logger.info(f"Loading latest model from registry: {self.config.registry_table}")
        full_registry_df = self.spark.read.table(self.config.registry_table)
        
        latest_model_row = (
            full_registry_df
            .filter(F.col("stage") == "Staging")
            .orderBy(F.col("created_utc").desc())
            .limit(1)
            .collect()
        )
        
        if not latest_model_row:
            self.logger.warning("No model found in registry. Skipping inference.")
            return
        
        latest_model_row = latest_model_row[0]
        loaded_model_path = latest_model_row["model_path"]
        loaded_model_name = latest_model_row["model_name"]
        loaded_model_version = latest_model_row["model_version"]
        
        self.logger.info(f"Loading model '{loaded_model_name}' v{loaded_model_version} from: {loaded_model_path}")
        loaded_model = joblib.load(loaded_model_path)
        
        # 6.2 Prepare exogenous variables for forecast
        daily_features_df = self.spark.read.table(self.config.daily_features_table).orderBy("pickup_date")
        daily_features_pd = daily_features_df.toPandas().sort_values("pickup_date")
        
        future_exog = daily_features_pd[self.config.exog_cols].tail(self.config.forecast_horizon).astype(float)
        
        self.logger.info(f"Using last {self.config.forecast_horizon} rows as future exogenous features")
        
        # 6.3 Generate forecasts
        preds = loaded_model.predict(
            start=loaded_model.nobs,
            end=loaded_model.nobs + len(future_exog) - 1,
            exog=future_exog,
            dynamic=False,
        )
        
        predictions_df = pd.DataFrame({
            "horizon_step": range(1, len(preds) + 1),
            "forecast_average_fare": preds.values,
        })
        
        self.logger.info("Batch forecast results:")
        self.logger.info(predictions_df.to_string())
        
        # 6.4 Persist predictions
        predictions_spark_df = self.spark.createDataFrame(predictions_df)
        predictions_spark_df.write.format("delta").mode("overwrite").saveAsTable(self.config.predictions_table)
        
        self.logger.info(f"Forecasts saved to Delta table: {self.config.predictions_table}")
        
        # 6.5 Log scoring metadata
        self._log_scoring_run(loaded_model_name, loaded_model_version)
    
    def _log_scoring_run(self, model_name, model_version):
        """Log scoring run metadata for auditability."""
        from pyspark.sql import Row
        
        scoring_run_id = str(uuid.uuid4())
        scoring_time = dt.datetime.utcnow().isoformat()
        
        self.logger.info(f"Scoring run ID: {scoring_run_id}")
        self.logger.info(f"Scoring time:   {scoring_time}")
        
        scoring_log_row = Row(
            scoring_run_id=scoring_run_id,
            model_name=model_name,
            model_version=model_version,
            registry_table=self.config.registry_table,
            predictions_table=self.config.predictions_table,
            scoring_time=scoring_time,
            forecast_horizon=int(self.config.forecast_horizon),
        )
        
        self.spark.createDataFrame([scoring_log_row]).write.format("delta").mode("append").option(
            "mergeSchema", "true"
        ).saveAsTable(self.config.scoring_log_table)
        
        self.logger.info(f"Scoring metadata logged to: {self.config.scoring_log_table}")
    
    # ========== FULL PIPELINE ORCHESTRATION ==========
    
    def run_full_pipeline(self):
        """Execute the complete end-to-end pipeline."""
        self.logger.info("=" * 70)
        self.logger.info("STARTING FULL E2E ML PIPELINE")
        self.logger.info("=" * 70)
        self.logger.info(f"Configuration: {self.config.to_dict()}")
        
        try:
            # Execute all steps in sequence
            self.ingest_data()
            self.clean_data()
            self.engineer_features()
            self.train_models()
            self.register_model()
            self.run_batch_inference()
            
            self.logger.info("=" * 70)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {repr(e)}", exc_info=True)
            raise
