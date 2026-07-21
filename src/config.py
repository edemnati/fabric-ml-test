"""
config.py
Central configuration for the E2E ML pipeline.
All parameters from the notebook PARAMETERS cell consolidated here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    """Configuration for the entire E2E time-series forecasting pipeline."""
    
    # ===== Input Data Paths =====
    raw_parquet_path: str = "Files/green_tripdata_2022-08.parquet"
    lookup_csv_path: str = "Files/taxi+_zone_lookup.csv"
    
    # ===== Delta Table Names =====
    raw_table_name: str = "spark_job_e2e_demo_green_tripdata_2022_08"
    clean_table_name: str = "spark_job_e2e_demo_green_tripdata_2022_08_cleansed"
    daily_features_table: str = "spark_job_e2e_demo_average_fare_per_day"
    registry_table: str = "spark_job_e2e_demo_time_series_model_registry"
    predictions_table: str = "spark_job_e2e_demo_average_fare_forecast"
    scoring_log_table: str = "spark_job_e2e_demo_scoring_runs"
    
    # ===== Model Configuration =====
    model_name: str = "spark_job_e2e_demo_sarimax_average_fare_per_day_with_exog"
    model_version: str = "v1"
    forecast_horizon: int = 5  # number of future steps to forecast
    
    # ===== Time-Series Model Parameters =====
    # Baseline SARIMA (no exogenous)
    baseline_order: tuple = (1, 0, 0)
    baseline_seasonal_order: tuple = (0, 0, 0, 0)
    
    # SARIMAX (with exogenous variables)
    # Will adjust based on data size in pipeline
    sarimax_order: tuple = (1, 1, 1)
    sarimax_seasonal_order: tuple = (1, 1, 1, 7)  # weekly seasonality
    
    # Exogenous feature columns
    exog_cols: List[str] = field(default_factory=lambda: ["avg_trip_distance", "trip_count"])
    
    # ===== Data Quality Thresholds =====
    min_observations: int = 10  # minimum data points for modeling
    min_aic_improvement: float = 1.0  # minimum AIC gain to register model
    
    # ===== MLflow Configuration =====
    mlflow_experiment_name: str = "e2e_demo_average_fare_per_day_sarimax"
    
    # ===== Model Storage =====
    # Will be set dynamically to notebookutils.nbResPath in Fabric
    model_storage_base_path: str = "builtin/models"
    
    # ===== Gradient Boosting Tree Parameters (optional) =====
    gbt_max_depth: int = 5
    gbt_max_iter: int = 50
    gbt_step_size: float = 0.1
    gbt_train_split: float = 0.8
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "raw_parquet_path": self.raw_parquet_path,
            "lookup_csv_path": self.lookup_csv_path,
            "raw_table_name": self.raw_table_name,
            "clean_table_name": self.clean_table_name,
            "daily_features_table": self.daily_features_table,
            "registry_table": self.registry_table,
            "predictions_table": self.predictions_table,
            "scoring_log_table": self.scoring_log_table,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "forecast_horizon": self.forecast_horizon,
            "exog_cols": self.exog_cols,
        }
    
    @classmethod
    def from_args(cls, **kwargs):
        """Create config from command-line arguments or dictionary."""
        return cls(**kwargs)


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
