from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/Crop_recommendation.csv")
EXPECTED_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]

def load_dataset(data_path=DATA_PATH):
    """
    Loads and validates the Crop Recommendation dataset.
    """
    if not data_path.exists():
        # Fallback to local path relative to project root
        project_root = Path(__file__).resolve().parent.parent
        data_path = project_root / data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
            
    df = pd.read_csv(data_path)
    
    # Validate column names
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
        
    return df

def get_dataset_summary(df):
    """
    Returns a dictionary summarizing dataset stats.
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "data_types": df.dtypes.astype(str).to_dict()
    }
    return summary
