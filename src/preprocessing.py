import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

def calculate_iqr_limits(df, columns=FEATURE_COLUMNS):
    """
    Calculates and returns IQR boundaries and potential outlier counts.
    """
    outlier_info = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_info[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers_count": int(outliers_count)
        }
    return outlier_info

def extract_seasonal_groups(df):
    """
    Categorizes crops into simple seasonal groups based on rules:
    - Summer: temperature > 30 and humidity > 50
    - Winter: temperature < 20 and humidity > 30
    - Rainy: rainfall > 200 and humidity > 50
    """
    summer_crops = df[(df["temperature"] > 30) & (df["humidity"] > 50)]["label"].unique()
    winter_crops = df[(df["temperature"] < 20) & (df["humidity"] > 30)]["label"].unique()
    rainy_crops = df[(df["rainfall"] > 200) & (df["humidity"] > 50)]["label"].unique()
    
    return {
        "summer": sorted(list(summer_crops)),
        "winter": sorted(list(winter_crops)),
        "rainy": sorted(list(rainy_crops))
    }

def split_dataset(df, test_size=0.20, random_state=42):
    """
    Splits the dataset into 80/20 stratified training and testing sets.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
