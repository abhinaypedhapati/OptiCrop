from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/Crop_recommendation.csv")
OUTPUT_DIR = Path("screenshots/graphs")
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
EXPECTED_COLUMNS = FEATURE_COLUMNS + ["label"]

def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    return df

def print_dataset_summary(df):
    print("First five rows:\n", df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nInfo:")
    df.info()
    print("\nNull values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

def create_univariate_plots(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for axis, column in zip(axes, FEATURE_COLUMNS):
        sns.histplot(df[column], bins=20, kde=True, ax=axis, color="#2d7a4a")
        axis.set_title(f"Distribution of {column}")
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "univariate_distributions.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, y="label", order=df["label"].value_counts().index, palette="viridis")
    plt.title("Crop Label Distribution")
    plt.xlabel("Count")
    plt.ylabel("Crop")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "crop_label_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

def create_bivariate_plots(df):
    plt.figure(figsize=(12, 8))
    # Using a subset of colors or standard palette
    sns.scatterplot(data=df, x="humidity", y="rainfall", hue="label", alpha=0.7, palette="tab20")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.title("Humidity vs Rainfall Across Crop Records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bivariate_humidity_rainfall.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(15, 7))
    sns.boxplot(data=df, x="label", y="temperature", palette="Set3")
    plt.xticks(rotation=70)
    plt.title("Temperature Variation Across Crops")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bivariate_temperature_by_crop.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # pH vs Crop boxplot is also required by Bivariate Analysis requirements in Section 3 of Prompt.
    plt.figure(figsize=(15, 7))
    sns.boxplot(data=df, x="label", y="ph", palette="Set2")
    plt.xticks(rotation=70)
    plt.title("Soil pH Variation Across Crops")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bivariate_ph_by_crop.png", dpi=200, bbox_inches="tight")
    plt.close()

def create_multivariate_plot(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[FEATURE_COLUMNS].corr(), annot=True, fmt=".2f", square=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap of Agricultural Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multivariate_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Optional: pairplot can be slow, so sample records first.
    sample_df = df.sample(min(400, len(df)), random_state=42)
    sns.pairplot(
        sample_df,
        vars=["N", "P", "K", "temperature", "humidity"],
        hue="label",
        corner=True,
        plot_kws={"alpha": 0.5, "s": 20}
    )
    plt.savefig(OUTPUT_DIR / "multivariate_pairplot.png", dpi=200, bbox_inches="tight")
    plt.close()

def inspect_outliers(df):
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df[FEATURE_COLUMNS], palette="vlag")
    plt.xticks(rotation=45)
    plt.title("Outlier Inspection Using Boxplots")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "outlier_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("--- IQR Limits ---")
    for column in FEATURE_COLUMNS:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        possible_outliers = ((df[column] < lower) | (df[column] > upper)).sum()
        print(f"{column}: lower={lower:.2f}, upper={upper:.2f}, possible outliers={possible_outliers}")

def print_seasonal_crops(df):
    summer = df[(df["temperature"] > 30) & (df["humidity"] > 50)]["label"].unique()
    winter = df[(df["temperature"] < 20) & (df["humidity"] > 30)]["label"].unique()
    rainy = df[(df["rainfall"] > 200) & (df["humidity"] > 50)]["label"].unique()
    
    print("\n--- Seasonal Grouping Exploration ---")
    print("Summer crops:", sorted(summer))
    print("Winter crops:", sorted(winter))
    print("Rainy crops:", sorted(rainy))

def main():
    ensure_directories()
    df = load_dataset()
    print_dataset_summary(df)
    create_univariate_plots(df)
    create_bivariate_plots(df)
    create_multivariate_plot(df)
    inspect_outliers(df)
    print_seasonal_crops(df)
    print(f"\nGraphs successfully saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
