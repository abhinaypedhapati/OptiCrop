import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(exist_ok=True)

def create_notebook(filepath, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Created notebook: {filepath}")

# ----------------- Notebook 1: Data Analysis -----------------
cells_01 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# OptiCrop - 01 Data Analysis\n",
            "This notebook performs Exploratory Data Analysis (EDA) on the crop recommendation dataset to understand feature distributions, correlations, outliers, and environmental patterns."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n",
            "import pandas as pd\n",
            "import seaborn as sns\n",
            "\n",
            "DATA_PATH = Path(\"../data/Crop_recommendation.csv\")\n",
            "OUTPUT_DIR = Path(\"../screenshots/graphs\")\n",
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "FEATURE_COLUMNS = [\"N\", \"P\", \"K\", \"temperature\", \"humidity\", \"ph\", \"rainfall\"]\n",
            "EXPECTED_COLUMNS = FEATURE_COLUMNS + [\"label\"]"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## A. Dataset Reading and Understanding"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = pd.read_csv(DATA_PATH)\n",
            "print(\"First five rows:\")\n",
            "display(df.head())\n",
            "print(\"\\nShape:\", df.shape)\n",
            "print(\"\\nColumns:\", df.columns.tolist())\n",
            "print(\"\\nDataset Information:\")\n",
            "df.info()\n",
            "print(\"\\nMissing values count per column:\")\n",
            "print(df.isnull().sum())\n",
            "print(\"\\nDuplicate rows count:\", df.duplicated().sum())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## B. Univariate Analysis\n",
            "Visualize feature distributions and the balance of the target label class."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Feature distributions\n",
            "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n",
            "axes = axes.flatten()\n",
            "for axis, column in zip(axes, FEATURE_COLUMNS):\n",
            "    sns.histplot(df[column], bins=20, kde=True, ax=axis, color=\"#2d7a4a\")\n",
            "    axis.set_title(f\"Distribution of {column}\")\n",
            "axes[-1].axis(\"off\")\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / \"univariate_distributions.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()\n",
            "\n",
            "# Crop label counts\n",
            "plt.figure(figsize=(12, 7))\n",
            "sns.countplot(data=df, y=\"label\", order=df[\"label\"].value_counts().index, palette=\"viridis\")\n",
            "plt.title(\"Crop Label Distribution\")\n",
            "plt.xlabel(\"Count\")\n",
            "plt.ylabel(\"Crop\")\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / \"crop_label_distribution.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## C. Bivariate Analysis\n",
            "Explore interactions between specific environmental features and crops."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Humidity vs Rainfall scatterplot colored by crop\n",
            "plt.figure(figsize=(12, 8))\n",
            "sns.scatterplot(data=df, x=\"humidity\", y=\"rainfall\", hue=\"label\", alpha=0.7, palette=\"tab20\")\n",
            "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)\n",
            "plt.title(\"Humidity vs Rainfall Across Crop Records\")\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / \"bivariate_humidity_rainfall.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()\n",
            "\n",
            "# Temperature vs Crop boxplot\n",
            "plt.figure(figsize=(15, 7))\n",
            "sns.boxplot(data=df, x=\"label\", y=\"temperature\", palette=\"Set3\")\n",
            "plt.xticks(rotation=70)\n",
            "plt.title(\"Temperature Variation Across Crops\")\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / \"bivariate_temperature_by_crop.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()\n",
            "\n",
            "# pH vs Crop boxplot\n",
            "plt.figure(figsize=(15, 7))\n",
            "sns.boxplot(data=df, x=\"label\", y=\"ph\", palette=\"Set2\")\n",
            "plt.xticks(rotation=70)\n",
            "plt.title(\"Soil pH Variation Across Crops\")\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / \"bivariate_ph_by_crop.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## D. Multivariate Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Correlation heatmap\n",
            "plt.figure(figsize=(10, 8))\n",
            "sns.heatmap(df[FEATURE_COLUMNS].corr(), annot=True, fmt=\".2f\", square=True, cmap=\"YlGnBu\")\n",
            "plt.title(\"Correlation Heatmap of Agricultural Features\")\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / \"multivariate_correlation_heatmap.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()\n",
            "\n",
            "# Multivariate pairplot of selected features\n",
            "sample_df = df.sample(min(400, len(df)), random_state=42)\n",
            "sns.pairplot(\n",
            "    sample_df,\n",
            "    vars=[\"N\", \"P\", \"K\", \"temperature\", \"humidity\"],\n",
            "    hue=\"label\",\n",
            "    corner=True,\n",
            "    plot_kws={\"alpha\": 0.5, \"s\": 20}\n",
            ")\n",
            "plt.savefig(OUTPUT_DIR / \"multivariate_pairplot.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## E. Outlier Analysis\n",
            "Calculate Interquartile Range (IQR) for numeric variables. Outliers will be observed but not blindly deleted because extreme weather and soil levels represent realistic situations."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Boxplots of all features\n",
            "plt.figure(figsize=(14, 6))\n",
            "sns.boxplot(data=df[FEATURE_COLUMNS], palette=\"vlag\")\n",
            "plt.xticks(rotation=45)\n",
            "plt.title(\"Outlier Inspection Using Boxplots\")\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / \"outlier_boxplots.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()\n",
            "\n",
            "# Calculate IQR boundaries\n",
            "for column in FEATURE_COLUMNS:\n",
            "    q1 = df[column].quantile(0.25)\n",
            "    q3 = df[column].quantile(0.75)\n",
            "    iqr = q3 - q1\n",
            "    lower = q1 - 1.5 * iqr\n",
            "    upper = q3 + 1.5 * iqr\n",
            "    outliers = ((df[column] < lower) | (df[column] > upper)).sum()\n",
            "    print(f\"{column}: lower={lower:.2f}, upper={upper:.2f}, potential outliers={outliers}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## F. Seasonal Crop Analysis\n",
            "Simple grouping of crops based on climate thresholds."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "summer = df[(df[\"temperature\"] > 30) & (df[\"humidity\"] > 50)][\"label\"].unique()\n",
            "winter = df[(df[\"temperature\"] < 20) & (df[\"humidity\"] > 30)][\"label\"].unique()\n",
            "rainy = df[(df[\"rainfall\"] > 200) & (df[\"humidity\"] > 50)][\"label\"].unique()\n",
            "\n",
            "print(\"Summer crops:\", sorted(summer))\n",
            "print(\"Winter crops:\", sorted(winter))\n",
            "print(\"Rainy crops:\", sorted(rainy))"
        ]
    }
]

# ----------------- Notebook 2: Preprocessing and Model Building -----------------
cells_02 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# OptiCrop - 02 Preprocessing and Model Building\n",
            "This notebook handles preprocessing, stratified split, and compares four classification models to select the best predictor."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import joblib\n",
            "import matplotlib.pyplot as plt\n",
            "import pandas as pd\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.neighbors import KNeighborsClassifier\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.tree import DecisionTreeClassifier\n",
            "\n",
            "DATA_PATH = Path(\"../data/Crop_recommendation.csv\")\n",
            "MODEL_DIR = Path(\"../models\")\n",
            "MODEL_DIR.mkdir(exist_ok=True)\n",
            "\n",
            "FEATURE_COLUMNS = [\"N\", \"P\", \"K\", \"temperature\", \"humidity\", \"ph\", \"rainfall\"]\n",
            "TARGET_COLUMN = \"label\""
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## A. Load Data and Check Validity"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = pd.read_csv(DATA_PATH)\n",
            "df = df.drop_duplicates().copy()\n",
            "assert not df.isnull().any().any(), \"Dataset contains null values!\"\n",
            "print(\"Dataset size after clean:\", df.shape)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## B. Train-Test Split (80/20 Stratified Split)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X = df[FEATURE_COLUMNS]\n",
            "y = df[TARGET_COLUMN]\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y, test_size=0.20, random_state=42, stratify=y\n",
            ")\n",
            "print(\"Train size:\", X_train.shape)\n",
            "print(\"Test size:\", X_test.shape)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## C. Model Definitions and Training"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = {\n",
            "    \"Logistic Regression\": Pipeline([\n",
            "        (\"scaler\", StandardScaler()),\n",
            "        (\"classifier\", LogisticRegression(max_iter=5000, random_state=42)),\n",
            "    ]),\n",
            "    \"KNN\": Pipeline([\n",
            "        (\"scaler\", StandardScaler()),\n",
            "        (\"classifier\", KNeighborsClassifier(n_neighbors=5)),\n",
            "    ]),\n",
            "    \"Decision Tree\": DecisionTreeClassifier(random_state=42),\n",
            "    \"Random Forest\": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),\n",
            "}\n",
            "\n",
            "trained_models = {}\n",
            "results = []\n",
            "\n",
            "for name, model in models.items():\n",
            "    model.fit(X_train, y_train)\n",
            "    pred = model.predict(X_test)\n",
            "    acc = accuracy_score(y_test, pred)\n",
            "    trained_models[name] = model\n",
            "    results.append({\"model\": name, \"accuracy\": acc})\n",
            "    print(f\"{name} Accuracy: {acc:.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## D. Model Performance Comparison"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "results_df = pd.DataFrame(results).sort_values(\"accuracy\", ascending=False)\n",
            "display(results_df)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## E. Evaluate and Save Best Model"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "best_name = results_df.iloc[0][\"model\"]\n",
            "best_model = trained_models[best_name]\n",
            "final_pred = best_model.predict(X_test)\n",
            "\n",
            "print(f\"Selected Best Model: {best_name}\\n\")\n",
            "print(classification_report(y_test, final_pred))\n",
            "\n",
            "# Save confusion matrix\n",
            "labels = sorted(y.unique())\n",
            "fig, ax = plt.subplots(figsize=(14, 12))\n",
            "ConfusionMatrixDisplay.from_predictions(y_test, final_pred, labels=labels, ax=ax, cmap=\"YlGn\", colorbar=False)\n",
            "plt.title(f\"Confusion Matrix - {best_name}\")\n",
            "plt.tight_layout()\n",
            "plt.savefig(\"../screenshots/graphs/confusion_matrix.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()\n",
            "\n",
            "# Dump model files\n",
            "joblib.dump(best_model, MODEL_DIR / \"crop_model.pkl\")\n",
            "joblib.dump(FEATURE_COLUMNS, MODEL_DIR / \"feature_columns.pkl\")\n",
            "joblib.dump(best_name, MODEL_DIR / \"model_name.pkl\")\n",
            "print(\"Model pipelines saved successfully in models/\")"
        ]
    }
]

# ----------------- Notebook 3: K-Means Clustering -----------------
cells_03 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# OptiCrop - 03 K-Means Clustering\n",
            "This notebook implements K-Means clustering for exploratory grouping of similar soil and climate parameters."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n",
            "import pandas as pd\n",
            "from sklearn.cluster import KMeans\n",
            "from sklearn.decomposition import PCA\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "\n",
            "DATA_PATH = Path(\"../data/Crop_recommendation.csv\")\n",
            "OUTPUT_DIR = Path(\"../screenshots/graphs\")\n",
            "\n",
            "FEATURE_COLUMNS = [\"N\", \"P\", \"K\", \"temperature\", \"humidity\", \"ph\", \"rainfall\"]"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## A. Scaling features"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = pd.read_csv(DATA_PATH)\n",
            "X = df[FEATURE_COLUMNS].copy()\n",
            "\n",
            "scaler = StandardScaler()\n",
            "X_scaled = scaler.fit_transform(X)\n",
            "print(\"Features scaled successfully.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## B. Elbow Method to Determine Optimal Clusters"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "wcss = []\n",
            "for k in range(1, 11):\n",
            "    model = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
            "    model.fit(X_scaled)\n",
            "    wcss.append(model.inertia_)\n",
            "\n",
            "plt.figure(figsize=(8, 5))\n",
            "plt.plot(range(1, 11), wcss, marker=\"o\", color=\"#174f34\")\n",
            "plt.title(\"Elbow Method for K-Means\")\n",
            "plt.xlabel(\"Number of clusters (k)\")\n",
            "plt.ylabel(\"WCSS\")\n",
            "plt.grid(True, linestyle=\"--\", alpha=0.6)\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / \"kmeans_elbow.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## C. Fitting K-Means with K = 4"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "SELECTED_K = 4\n",
            "model = KMeans(n_clusters=SELECTED_K, random_state=42, n_init=10)\n",
            "df[\"cluster\"] = model.fit_predict(X_scaled)\n",
            "\n",
            "cluster_crops = df.groupby(\"cluster\")[\"label\"].agg(lambda values: sorted(values.unique()))\n",
            "print(\"Crops observed in each cluster:\\n\")\n",
            "for cid, crops in cluster_crops.items():\n",
            "    print(f\"Cluster {cid}: {crops}\\n\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## D. Visualizing Clusters using 2D PCA projection"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pca = PCA(n_components=2, random_state=42)\n",
            "components = pca.fit_transform(X_scaled)\n",
            "\n",
            "plt.figure(figsize=(8, 6))\n",
            "scatter = plt.scatter(components[:, 0], components[:, 1], c=df[\"cluster\"], cmap=\"viridis\", alpha=0.65)\n",
            "plt.title(\"K-Means Clusters in PCA Space\")\n",
            "plt.xlabel(\"Principal Component 1\")\n",
            "plt.ylabel(\"Principal Component 2\")\n",
            "plt.legend(*scatter.legend_elements(), title=\"Cluster\")\n",
            "plt.grid(True, linestyle=\"--\", alpha=0.4)\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / \"kmeans_clusters_pca.png\", dpi=200, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    }
]

# Write out notebooks
create_notebook(NOTEBOOKS_DIR / "01_data_analysis.ipynb", cells_01)
create_notebook(NOTEBOOKS_DIR / "02_preprocessing_model.ipynb", cells_02)
create_notebook(NOTEBOOKS_DIR / "03_kmeans_clustering.ipynb", cells_03)
print("All notebooks created successfully.")
