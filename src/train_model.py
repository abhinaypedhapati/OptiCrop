from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("data/Crop_recommendation.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("screenshots/graphs")
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

def load_data():
    df = pd.read_csv(DATA_PATH)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop duplicates if they exist
    df = df.drop_duplicates().copy()
    
    # Verify no null values exist before training
    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        raise ValueError("The dataset contains null values. Resolve them before training.")
    return df

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
    }

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("Train/test shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    trained = {}
    scores = []
    
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        score = accuracy_score(y_test, prediction)
        trained[name] = model
        scores.append({"model": name, "accuracy": score})
        print(f"{name}: {score:.4f}")
        
    results = pd.DataFrame(scores).sort_values("accuracy", ascending=False)
    print("\nModel comparison:\n", results)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    
    best_name = results.iloc[0]["model"]
    best_model = trained[best_name]
    final_prediction = best_model.predict(X_test)
    
    print("\nSelected model:", best_name)
    print("Accuracy:", accuracy_score(y_test, final_prediction))
    print("\nClassification report:\n")
    print(classification_report(y_test, final_prediction))
    
    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, final_prediction, labels=labels)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=ax, xticks_rotation=90, colorbar=False, cmap="YlGn"
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Save best model pipeline and supporting metadata
    joblib.dump(best_model, MODEL_DIR / "crop_model.pkl")
    joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_columns.pkl")
    joblib.dump(best_name, MODEL_DIR / "model_name.pkl")
    
    print("\nSaved deployment model to models/crop_model.pkl")

if __name__ == "__main__":
    main()
