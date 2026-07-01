from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

OUTPUT_DIR = Path("screenshots/graphs")

def evaluate_predictions(y_true, y_pred, model_name, output_dir=OUTPUT_DIR):
    """
    Evaluates predictions, prints metrics and saves the confusion matrix plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print(f"=== Model Evaluation: {model_name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Plot Confusion Matrix
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, xticks_rotation=90, cmap='YlGn', colorbar=False)
    
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"Confusion Matrix plot saved to {output_path}")
    
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix_path": str(output_path)
    }
