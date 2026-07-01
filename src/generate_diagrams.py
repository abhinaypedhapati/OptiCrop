from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DIAGRAMS_DIR = Path("diagrams")
SCREENSHOTS_DIR = Path("screenshots")
DIAGRAMS_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Set common font styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

def draw_box(ax, x, y, width, height, title, fields, bg_color="#ecf7ed", border_color="#174f34"):
    """
    Draws a box representing a database table or entity.
    """
    # Draw box border
    rect = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.1", 
        linewidth=2, edgecolor=border_color, facecolor=bg_color
    )
    ax.add_patch(rect)
    
    # Title text
    ax.text(x + width/2, y + height - 0.3, title, fontsize=12, fontweight='bold', 
            ha='center', va='center', color="#17231a")
    
    # Title divider line
    ax.plot([x - 0.05, x + width + 0.05], [y + height - 0.5, y + height - 0.5], color=border_color, linewidth=1.5)
    
    # Fields text
    field_text = "\n".join(fields)
    ax.text(x + 0.2, y + height - 0.7, field_text, fontsize=10, 
            ha='left', va='top', color="#17231a", linespacing=1.6)

def draw_arrow(ax, start_x, start_y, end_x, end_y, label=""):
    """
    Draws a directed connection line.
    """
    ax.annotate(
        label, 
        xy=(end_x, end_y), 
        xytext=(start_x, start_y),
        arrowprops=dict(facecolor='#174f34', edgecolor='#174f34', arrowstyle="->", lw=2, shrinkA=5, shrinkB=5),
        fontsize=9, fontweight='bold', color='#2d7a4a', ha='center', va='center'
    )

def generate_erd():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 1. User
    draw_box(ax, 1, 8, 2.5, 2.5, "User", [
        "user_id (PK)",
        "name",
        "email",
        "location"
    ])
    
    # 2. SoilData
    draw_box(ax, 5.5, 7.5, 3.0, 3.5, "SoilData", [
        "soil_id (PK)",
        "user_id (FK)",
        "nitrogen",
        "phosphorous",
        "potassium",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "created_at"
    ])
    
    # 3. Prediction
    draw_box(ax, 10.5, 7.5, 3.0, 3.5, "Prediction", [
        "prediction_id (PK)",
        "soil_id (FK)",
        "crop_id (FK)",
        "model_id (FK)",
        "predicted_at",
        "confidence_score"
    ])
    
    # 4. Crop
    draw_box(ax, 15, 8, 2.5, 2.5, "Crop", [
        "crop_id (PK)",
        "crop_name",
        "season",
        "crop_type"
    ])
    
    # 5. MLModel
    draw_box(ax, 10.5, 4, 3.0, 2.5, "MLModel", [
        "model_id (PK)",
        "model_name",
        "algorithm",
        "accuracy",
        "created_at"
    ])
    
    # 6. Report
    draw_box(ax, 10.5, 0.5, 3.0, 2.2, "Report", [
        "report_id (PK)",
        "prediction_id (FK)",
        "summary",
        "recommendations"
    ])
    
    # Draw connections
    # User to SoilData
    draw_arrow(ax, 3.6, 9.25, 5.4, 9.25, "1 : many")
    # SoilData to Prediction
    draw_arrow(ax, 8.6, 9.25, 10.4, 9.25, "1 : 1")
    # Crop to Prediction
    draw_arrow(ax, 14.9, 9.25, 13.6, 9.25, "1 : many")
    # MLModel to Prediction
    draw_arrow(ax, 12.0, 6.6, 12.0, 7.4, "1 : many")
    # Prediction to Report
    draw_arrow(ax, 12.0, 7.4, 12.0, 2.8, "1 : many")
    
    plt.title("OptiCrop Entity Relationship Diagram (ERD)", fontsize=16, fontweight='bold', color="#174f34", pad=20)
    plt.tight_layout()
    plt.savefig(DIAGRAMS_DIR / "erd.png", dpi=200, bbox_inches="tight")
    plt.savefig(SCREENSHOTS_DIR / "erd.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("ERD Diagram generated successfully.")

def generate_workflow():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    steps = [
        ("Farmer Input", ["N, P, K,", "Temp, Hum,", "pH, Rainfall"]),
        ("Flask Form", ["Interactive web", "page inputs"]),
        ("Input Validation", ["Check numeric,", "Non-negative", "ranges"]),
        ("Saved ML Model", ["Load trained", "Random Forest", "Pipeline"]),
        ("Crop Prediction", ["Generate recommendation", "& confidence"]),
        ("Recommendation\nResult", ["Show farmer-friendly", "card and summary"])
    ]
    
    box_width = 2.0
    box_height = 2.2
    y_pos = 1.0
    
    for i, (title, lines) in enumerate(steps):
        x_pos = 0.5 + i * 2.5
        draw_box(ax, x_pos, y_pos, box_width, box_height, title, lines, bg_color="#eff7ef")
        
        # Connect to next box
        if i < len(steps) - 1:
            draw_arrow(ax, x_pos + box_width + 0.1, y_pos + box_height/2, x_pos + 2.4, y_pos + box_height/2)
            
    plt.title("OptiCrop Farmer-to-Prediction Workflow", fontsize=14, fontweight='bold', color="#174f34", pad=15)
    plt.tight_layout()
    plt.savefig(DIAGRAMS_DIR / "workflow.png", dpi=200, bbox_inches="tight")
    plt.savefig(SCREENSHOTS_DIR / "workflow.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Workflow Diagram generated successfully.")

def generate_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw Architecture boundaries
    # Client Browser
    draw_box(ax, 1, 5, 2.5, 3.5, "Client Browser", [
        "Home Page (/) ",
        "About Page (/about)",
        "Find Crop (/findyourcrop)",
        "Prediction Form",
        "CSS Layout (style.css)",
        "Client JS (main.js)"
    ], bg_color="#ffffff", border_color="#5d6f62")
    
    # Flask Web Backend
    draw_box(ax, 5, 5, 3.0, 3.5, "Flask Web App", [
        "app.py",
        "Routes Handler",
        "Input Validator",
        "Feature Formatter",
        "Jinja2 Renderer",
        "Exception Handler"
    ], bg_color="#eff7ef")
    
    # ML Pipeline
    draw_box(ax, 9, 5, 2.5, 2.5, "Saved ML Model", [
        "crop_model.pkl",
        "StandardScaler",
        "RandomForest Classifier",
        "feature_columns.pkl",
        "model_name.pkl"
    ], bg_color="#ecf7ed")
    
    # Dataset
    draw_box(ax, 5, 1, 3.0, 2.5, "Dataset & Scripts", [
        "Crop_recommendation.csv",
        "data_loader.py",
        "preprocessing.py",
        "train_model.py",
        "evaluate_model.py"
    ], bg_color="#eff7ef")
    
    # Connect diagrams
    # Browser to Backend Request
    draw_arrow(ax, 3.6, 7.5, 4.9, 7.5, "POST /predict")
    # Backend to Browser HTML
    draw_arrow(ax, 4.9, 6.0, 3.6, 6.0, "Render HTML")
    # Backend loads model
    draw_arrow(ax, 8.1, 7.0, 8.9, 7.0, "predict()")
    # Dataset to Backend (for offline training)
    draw_arrow(ax, 6.5, 3.6, 6.5, 4.9, "Offline Train")
    
    plt.title("OptiCrop Technical Architecture Diagram", fontsize=14, fontweight='bold', color="#174f34", pad=15)
    plt.tight_layout()
    plt.savefig(DIAGRAMS_DIR / "architecture.png", dpi=200, bbox_inches="tight")
    plt.savefig(SCREENSHOTS_DIR / "architecture.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Architecture Diagram generated successfully.")

def main():
    generate_erd()
    generate_workflow()
    generate_architecture()
    print("All diagrams generated successfully.")

if __name__ == "__main__":
    main()
