import os
from pathlib import Path
from PIL import Image

# Source composite image path
composite_path = Path("C:/Users/pedha/.gemini/antigravity-ide/brain/a941cd37-8820-4a61-a122-d2cf9910e499/media__1782933398316.png")

# Target directories
graphs_dir = Path("screenshots/graphs")
diagrams_dir = Path("diagrams")
screenshots_dir = Path("screenshots")
app_dir = Path("screenshots/app")

graphs_dir.mkdir(parents=True, exist_ok=True)
diagrams_dir.mkdir(parents=True, exist_ok=True)
app_dir.mkdir(parents=True, exist_ok=True)

# Load the composite image
img = Image.open(composite_path)

# Coordinates dictionary for cropping (left, upper, right, lower)
crops = {
    "crop_label_distribution.png": (4, 4, 376, 268),
    "multivariate_correlation_heatmap.png": (378, 4, 672, 268),
    "univariate_distributions.png": (676, 4, 1020, 268),
    "outlier_boxplots.png": (4, 274, 337, 475),
    "kmeans_elbow.png": (339, 274, 661, 475),
    "kmeans_clusters_pca.png": (663, 274, 1020, 475),
    "confusion_matrix.png": (4, 481, 314, 678),
    "model_comparison.png": (316, 481, 622, 678),
    "architecture.png": (624, 481, 1020, 678)
}

# Crop and save each image
for filename, box in crops.items():
    cropped = img.crop(box)
    
    if filename == "architecture.png":
        # System architecture diagram has multiple target places
        cropped.save(diagrams_dir / "architecture.png")
        cropped.save(screenshots_dir / "architecture.png")
        cropped.save(app_dir / "architecture.png")
        print(f"Saved system architecture diagram to {diagrams_dir / 'architecture.png'}, {screenshots_dir / 'architecture.png'}, and {app_dir / 'architecture.png'}")
    else:
        # Standard graph images
        cropped.save(graphs_dir / filename)
        print(f"Saved cropped image to {graphs_dir / filename}")

print("\nCropping and replacement completed successfully!")
