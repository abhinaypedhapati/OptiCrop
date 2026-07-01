from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/Crop_recommendation.csv")
OUTPUT_DIR = Path("screenshots/graphs")
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS].copy()
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate WCSS for Elbow Method
    wcss = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        wcss.append(model.inertia_)
        
    # Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker="o", color="#174f34")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_elbow.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Run K-Means with selected K = 4
    SELECTED_K = 4
    model = KMeans(n_clusters=SELECTED_K, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X_scaled)
    
    # Print crops in each cluster
    cluster_crops = df.groupby("cluster")["label"].agg(lambda values: sorted(values.unique()))
    print("Crops observed in each cluster:\n")
    for cluster_id, crops in cluster_crops.items():
        print(f"Cluster {cluster_id}: {crops}\n")
        
    # PCA 2D Projection
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(components[:, 0], components[:, 1], c=df["cluster"], cmap="viridis", alpha=0.65)
    plt.title("K-Means Clusters in PCA Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_clusters_pca.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Save dataset with cluster assignment
    output_path = Path("data/crop_recommendation_with_clusters.csv")
    df.to_csv(output_path, index=False)
    print(f"K-Means output saved to {output_path}")

if __name__ == "__main__":
    main()
