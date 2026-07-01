# Literature Survey

Based on existing agricultural studies and machine learning projects:

1. **Precision Decision Support**: Modern crop recommendation systems combine soil nutrient information (N, P, K) and climate indicators (temperature, humidity, pH, rainfall) to support complex, data-driven farming decisions.
2. **Supervised Classification**: Supervised classifiers such as Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forests can learn distinct crop-environment boundaries from historical data with high accuracy.
3. **Unsupervised Exploratory Grouping**: Unsupervised approaches such as K-Means clustering help identify clusters of similar agricultural parameters. This is highly useful for exploratory data grouping, though supervised learning remains necessary for precise label predictions.
4. **Data Leakage & Preprocessing**: Preprocessing pipelines (e.g. `StandardScaler` applied inside a Pipeline) ensure features are appropriately scaled for distance-sensitive models (like KNN or Logistic Regression) while preventing data leakage during cross-validation or testing.
5. **Human-Centric Web Tools**: Lightweight web applications (such as Flask combined with clean responsive HTML/CSS UI) bridge the gap between technical ML model assets and non-technical stakeholders (farmers, researchers, policy makers).
