# OptiCrop - Smart Agricultural Production Optimization Engine

OptiCrop is an end-to-end Machine Learning web application designed to recommend the most suitable crop for cultivation based on specific soil characteristics and climate parameters.

## Overview
Optimizing crop selection is critical for sustainable, high-yield agriculture. OptiCrop utilizes supervised learning algorithms to match seven input soil and environmental features against historical patterns to predict the ideal crop.

## Problem Statement
Farmers often make cropping decisions based on historical intuition or market trends, leading to poor soil matching, reduced crop yields, and high resource waste. OptiCrop provides data-driven decision support by analyzing soil chemistry (Nitrogen, Phosphorous, Potassium, pH) and microclimate parameters (temperature, humidity, rainfall).

## Key Features
- **Agricultural Data Analysis**: Detailed Exploratory Data Analysis notebooks detailing feature distribution, correlations, and outliers.
- **Unsupervised Grouping**: K-Means clustering to examine groups of similar environmental and nutrient configurations.
- **Supervised Classification**: Training and validation comparison of Logistic Regression, K-Nearest Neighbors, Decision Trees, and Random Forests.
- **Model Persistence**: Pipelines saved via Joblib to ensure standard scaling and feature orders.
- **Responsive Flask Frontend**: A clean, modern green-themed web UI optimized for both desktop and mobile layouts.
- **Robust Input Validation**: Strict validation logic protecting predictions from invalid values.

## Technology Stack
- **Backend & Web framework**: Python, Flask, Jinja2, Gunicorn
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, Joblib
- **Visualization**: Matplotlib, Seaborn
- **Testing**: Pytest

## Dataset Information
The model trains on a 2,200-row crop recommendation dataset with the following schema:
- `N`: Ratio of Nitrogen content in soil
- `P`: Ratio of Phosphorous content in soil
- `K`: Ratio of Potassium content in soil
- `temperature`: Temperature in degree Celsius
- `humidity`: Relative humidity in %
- `ph`: pH value of the soil
- `rainfall`: Rainfall in mm
- `label`: Target crop class (22 distinct crops)

---

## Folder Structure
```
OptiCrop/
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── .gitignore
├── README.md
├── data/
│   └── Crop_recommendation.csv
├── docs/
│   ├── problem_statement.md
│   ├── business_requirements.md
│   ├── literature_survey.md
│   ├── social_business_impact.md
│   └── conclusion.md
├── models/
│   └── crop_model.pkl
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_preprocessing_model.ipynb
│   └── 03_kmeans_clustering.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── about.html
│   └── findyourcrop.html
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
├── screenshots/
│   ├── graphs/
│   └── app/
└── diagrams/
    ├── erd.png
    ├── workflow.png
    └── architecture.png
```

---

## Installation & Local Execution

### 1. Clone the repository and navigate to folder
```bash
git clone <repository-url>
cd OptiCrop
```

### 2. Set up virtual environment
**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Model Training (Generate pkl file)
```bash
python src/train_model.py
```

### 5. Run Flask Web Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## Model Training & Evaluation
We compared four different classifiers. The test accuracy scores on an 80/20 stratified split are:
- **Random Forest**: 99.32% (Selected for deployment)
- **K-Nearest Neighbors**: 97.95%
- **Decision Tree**: 97.95%
- **Logistic Regression**: 97.27%

---

## Diagrams & Screenshots
- **ER Diagram**: Located in [diagrams/erd.png](file:///C:/Users/pedha/.gemini/antigravity-ide/scratch/OptiCrop/diagrams/erd.png) and [screenshots/erd.png](file:///C:/Users/pedha/.gemini/antigravity-ide/scratch/OptiCrop/screenshots/erd.png).
- **Workflow Diagram**: Located in [diagrams/workflow.png](file:///C:/Users/pedha/.gemini/antigravity-ide/scratch/OptiCrop/diagrams/workflow.png) and [screenshots/workflow.png](file:///C:/Users/pedha/.gemini/antigravity-ide/scratch/OptiCrop/screenshots/workflow.png).
- **Architecture Diagram**: Located in [diagrams/architecture.png](file:///C:/Users/pedha/.gemini/antigravity-ide/scratch/OptiCrop/diagrams/architecture.png).

---

## Future Enhancements
- Real-time weather API integration.
- IoT soil sensor integration.
- Fertilizer correction recommendations.
- Yield forecasting.
- Mobile application.

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
