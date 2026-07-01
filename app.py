from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_model.pkl"
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

app = Flask(__name__)

# Load the saved model pipeline
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Warning: Model file not found at {MODEL_PATH}. Please run train_model.py first.")

@app.route("/diagrams/<path:filename>")
def serve_diagrams(filename):
    return send_from_directory(BASE_DIR / "diagrams", filename)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    # Load model name if available
    model_name_path = BASE_DIR / "models" / "model_name.pkl"
    model_name = "Random Forest Classifier"
    if model_name_path.exists():
        model_name = joblib.load(model_name_path)
    return render_template("about.html", model_name=model_name)

@app.route("/findyourcrop")
def findyourcrop():
    return render_template("findyourcrop.html", prediction_text=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "findyourcrop.html",
            prediction_text="Error: Model is not loaded. Contact administrator.",
            submitted_values=request.form
        ), 500
        
    try:
        # Retrieve and parse numeric inputs
        values = {
            "N": float(request.form["N"]),
            "P": float(request.form["P"]),
            "K": float(request.form["K"]),
            "temperature": float(request.form["temperature"]),
            "humidity": float(request.form["humidity"]),
            "ph": float(request.form["ph"]),
            "rainfall": float(request.form["rainfall"]),
        }
        
        # Validations: Reject negative values, check pH bounds, relative humidity limits
        if any(value < 0 for value in values.values()):
            raise ValueError("Negative values are not allowed.")
        if values["ph"] > 14.0:
            raise ValueError("Soil pH cannot exceed 14.0.")
        if values["humidity"] > 100.0:
            raise ValueError("Relative humidity cannot exceed 100.0%.")
            
        # Create input DataFrame with exact feature order
        input_frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        
        # Run prediction
        predicted_crop = model.predict(input_frame)[0]
        
        # Calculate confidence score if supported by model
        confidence_text = ""
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(input_frame)[0]
                # Find index of predicted class
                class_index = list(model.classes_).index(predicted_crop)
                confidence_score = probabilities[class_index] * 100
                confidence_text = f" with {confidence_score:.2f}% confidence"
            except Exception as e:
                print("Error calculating confidence:", e)
                
        # Farmer friendly message
        crop_title = predicted_crop.title()
        result_message = f"Based on your soil and climate parameters, the most suitable crop to cultivate is {crop_title}{confidence_text}."
        
        return render_template(
            "findyourcrop.html",
            prediction_text=result_message,
            predicted_crop_name=crop_title,
            submitted_values=values,
            success=True
        )
        
    except (KeyError, TypeError, ValueError) as e:
        error_msg = str(e) if str(e) else "Please enter valid numeric values in all fields."
        if "could not convert string to float" in error_msg:
            error_msg = "Please enter valid numbers in all fields."
        return render_template(
            "findyourcrop.html",
            prediction_text=f"Validation Error: {error_msg}",
            submitted_values=request.form,
            success=False
        ), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
