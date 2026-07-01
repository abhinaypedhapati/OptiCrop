# Business and Technical Requirements

## 1. Functional Requirements (FR)
- **Input Form**: The application must provide a responsive web form to accept seven agricultural values:
  - Nitrogen (N): Numeric, non-negative range.
  - Phosphorous (P): Numeric, non-negative range.
  - Potassium (K): Numeric, non-negative range.
  - Temperature: Numeric, Celsius scale.
  - Humidity: Numeric, relative humidity percentage (0-100%).
  - Soil pH: Numeric, standard pH range (0-14).
  - Rainfall: Numeric, annual/seasonal millimeters.
- **Form Validation**: Reject negative numbers, missing fields, or empty submissions with clear warnings.
- **Model Inference**: Load the saved `crop_model.pkl` pipeline to run predictions on form submission.
- **Farmer-Friendly UI**: Display the predicted crop name in bold with clear visual success highlights.
- **Model Details**: Display the algorithm name and classification metrics on the About page.

## 2. Non-Functional Requirements (NFR)
- **Performance**: Return prediction results in under 500ms from form submission.
- **Portability**: Deployable on cloud platforms like Render or Heroku with a single command (`gunicorn`).
- **Aesthetics**: Polished agricultural-themed styles, responsive elements, and clean typography.
- **Reliability & Exception Handling**: Catch typing errors or server failures and return clear, non-breaking user-facing messages.
- **Portability**: Code path must be relative to work on any target host OS (Windows, Linux, macOS).
