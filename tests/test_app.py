import sys
from pathlib import Path

# Add project root to path to import app correctly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app import app

def test_home_page_loads():
    """
    Asserts the home page loads successfully (200 OK) and contains OptiCrop.
    """
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"OptiCrop" in response.data

def test_about_page_loads():
    """
    Asserts the about page loads successfully.
    """
    client = app.test_client()
    response = client.get("/about")
    assert response.status_code == 200
    assert b"Smart Agricultural" in response.data

def test_findyourcrop_page_loads():
    """
    Asserts the input form page loads successfully.
    """
    client = app.test_client()
    response = client.get("/findyourcrop")
    assert response.status_code == 200
    assert b"Crop Recommendation Form" in response.data or b"CROP RECOMMENDATION FORM" in response.data

def test_prediction_requires_all_values():
    """
    Asserts prediction fails (400) with empty POST data.
    """
    client = app.test_client()
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert b"Validation Error" in response.data

def test_prediction_validation_negative_values():
    """
    Asserts validation blocks negative NPK/climate values.
    """
    client = app.test_client()
    bad_data = {
        "N": -10, "P": 40, "K": 40,
        "temperature": 25.0, "humidity": 80.0,
        "ph": 6.5, "rainfall": 100.0
    }
    response = client.post("/predict", data=bad_data)
    assert response.status_code == 400
    assert b"Negative values are not allowed" in response.data

def test_prediction_validation_invalid_ph():
    """
    Asserts validation blocks pH values exceeding 14.0.
    """
    client = app.test_client()
    bad_data = {
        "N": 50, "P": 40, "K": 40,
        "temperature": 25.0, "humidity": 80.0,
        "ph": 15.2, "rainfall": 100.0
    }
    response = client.post("/predict", data=bad_data)
    assert response.status_code == 400
    assert b"Soil pH cannot exceed 14.0" in response.data

def test_prediction_valid_input():
    """
    Asserts a correct crop prediction returns 200 and matches the expected crop.
    """
    client = app.test_client()
    valid_data = {
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.87, "humidity": 82.00,
        "ph": 6.50, "rainfall": 202.93
    }
    response = client.post("/predict", data=valid_data)
    assert response.status_code == 200
    assert b"most suitable crop to cultivate is Rice" in response.data or b"Rice" in response.data
