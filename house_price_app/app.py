from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model and preprocessing info
model_info = joblib.load("model.pkl")
model = model_info['model']
age_mapping = model_info['age_mapping']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Convert Age using the same mapping as training
        age_value = request.form["Age"]
        age_numeric = age_mapping.get(age_value, 1)  # Default to Moderate if not found
        
        data = {
            "Area_sqft": float(request.form["Area_sqft"]),
            "BHK": int(request.form["BHK"]),
            "Location": request.form["Location"],
            "Furnishing": request.form["Furnishing"],
            "Parking": int(request.form["Parking"]),
            "Age": age_numeric  # Use the numeric value
        }

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return jsonify({
            "success": True,
            "prediction": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)