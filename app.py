from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("svm_model.joblib")

# Define feature names (Ensure it matches the training features)
feature_names = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 
    'ca', 'cp_1', 'cp_2', 'cp_3', 'restecg_1', 'restecg_2', 'slope_1',
    'thal_1', 'thal_2'
]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get user input from the form
            age = float(request.form["age"])
            sex = int(request.form["sex"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = int(request.form["fbs"])
            thalach = float(request.form["thalach"])
            exang = int(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            ca = int(request.form["ca"])

            # One-hot encoding for categorical features
            cp = int(request.form["cp"])
            cp_1, cp_2, cp_3 = (1 if cp == 1 else 0), (1 if cp == 2 else 0), (1 if cp == 3 else 0)

            restecg = int(request.form["restecg"])
            restecg_1, restecg_2 = (1 if restecg == 2 else 0), (1 if restecg == 3 else 0)

            slope = int(request.form["slope"])
            slope_1 = 1 if slope == 1 else 0  # If slope = 1 (Upsloping), else 0

            thal = int(request.form["thal"])
            thal_1, thal_2 = (1 if thal == 2 else 0), (1 if thal == 3 else 0)

            # Create an input array for prediction
            input_data = np.array([[age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, ca, 
                                    cp_1, cp_2, cp_3, restecg_1, restecg_2, slope_1, thal_1, thal_2]])

            # Make a prediction
            prediction = model.predict(input_data)[0]

            # Map the prediction to a user-friendly message
            result = "Disease Detected" if prediction == 1 else "No Disease Detected"

            return render_template("result.html", result=result)

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
