# ==========================================================
# SMART LENDER - FLASK APPLICATION
# ==========================================================

# ===========================
# Import Required Libraries
# ===========================

from flask import Flask, render_template, request
import numpy as np
import pickle

# ===========================
# Create Flask Application
# ===========================

app = Flask(__name__)

# ===========================
# Load Trained XGBoost Model
# ===========================

with open("model/xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# ===========================
# Load Standard Scaler
# ===========================

with open("model/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ===========================
# Load Label Encoders
# ===========================

with open("model/encoders.pkl", "rb") as file:
    encoders = pickle.load(file)

# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template("home.html")

@app.route("/predict_page")
def predict_page():

    return render_template("predict.html")

# ==========================================================
# PREDICTION ROUTE
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ==========================================
        # Get User Inputs from HTML Form
        # ==========================================

        gender = request.form["gender"]
        married = request.form["married"]
        dependents = request.form["dependents"]
        education = request.form["education"]
        self_employed = request.form["self_employed"]

        applicant_income = float(request.form["applicant_income"])
        coapplicant_income = float(request.form["coapplicant_income"])
        loan_amount = float(request.form["loan_amount"])
        loan_amount_term = float(request.form["loan_amount_term"])
        credit_history = float(request.form["credit_history"])

        property_area = request.form["property_area"]

        # ==========================================
        # Encode Categorical Inputs
        # ==========================================

        gender = encoders["Gender"].transform([gender])[0]

        married = encoders["Married"].transform([married])[0]

        education = encoders["Education"].transform([education])[0]

        self_employed = encoders["Self_Employed"].transform([self_employed])[0]

        property_area = encoders["Property_Area"].transform([property_area])[0]

        # ==========================================
        # Convert Dependents
        # ==========================================

        if dependents == "3+":
            dependents = 3

        dependents = int(dependents)

        # ==========================================
        # Arrange Features
        # IMPORTANT:
        # Same order used during training
        # ==========================================

        input_data = [[

            gender,

            married,

            dependents,

            education,

            self_employed,

            applicant_income,

            coapplicant_income,

            loan_amount,

            loan_amount_term,

            credit_history,

            property_area

        ]]

        # ==========================================
        # Feature Scaling
        # ==========================================

        input_data = scaler.transform(input_data)

        # ==========================================
        # Model Prediction
        # ==========================================

        prediction = model.predict(input_data)

        # ==========================================
        # Convert Prediction
        # ==========================================

        if prediction[0] == 1:

            result = "Loan Approved ✅"

        else:

            result = "Loan Rejected ❌"

        # ==========================================
        # Send Result to HTML
        # ==========================================

        return render_template(
            "submit.html",
            prediction=result
        )

    except Exception as e:

        return render_template(
            "submit.html",
            prediction="Error : " + str(e)
        )

# ==========================================================
# RUN FLASK APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )