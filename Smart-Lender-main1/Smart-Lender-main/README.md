# Smart Lender - AI Powered Loan Approval Prediction System

## Overview

Smart Lender is a Machine Learning-powered web application that predicts
whether a loan application is likely to be approved or rejected based on
applicant information. The project automates loan screening and provides
real-time predictions through a Flask web application.

The deployed model used in this project is **XGBoost**.

------------------------------------------------------------------------

## Features

-   Real-time loan approval prediction
-   Flask-based web interface
-   Complete ML pipeline
-   Data preprocessing
-   XGBoost model deployment
-   Pickle model serialization

------------------------------------------------------------------------

## Tech Stack

-   Python
-   Flask
-   HTML
-   CSS
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   Matplotlib
-   Seaborn
-   Pickle

------------------------------------------------------------------------

## Working of the Application

The Smart Lender application begins by collecting applicant information through a user-friendly web interface developed using HTML, CSS, and Flask. The user provides essential loan-related details, including gender, marital status, dependents, education, employment status, applicant income, co-applicant income, loan amount, loan term, credit history, and property area. Once the application form is submitted, the Flask backend receives the input data and performs the same preprocessing steps that were applied during model training to ensure consistency between training and prediction.

The preprocessing pipeline starts by converting categorical attributes into numerical values using the saved Label Encoders. The application then applies the previously trained Standard Scaler to normalize the numerical features so that the input data follows the same distribution as the training dataset. This preprocessing step is essential because the machine learning model was trained on standardized data, and using identical transformations during prediction helps maintain accuracy and reliability.

After preprocessing, the transformed feature vector is passed to the trained **XGBoost** classification model, which serves as the prediction engine of the application. The model analyzes the applicant's financial and personal information by identifying patterns learned during the training phase and predicts whether the loan application is likely to be approved or rejected. The prediction is generated within a fraction of a second and returned to the Flask backend.

Finally, the prediction result is displayed on a dedicated results page, where the applicant is informed whether the loan is **Approved** or **Rejected**. This end-to-end workflow enables financial institutions to automate the loan evaluation process, reduce manual effort, improve consistency in credit assessment, accelerate decision-making, and provide a seamless user experience through a web-based platform. The application demonstrates how Machine Learning can be integrated with modern web technologies to deliver an efficient, scalable, and intelligent loan approval prediction system.

------------------------------------------------------------------------

## 🎥 Demo Video

Watch the complete project demonstration here:

🔗 **Demo Video:**  
https://drive.google.com/file/d/1ntAr8gmSkkyPTXc1B3cNrbhuD6P3gvQx/view?usp=sharing

The demo showcases:
- Home Page
- Loan Prediction Form
- Real-time Loan Approval Prediction
- Approved & Rejected Scenarios
- End-to-End Application Workflow

------------------------------------------------------------------------

## Project Structure

``` text
Smart_Lender/
├── app.py
├── dataset/
├── model/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
├── notebooks/
├── templates/
├── static/
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## Machine Learning Pipeline

1.  Dataset Collection
2.  Exploratory Data Analysis
3.  Missing Value Handling
4.  Label Encoding
5.  Outlier Handling
6.  SMOTE
7.  Feature Scaling
8.  Train-Test Split
9.  Model Training
10. Model Evaluation
11. Model Saving
12. Flask Integration
13. Real-Time Prediction

------------------------------------------------------------------------

## Models Evaluated

-   Decision Tree
-   Random Forest
-   KNN
-   **XGBoost (Deployment Model)**

------------------------------------------------------------------------

## Run Locally

``` bash
git clone https://github.com/AmruthaImmidisetti/Smart-Lender.git
cd Smart-Lender
python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open:

http://127.0.0.1:5000

------------------------------------------------------------------------

## Future Enhancements

-   OCR document verification
-   Explainable AI
-   Chatbot support
-   Mobile application
-   Cloud deployment
