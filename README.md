# Heart Disease Prediction Flask App

This is a web application for predicting heart disease risk using machine learning models. It uses a Flask backend with multiple ML algorithms (Decision Tree, SVM, Random Forest, XGBoost, ANN) to provide predictions based on user input.

## Features
- User-friendly web interface for inputting health data.
- Ensemble prediction from multiple models.
- Age-based recommendations if risk is detected.
- Displays probabilities and individual model results.

## Local Setup
1. Clone the repository:
   ```
   git clone https://github.com/neosaqlain2004/heart-disease-prediction-.git
   cd heart-disease-prediction-
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   python app.py
   ```

4. Open your browser to `http://localhost:5000`.

## Deployment
This app is designed for deployment on platforms like Render. Ensure the entry point is `app.py` and build command is `pip install -r requirements.txt`.

## Requirements
- Flask==2.3.3
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- tensorflow-cpu==2.13.0
- xgboost==1.7.6
- imbalanced-learn==0.11.0

## Models
Pre-trained models are included in the `model/` directory. Retrain using `model/train.py` if needed.
