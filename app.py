import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static', template_folder='static')

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

dt = pickle.load(open(os.path.join(model_dir, 'dt.pkl'), 'rb'))
svm = pickle.load(open(os.path.join(model_dir, 'svm.pkl'), 'rb'))
rf = pickle.load(open(os.path.join(model_dir, 'rf.pkl'), 'rb'))
xgb = pickle.load(open(os.path.join(model_dir, 'xgb.pkl'), 'rb'))
ann = load_model(os.path.join(model_dir, 'ann.h5'))
scaler = pickle.load(open(os.path.join(model_dir, 'scaler.pkl'), 'rb'))

# Mappings for categorical
mappings = {
    'Smoking': {'Yes': 1, 'No': 0},
    'AlcoholDrinking': {'Yes': 1, 'No': 0},
    'Stroke': {'Yes': 1, 'No': 0},
    'DiffWalking': {'Yes': 1, 'No': 0},
    'Sex': {'Male': 1, 'Female': 0},
    'AgeCategory': {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80 or older': 12},
    'Race': {'White': 0, 'Black': 1, 'Asian': 2, 'American Indian/Alaskan Native': 3, 'Hispanic': 4, 'Other': 5},
    'Diabetic': {'Yes': 1, 'No': 0, 'No, borderline diabetes': 0, 'Yes (during pregnancy)': 1},
    'PhysicalActivity': {'Yes': 1, 'No': 0},
    'GenHealth': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4},
    'Asthma': {'Yes': 1, 'No': 0},
    'KidneyDisease': {'Yes': 1, 'No': 0},
    'SkinCancer': {'Yes': 1, 'No': 0}
}

# Age-based recommendations for heart disease risks
age_recommendations = {
    '18-24': 'Rare, but lifestyle risks (smoking, obesity, poor diet) may start raising long-term risk; Rheumatic Heart Disease (RHD) possible in developing regions.',
    '25-29': 'Early signs of hypertension, obesity, high cholesterol; beginning of atherosclerosis in some individuals.',
    '30-34': 'Coronary artery plaque buildup may begin; metabolic syndrome (obesity + diabetes + high BP) risk rises.',
    '35-39': 'Ischemic heart disease (IHD) cases start appearing; small but real risk of heart attack (~16.9 cases/100,000).',
    '40-44': 'Risk of coronary heart disease (CHD) increases; early stroke cases linked to hypertension/diabetes; ~1% show elevated 10-year CVD risk.',
    '45-49': 'Significant rise in sudden cardiac death (SCD) (especially men); diabetes/obesity amplify risk; hypertension common.',
    '50-54': 'Moderate-to-high risk of heart attack, CHD, and atrial fibrillation; women’s risk rises post-menopause.',
    '55-59': 'Increasing angina, valve issues, higher incidence of ischemic stroke; risk accelerates with smoking/diabetes.',
    '60-64': 'Heart failure risk grows; more likely to show symptoms of CHD; higher 10-year CVD risk (>18%).',
    '65-69': 'High risk of heart attack, stroke, heart failure; 66%+ have elevated CVD risk; valvular disease often diagnosed.',
    '70-74': 'Arrhythmias (atrial fibrillation) become common; heart attack risk ~7× higher than in 35–44 group.',
    '75-79': 'Advanced CHD, valve disease, heart failure; high hospitalization rates; multi-condition overlap common.',
    '80 or older': 'Very high risk of heart attack, stroke, valve disease, and heart failure; frailty and multiple co-morbidities.'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract features in the same order as training (top 10)
    features_list = [
        float(data['BMI']),
        float(data['PhysicalHealth']),
        float(data['MentalHealth']),
        float(data['SleepTime']),
        mappings['Smoking'][data['Smoking']],
        mappings['AlcoholDrinking'][data['AlcoholDrinking']],
        mappings['Stroke'][data['Stroke']],
        mappings['PhysicalActivity'][data['PhysicalActivity']],
        mappings['DiffWalking'][data['DiffWalking']],
        mappings['Sex'][data['Sex']],
        mappings['AgeCategory'][data['AgeCategory']],
        mappings['Race'][data['Race']],
        mappings['Diabetic'][data['Diabetic']],
        mappings['GenHealth'][data['GenHealth']],
        mappings['Asthma'][data['Asthma']],
        mappings['KidneyDisease'][data['KidneyDisease']],
        mappings['SkinCancer'][data['SkinCancer']]
    ]

    features = pd.DataFrame([features_list], columns=['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalActivity', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer'])

    # Load top features and poly
    top_features = pickle.load(open(os.path.join(model_dir, 'top_features.pkl'), 'rb'))
    poly = pickle.load(open(os.path.join(model_dir, 'poly.pkl'), 'rb'))

    # Select top features
    features_selected = features[top_features]

    # Apply polynomial features
    features_poly = poly.transform(features_selected)

    results = {}

    # Decision Tree
    dt_pred = dt.predict(features_poly)[0]
    dt_prob = dt.predict_proba(features_poly)[0][1]
    results['dt'] = {'prediction': 'Yes' if dt_pred == 1 else 'No', 'probability': float(dt_prob)}

    # SVM
    features_scaled = scaler.transform(features_poly)
    svm_pred = svm.predict(features_scaled)[0]
    svm_prob = svm.predict_proba(features_scaled)[0][1]
    results['svm'] = {'prediction': 'Yes' if svm_pred == 1 else 'No', 'probability': float(svm_prob)}

    # Random Forest
    rf_pred = rf.predict(features_poly)[0]
    rf_prob = rf.predict_proba(features_poly)[0][1]
    results['rf'] = {'prediction': 'Yes' if rf_pred == 1 else 'No', 'probability': float(rf_prob)}

    # XGBoost
    xgb_pred = xgb.predict(features_poly)[0]
    xgb_prob = xgb.predict_proba(features_poly)[0][1]
    results['xgb'] = {'prediction': 'Yes' if xgb_pred == 1 else 'No', 'probability': float(xgb_prob)}

    # ANN
    ann_pred_prob = ann.predict(features_scaled)[0][0]
    ann_pred = 1 if ann_pred_prob > 0.5 else 0
    results['ann'] = {'prediction': 'Yes' if ann_pred == 1 else 'No', 'probability': float(ann_pred_prob)}

    # Select the highest probability 
    models = ['svm', 'rf', 'ann', 'xgb']
    max_prob = 0
    for model in models:
        prob_yes = results[model]['probability']
        if prob_yes > max_prob:
            max_prob = prob_yes
    ultimate_prediction = 'Yes' if max_prob > 0.5 else 'No'
    ultimate_probability = max_prob
    results['ultimate'] = {'prediction': ultimate_prediction, 'probability': float(ultimate_probability)}

    # Add recommendations if prediction is Yes
    if ultimate_prediction == 'Yes':
        age = data['AgeCategory']
        results['recommendations'] = age_recommendations.get(age, 'General heart disease risks apply.')
        results['note'] = 'Please consult a healthcare professional for a detailed evaluation and medical advice.'

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
