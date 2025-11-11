import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

# Load data
data = pd.read_csv('heart_2020_1.csv')

# Sample 10,000 instances for better training
if len(data) > 10000:
    data = data.sample(n=10000, random_state=42).reset_index(drop=True)

# Preprocess
data['HeartDisease'] = data['HeartDisease'].map({'Yes': 1, 'No': 0})

# Encode categorical
categorical_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalActivity', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Features and target
features = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalActivity', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
X = data[features]
y = data['HeartDisease']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f'Original training set: {y_train.value_counts().to_dict()}')
print(f'Balanced training set: {pd.Series(y_train_sm).value_counts().to_dict()}')

# Feature Selection using RF importance
rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(X_train_sm, y_train_sm)
importances = rf_temp.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = [features[i] for i in indices[:10]]  # top 10
print(f'Top features: {top_features}')
X_train_sm = X_train_sm[top_features]
X_test = X_test[top_features]

# Polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train_poly = poly.fit_transform(X_train_sm)
X_test_poly = poly.transform(X_test)
features_poly = poly.get_feature_names_out(top_features)

# Scaling for SVM and ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Hyperparameter Tuning for DT
param_dist_dt = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_dt = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_dist_dt, n_iter=50, cv=2, scoring='recall', n_jobs=1, random_state=42)
grid_dt.fit(X_train_poly, y_train_sm)
dt = grid_dt.best_estimator_
print(f'Best DT params: {grid_dt.best_params_}')
y_pred_dt = dt.predict(X_test_poly)
y_prob_dt = dt.predict_proba(X_test_poly)[:, 1]
print(f'DT Accuracy: {accuracy_score(y_test, y_pred_dt)}')
print(f'DT Precision: {precision_score(y_test, y_pred_dt)}')
print(f'DT Recall: {recall_score(y_test, y_pred_dt)}')
print(f'DT F1-Score: {f1_score(y_test, y_pred_dt)}')
print(f'DT ROC-AUC: {roc_auc_score(y_test, y_prob_dt)}')

# Hyperparameter Tuning for SVM
param_dist_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}
grid_svm = RandomizedSearchCV(SVC(probability=True, random_state=42), param_dist_svm, n_iter=50, cv=2, scoring='recall', n_jobs=1, random_state=42)
grid_svm.fit(X_train_scaled, y_train_sm)
svm = grid_svm.best_estimator_
print(f'Best SVM params: {grid_svm.best_params_}')
y_pred_svm = svm.predict(X_test_scaled)
y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]
print(f'SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}')
print(f'SVM Precision: {precision_score(y_test, y_pred_svm)}')
print(f'SVM Recall: {recall_score(y_test, y_pred_svm)}')
print(f'SVM F1-Score: {f1_score(y_test, y_pred_svm)}')
print(f'SVM ROC-AUC: {roc_auc_score(y_test, y_prob_svm)}')

# Hyperparameter Tuning for RF
param_dist_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}
grid_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist_rf, n_iter=50, cv=2, scoring='recall', n_jobs=1, random_state=42)
grid_rf.fit(X_train_poly, y_train_sm)
rf = grid_rf.best_estimator_
print(f'Best RF params: {grid_rf.best_params_}')
y_pred_rf = rf.predict(X_test_poly)
y_prob_rf = rf.predict_proba(X_test_poly)[:, 1]
print(f'RF Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'RF Precision: {precision_score(y_test, y_pred_rf)}')
print(f'RF Recall: {recall_score(y_test, y_pred_rf)}')
print(f'RF F1-Score: {f1_score(y_test, y_pred_rf)}')
print(f'RF ROC-AUC: {roc_auc_score(y_test, y_prob_rf)}')

# Hyperparameter Tuning for XGB
param_dist_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}
grid_xgb = RandomizedSearchCV(XGBClassifier(random_state=42), param_dist_xgb, n_iter=50, cv=2, scoring='recall', n_jobs=1, random_state=42)
grid_xgb.fit(X_train_poly, y_train_sm)
xgb = grid_xgb.best_estimator_
print(f'Best XGB params: {grid_xgb.best_params_}')
y_pred_xgb = xgb.predict(X_test_poly)
y_prob_xgb = xgb.predict_proba(X_test_poly)[:, 1]
print(f'XGB Accuracy: {accuracy_score(y_test, y_pred_xgb)}')
print(f'XGB Precision: {precision_score(y_test, y_pred_xgb)}')
print(f'XGB Recall: {recall_score(y_test, y_pred_xgb)}')
print(f'XGB F1-Score: {f1_score(y_test, y_pred_xgb)}')
print(f'XGB ROC-AUC: {roc_auc_score(y_test, y_prob_xgb)}')

# Improved ANN
ann = Sequential()
ann.add(Input(shape=(X_train_scaled.shape[1],)))
ann.add(Dense(256, activation='relu'))
ann.add(BatchNormalization())
ann.add(Dropout(0.3))
ann.add(Dense(128, activation='relu'))
ann.add(BatchNormalization())
ann.add(Dropout(0.3))
ann.add(Dense(64, activation='relu'))
ann.add(BatchNormalization())
ann.add(Dropout(0.3))
ann.add(Dense(32, activation='relu'))
ann.add(BatchNormalization())
ann.add(Dropout(0.3))
ann.add(Dense(1, activation='sigmoid'))
ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ann.fit(X_train_scaled, y_train_sm, epochs=50, batch_size=128, validation_split=0.2, callbacks=[early_stop], verbose=0)
y_pred_prob_ann = ann.predict(X_test_scaled)
y_pred_ann = (y_pred_prob_ann > 0.5).astype(int).flatten()
print(f'ANN Accuracy: {accuracy_score(y_test, y_pred_ann)}')
print(f'ANN Precision: {precision_score(y_test, y_pred_ann)}')
print(f'ANN Recall: {recall_score(y_test, y_pred_ann)}')
print(f'ANN F1-Score: {f1_score(y_test, y_pred_ann)}')
print(f'ANN ROC-AUC: {roc_auc_score(y_test, y_pred_prob_ann.flatten())}')

# Save models and additional objects
pickle.dump(dt, open('model/dt.pkl', 'wb'))
pickle.dump(svm, open('model/svm.pkl', 'wb'))
pickle.dump(rf, open('model/rf.pkl', 'wb'))
pickle.dump(xgb, open('model/xgb.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
pickle.dump(poly, open('model/poly.pkl', 'wb'))
pickle.dump(top_features, open('model/top_features.pkl', 'wb'))
ann.save('model/ann.h5')

print('Models and preprocessing objects saved successfully')
