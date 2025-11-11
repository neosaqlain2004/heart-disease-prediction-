# TODO: Heart Disease Prediction Model Improvements - Implementation Steps

## Step 1: Expand Hyperparameter Tuning in train.py
- [x] Update GridSearchCV for Decision Tree: expand params (max_depth: [5,10,15,None], min_samples_split: [2,5,10], min_samples_leaf: [1,2,4])
- [x] Update GridSearchCV for SVM: expand params (C: [0.1,1,10,100], gamma: ['scale','auto',0.01,0.1], kernel: ['rbf','linear'])
- [x] Update GridSearchCV for Random Forest: expand params (n_estimators: [50,100,200], max_depth: [5,10,15,None], min_samples_split: [2,5,10])
- [x] Update GridSearchCV for XGBoost: expand params (n_estimators: [50,100,200], max_depth: [3,5,7], learning_rate: [0.01,0.1,0.2], subsample: [0.6,0.8,1.0])

## Step 2: Enhance ANN in train.py
- [x] Switch to Input layer (already done, confirm compatibility)
- [x] Add more Dense layers (128, 64, 32)
- [x] Add Dropout layers for regularization (after each Dense)
- [x] Increase epochs to 50 with early stopping (patience=5)

## Step 3: Confirm Feature Selection/Engineering
- [x] Feature selection via RF importance (top 10) - already implemented
- [x] Polynomial features (degree=2, interaction_only=True) - already implemented

## Step 4: Retrain and Save Models
- [x] Run updated train.py to retrain all models with tuned params and selected features
- [x] Evaluate models on recall, accuracy, precision, f1, roc-auc
- [x] Save updated models, scaler, poly, top_features

## Step 5: Update App if Needed
- [x] Check if top_features changed; update app.py features_list if necessary
- [x] Update mappings in app.py if any categorical changes

## Step 6: Test App Locally
- [x] Run app.py and test predictions
- [x] Ensure no errors in predictions

## Step 7: Verification
- [ ] Verify improved metrics (higher recall, etc.)
- [ ] Check for any prediction errors or issues

## Step 8: Improve Model Stats
- [x] Update train.py with optimizations (sample 10k, RandomizedSearchCV, more params, ANN improvements)
- [x] Run updated train.py to retrain models and check metrics
- [x] Update app.py for ensemble voting in ultimate prediction
- [x] Test app with new models and verify improvements
