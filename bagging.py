import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import BaggingClassifier
import joblib


# Load Data
training_data = pd.read_csv('data/training_data.csv', delimiter=';')
test_data = pd.read_csv('data/test_data_no_target.csv', delimiter=';')

# Convert Numerical Columns from strings to floats
def convert_to_float(df):
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['Group', 'Class', 'Perform']:
            df[col] = df[col].str.replace(',', '.').astype(float)
    return df

training_data = convert_to_float(training_data)
test_data = convert_to_float(test_data)

# Handle Missing Values using Median Imputation
training_data = training_data.fillna(training_data.median(numeric_only=True))
test_data = test_data.fillna(test_data.median(numeric_only=True))

# One-Hot Encoding for the 'Group' column
training_data = pd.get_dummies(training_data, columns=['Group'])
test_data = pd.get_dummies(test_data, columns=['Group'])

# Ensure the test set has the same columns as the training set
missing_cols = set(training_data.columns) - set(test_data.columns) - {'Class', 'Perform'}
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[training_data.columns.drop(['Class', 'Perform'])]

# Separate features and target
X_train = training_data.drop(columns=['Class', 'Perform'])
y_train = training_data['Class'] + 1

# Feature Engineering Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Transform data
X_train_transformed = pipeline.fit_transform(X_train_split)
X_val_transformed = pipeline.transform(X_val_split)
X_test_transformed = pipeline.transform(test_data)

# Initialize XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Define Grid Search Parameters
param_grid = {
    'feature_selection__k': list(range(30, 110, 5)),
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1]
}

# Define a list to hold different models
models = []

# Perform feature selection and create multiple models
for i in range(5):
    feature_selector = SelectKBest(score_func=mutual_info_classif, k=10 + i*5)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', feature_selector),
        ('xgb', xgb_model)
    ])
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=10, error_score='raise')
    # import pdb; pdb.set_trace()
    grid_search.fit(X_train_transformed, y_train_split)
    
    # Append the best model to the list
    models.append(grid_search.best_estimator_)

# Use Bagging to ensemble the models
bagging_ensemble = BaggingClassifier(base_estimator=xgb_model, n_estimators=5, random_state=42, n_jobs=-1)

# Fit the Bagging ensemble
bagging_ensemble.fit(X_train_transformed, y_train_split)

# Evaluate the Bagging ensemble
y_val_pred = bagging_ensemble.predict(X_val_transformed)
accuracy = accuracy_score(y_val_split, y_val_pred)
precision = precision_score(y_val_split, y_val_pred, average='weighted')
recall = recall_score(y_val_split, y_val_pred, average='weighted')
f1 = f1_score(y_val_split, y_val_pred, average='weighted')

print(f'Bagging Ensemble - Accuracy: {accuracy}')
print(f'Bagging Ensemble - Precision: {precision}')
print(f'Bagging Ensemble - Recall: {recall}')
print(f'Bagging Ensemble - F1 Score: {f1}')

# Make predictions on the test set
y_test_pred = bagging_ensemble.predict(X_test_transformed)
np.savetxt('predictions_bagging.txt', y_test_pred, fmt='%d', newline='\n')

# Save the models
print("Saving models...")
for idx, model in enumerate(models):
    joblib.dump(model, f'xgb_model_{idx}.pkl')
joblib.dump(bagging_ensemble, 'bagging_ensemble_model.pkl')

print("Finished!")
