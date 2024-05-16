import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.impute import KNNImputer
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import numpy as np
import joblib

# Step 1: Load Data
training_data = pd.read_csv('data/training_data.csv', delimiter=';')
test_data = pd.read_csv('data/test_data_no_target.csv', delimiter=';')

# Step 2: Convert Numerical Columns from strings to floats
def convert_to_float(df):
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['Group', 'Class', 'Perform']:
            df[col] = df[col].str.replace(',', '.').astype(float)
    return df

# Apply conversion to the training data
training_data = convert_to_float(training_data)
test_data = convert_to_float(test_data)

# Step 3: One-Hot Encoding for the 'Group' column
training_data = pd.get_dummies(training_data, columns=['Group'])
test_data = pd.get_dummies(test_data, columns=['Group'])

# Ensure the test set has the same columns as the training set
missing_cols = set(training_data.columns) - set(test_data.columns) - {'Class', 'Perform'}
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[training_data.columns.drop(['Class', 'Perform'])]

# Step 4: Separate features and target
X_train = training_data.drop(columns=['Class', 'Perform'])
y_train = training_data['Class']

# Step 5: Handle Missing Values with KNN Imputer
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(test_data)

# Step 6: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Convert scaled data back to DataFrame for easier manipulation
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

# Step 7: Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=5000, solver='lbfgs', n_jobs=-1)

# Perform Exhaustive Feature Selection
efs = EFS(model,
        min_features=50,
        max_features=50,  # You may adjust this based on the computational limits
        scoring='accuracy',
        print_progress=True,
        cv=5, 
        n_jobs=-1)

efs = efs.fit(X_train_split, y_train_split)

# Print the selected features and their performance
selected_features = X_train_split.columns[list(efs.best_idx_)]
print(f"Selected features: {selected_features}")
print(f"Best score: {efs.best_score_}")

# Optionally, you can use the selected features to train and evaluate your final model
X_train_selected = efs.transform(X_train_split)
X_val_selected = efs.transform(X_val_split)

# Train the model on selected features
model.fit(X_train_selected, y_train_split)
y_val_pred = model.predict(X_val_selected)

# Evaluate the model
accuracy = accuracy_score(y_val_split, y_val_pred)
precision = precision_score(y_val_split, y_val_pred, average='weighted')
recall = recall_score(y_val_split, y_val_pred, average='weighted')
f1 = f1_score(y_val_split, y_val_pred, average='weighted')
report = classification_report(y_val_split, y_val_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Classification Report:")
print(report)

# Save the model to a file
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(efs, 'efs_selector.pkl')

# Make predictions on the test set
X_test_selected = efs.transform(X_test_scaled)
y_test_pred = model.predict(X_test_selected)
np.savetxt('logistic_regression_predictions.txt', y_test_pred.astype(int), fmt='%d', newline='\n')