import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
    
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

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

# Step 3: Handle Missing Values (fill with the median)
training_data.fillna(training_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# Step 4: One-Hot Encoding for the 'Group' column
training_data = pd.get_dummies(training_data, columns=['Group'])
test_data = pd.get_dummies(test_data, columns=['Group'])

# Ensure the test set has the same columns as the training set
missing_cols = set(training_data.columns) - set(test_data.columns) - {'Class', 'Perform'}
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[training_data.columns.drop(['Class', 'Perform'])]

# Step 5: Separate features and target
X_train = training_data.drop(columns=['Class', 'Perform'])
y_train = training_data['Class']

# Step 6: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data)

# Convert scaled data back to DataFrame for easier manipulation
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

# Step 7: Perform Exhaustive Feature Selection
# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
# model = LogisticRegression(max_iter=1000)
# model = SVC(kernel='linear')
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', verbose=1)
# Perform RFECV
rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy', verbose=2)
rfecv.fit(X_train_scaled, y_train)

# Print the selected features and their performance
selected_features = X_train_scaled.columns[rfecv.support_]
print(f"Selected features: {selected_features}")
print(f"Number of features selected: {rfecv.n_features_}")

# Optionally, you can use the selected features to train and evaluate your final model
X_train_selected = rfecv.transform(X_train_scaled)
X_val_selected = rfecv.transform(X_val_split)

# Train the model on selected features
model.fit(X_train_selected, y_train)
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

X_test_selected = rfecv.transform(X_test_scaled)
y_test_pred = model.predict(X_test_selected)
np.savetxt('grid_search.txt', y_test_pred.astype(int), fmt='%d', newline='\n')


# Selected features: Index(['I5', 'I8', 'I9', 'I18', 'I37', 'I38', 'I44', 'I47', 'I57', 'dI5',
#        'dI6', 'dI23', 'dI25', 'dI28', 'dI35', 'dI40', 'dI42', 'dI46', 'dI47',
#        'dI54', 'dI56', 'dI57', 'dI58'],
#       dtype='object')
# Number of features selected: 23
# Accuracy: 1.0
# Precision: 1.0
# Recall: 1.0
# F1 Score: 1.0
# Classification Report:
#               precision    recall  f1-score   support

#           -1       1.00      1.00      1.00       622
#            0       1.00      1.00      1.00       216
#            1       1.00      1.00      1.00       762

#     accuracy                           1.00      1600
#    macro avg       1.00      1.00      1.00      1600
# weighted avg       1.00      1.00      1.00      1600