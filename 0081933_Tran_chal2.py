import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np


class CustomRandomForestClassifier:
    def __init__(self, n_estimators=40, random_state=42, cost_matrix=None):
        """Custom Random Forest Classifier

        Args:
            n_estimators (int, optional): Number of estimators. Defaults to 40.
            random_state (int, optional): Random state. Defaults to 42.
            cost_matrix (np.array, optional): Challenge Metric. Defaults to None.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        self.cost_matrix = (
            cost_matrix
            if cost_matrix is not None
            else np.array([[0, 1, 2], 
                            [1, 0, 1], 
                            [2, 1, 0]])
        )
        self.label_encoder = LabelEncoder()

    @staticmethod
    def load_csv(file_path):
        """Load a CSV file

        Args:
            file_path (str): Path to the CSV file

        Returns:
            data: DataFrame containing the data
        """
        data = pd.read_csv(file_path, delimiter=";")
        data = data.replace(",", ".", regex=True).apply(pd.to_numeric, errors="ignore")
        if "Perform" in data.columns:
            data.drop("Perform", axis=1, inplace=True)
        return data

    @staticmethod
    def calculate_custom_error(preds, gt, cost_matrix):
        """Calculate the custom challenge error

        Args:
            preds (list/array): Predictions
            gt (list/array): Ground truth
            cost_matrix (array): Cost matrix

        Raises:
            ValueError: Error message
            ValueError: Error message
        """
        cm = confusion_matrix(gt, preds)
        if cm.shape != cost_matrix.shape:
            raise ValueError(
                "Cost matrix dimensions must match the confusion matrix dimensions."
            )
        weighted_cm = cm * cost_matrix
        total_samples = len(gt)
        if total_samples == 0:
            raise ValueError("The length of ground truth cannot be zero.")
        error = np.sum(weighted_cm) / total_samples
        return error

    def preprocess_data(self, data):
        """Preprocess the data

        Args:
            data (DataFrame): Input data

        Returns:
            data: Preprocessed data
        """
        data["Group"] = self.label_encoder.fit_transform(data["Group"])
        cols = data.columns.tolist()
        cols = cols[1:] + cols[0:1]
        data = data[cols]
        data = data.fillna(0)
        return data

    def fit(self, X, y):
        """Fit the model

        Args:
            X (array): Features
            y (array): Target
        """
        self.classifier.fit(X, y)

    def predict(self, X):
        """Make predictions

        Args:
            X (array): Features

        Returns:
            predictions: Predictions
        """
        return self.classifier.predict(X)

    def evaluate(self, X, y):
        """Evaluate the model

        Args:
            X (array): Features
            y (array): Target

        Returns:
            metrics: Accuracy, classification report, confusion matrix, custom error
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        matrix = confusion_matrix(y, y_pred)
        custom_error = self.calculate_custom_error(y_pred, y, self.cost_matrix)
        return accuracy, report, matrix, custom_error

    def save_predictions(self, predictions, filename="predictions.txt"):
        """Save the predictions to a file

        Args:
            predictions (list/array): Predictions
            filename (str, optional): Path to export file. Defaults to "predictions.txt".
        """
        np.savetxt(filename, predictions, fmt="%d", delimiter="\n")


if __name__ == "__main__":
    model = CustomRandomForestClassifier()

    # Load and preprocess training data
    train_data = model.load_csv("./data/training_data.csv")
    train_data = model.preprocess_data(train_data)
    X = train_data.drop("Class", axis=1)
    y = train_data["Class"] + 1

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy, report, matrix, custom_error = model.evaluate(X_val, y_val)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)
    print("Custom Error:", custom_error)

    # Load and preprocess test data
    test_data = model.load_csv("./data/test_data_no_target.csv")
    test_data["Group"] = model.label_encoder.transform(test_data["Group"])
    X_test = test_data[X_train.columns]
    X_test = X_test.fillna(0)

    # Predict on the test data and save predictions
    predicts = model.predict(X_test)
    model.save_predictions(np.array(predicts - 1))
