from typing import Any
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

class DataLoader:
    def __init__(self, data_path, 
                imputation_strategy='median', 
                scaler=StandardScaler(),
                type='train', # 'train' or 'test'
                split=False, 
                test_size=0.2, 
                drop_nan=None, 
                features='all'):
        """Load and preprocess data for classification tasks

        Args:
            data_path (str): Path to the dataset
            imputation_strategy (str, optional): Impupation method. Defaults to 'median'.
            scaler (object, optional): Scaler object. Defaults to StandardScaler().
            type (str, optional): Type of data. Defaults to 'train'.
            split (bool, optional): Split the data into training and test sets. Defaults to False.
            drop_nan (float, optional): Drop columns with more than a certain percentage of missing values. Defaults to None.
            features (str, optional): Features to keep. Defaults to 'all'.
        """
        self.data_path = data_path
        self.data = None
        self.imputation_strategy = imputation_strategy
        self.scaler = scaler
        self.split = split
        self.test_size = test_size
        self.drop_nan = drop_nan
        self.type = type
        self.features = features

    def load_data(self):
        """Load the dataset"""
        self.data = pd.read_csv(self.data_path, delimiter=';')
        return self.data
    
    def drop_nan_columns(self, threshold=0.5):
        """Drop columns with more than a certain percentage of missing values

        Args:
            threshold (float, optional): Threshold for missing values. Defaults to 0.5.
        """
        self.data = self.data.drop(columns=[x for x in self.data if self.data[x].isna().sum() > len(self.data)*threshold])


    def convert_to_float(self):
        """Convert numerical columns from strings to floats

        Returns:
            data: DataFrame with numerical columns converted to floats
        """
        for col in self.data.columns:
            if self.data[col].dtype == 'object' and col not in ['Group', 'Class']:
                self.data[col] = self.data[col].str.replace(',', '.').astype(float)
        return self.data
    
    def one_hot_encoding(self):
        """Perform one-hot encoding for the 'Group' column

        Returns:
            data: DataFrame with one-hot encoded 'Group' column
        """
        self.data = pd.get_dummies(self.data, columns=['Group'])
        return self.data

    def handle_missing_values(self):
        """Handle missing values using imputation which defines in the constructor
        """
        if self.imputation_strategy == 'median':
            self.data.fillna(self.data.median(numeric_only=True), inplace=True)
        elif self.imputation_strategy == 'mean':
            self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
        elif self.imputation_strategy == 'mode':
            self.data.fillna(self.data.mode(numeric_only=True), inplace=True)
        elif self.imputation_strategy == 'zero':
            self.data.fillna(0, inplace=True)
        elif self.imputation_strategy == 'drop':
            self.data.dropna(inplace=True)
        elif self.imputation_strategy == 'ffill':
            self.data.fillna(method='ffill', inplace=True)
        elif self.imputation_strategy == 'bfill':
            self.data.fillna(method='bfill', inplace=True)
        elif self.imputation_strategy == 'interpolate':
            self.data.interpolate(inplace=True)
        elif self.imputation_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=2)
            self.data = imputer.fit_transform(self.data)
        else:
            raise ValueError('Invalid imputation strategy')
            
    def separate_features_target(self):
        """Separate features and target

        Returns:
            X, y: Features and target
        """
        X = self.data.drop(columns=['Class', 'Perform'])
        y = self.data['Class']
        return X, y

    def scale_features(self, X):
        """Scale the features using StandardScaler

        Returns:
            features: DataFrame with scaled features
        """
        scaler = self.scaler
        cols_to_scale = [col for col in X.columns if 'Group' not in col]
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
        return X

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=self.test_size, random_state=42, stratify=y)
    
    def filter_features(self, X, features):
        if features == 'average':
            X = X[[f for f in X.columns.values if ('d' not in f or 'Group' in f)]]
        elif features == '1-year':
            X = X[[f for f in X.columns.values if ('d' in f or 'Group' in f)]]   
        return X

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.load_data()
        if self.drop_nan:
            self.drop_nan_columns(self.drop_nan)
        self.one_hot_encoding()
        self.convert_to_float()
        self.handle_missing_values()
        if self.type == 'train':
            X, y = self.separate_features_target()
            X_scaled = self.scale_features(X)
            X_scaled = self.filter_features(X_scaled, features=self.features)
            if self.split:
                return self.split_data(X_scaled, y)
        else:
            X_scaled = self.scale_features(self.data)  
            X_scaled = self.filter_features(X_scaled, features=self.features)
            return X_scaled
            