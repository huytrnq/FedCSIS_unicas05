from utils.dataloader import DataLoader
from utils.metrics import calculate_custom_error

if __name__ == "__main__":
    ### Load training and test data
    train_dataloader = DataLoader('data/training_data.csv', imputation_strategy='median', split=True, test_size=0.2)
    X_train, y_train, X_val, y_val = train_dataloader()
    test_dataloader = DataLoader('data/test_data_no_target.csv', imputation_strategy='median', split=False)
    X_test, y_test = test_dataloader()
    
    