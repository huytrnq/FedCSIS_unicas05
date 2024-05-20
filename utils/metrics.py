import numpy as np
from sklearn.metrics import confusion_matrix

cost_matrix = np.array([[0, 1, 2],
                        [1, 0, 1],
                        [2, 1, 0]])
def calculate_custom_error(preds, gt, cost_matrix=cost_matrix):
    """
    Calculate a custom error metric based on a confusion matrix and a cost matrix.

    Args:
    preds (array-like): Predicted labels.
    gt (array-like): Ground truth (actual) labels.
    cost_matrix (numpy.ndarray): A matrix of costs associated with misclassifications.

    Returns:
    float: The calculated error metric.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(gt, preds)
    
    # Validate dimensions of cost_matrix
    if cm.shape != cost_matrix.shape:
        raise ValueError("Cost matrix dimensions must match the confusion matrix dimensions.")
    
    # Calculate weighted confusion matrix
    weighted_cm = cm * cost_matrix
    
    # Calculate the custom error
    total_samples = len(gt)
    if total_samples == 0:
        raise ValueError("The length of ground truth cannot be zero.")
    
    error = np.sum(weighted_cm) / total_samples
    return error
