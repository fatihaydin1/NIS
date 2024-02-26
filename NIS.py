# CODING HAS BEEN CARRIED OUT ON GOOGLE COLAB

import numpy as np
import pandas as pd
from time import time
    
# Nimble Instance Selection (NIS)
def NIS(X, alpha=1.0):
    # Find the minimum of each column of the data set (X)
    u = X.min(axis=0);

    # Subtract the data set (X) from the vector containing the minimum elements (u)
    transformedX = X - u;

    # Multiply the transformed data set (transformedX) by the scaling parameter (alpha)
    transformedX = alpha * transformedX;
    
    # Find the standard deviation of each column of the data set (X)
    v = X.std(1);
    
    # Divide the transformed data set (transformedX) by the standard deviation of each column of the data set (X) as element-wise
    np.seterr(divide='ignore', invalid='ignore');
    transformedX = transformedX / v[:, None];
    
    # Round the each element of the transformed data set to the nearest whole number
    transformedX = np.around(transformedX, 0);
    
    # Replace the NaN values with zero
    transformedX = np.nan_to_num(transformedX);
    
    # Find the indices of the unique rows in the transformed data set (transformedX)
    _, indices = np.unique(transformedX, return_index=True, axis=0);

    return indices;


# MAIN METHOD
# Load dataset
dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/datasets/cardiotocography.csv');
d = dataset.shape[1] - 1;
X = dataset.iloc[:, 0:d].to_numpy();

t0 = time()
indices = NIS(X, 1);
elapsed_time = time() - t0;

print("Elapsed time:\t", elapsed_time);
print("The number of data:", X.shape[0], sep='\t');
print('The number of unique data:', indices.size, sep='\t', end='\n\n');
print("Instances have been reduced by {r:8.2f}%".format(r=(X.shape[0]-indices.size)*100/X.shape[0]));
