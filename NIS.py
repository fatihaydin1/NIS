import numpy as np
    
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
    transformedX, indices = np.unique(transformedX, return_index=True, axis=0);
    
    return indices;

# main method
X = np.around(np.random.rand(1000, 10), 2);
indices = NIS(X, 0.2);
print('The number of unique data: ', indices.size, sep='\n', end='\n\n');

