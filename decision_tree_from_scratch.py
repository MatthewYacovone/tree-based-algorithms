from metrics_for_measuring_the_split import weighted_impurity
import numpy as np

def split_node(X, y, index, value):
    """
    Splits the dataset (X, y) into two subsets (left and right) based on a feature column (index) 
    and a split criterion (value). Handles numerical and categorical splits appropriately using boolean masks.
    """
    x_index = X[:, index]
    # if this feature is numerical
    if X[0, index].dtype.kind in ['i', 'f']:
        mask = x_index >= value
    # if this feature is categorical
    else:
        mask = x_index == value
    # split into left and right child
    left = [X[~mask, :], y[~mask]]
    right = [X[mask, :], y[mask]]

    return left, right

def get_best_split(X, y, criterion):
    best_index, best_value, best_score, children = None, None, 1, None
    for index in range(len[X[0]]):
        for value in np.sort(np.unique(X[:, index])):
            groups = split_node(X, y, index, value)
            impurity = weighted_impurity([groups[0][1], groups[1][1]], criterion)
            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups
    
    return {'index': best_index, 'value': best_value, 'children': children}

def get_leaf(labels):
    # Obtain the leaf as the majority of the labels
    return np.bincount(labels).argmax()

def split(node, max_depth, min_size, depth, criterion):
    """
    Recursive function that links all of them together.
        - assigns a leaf node if one of two child nodes is empty
        - assigns a leaf node if the current branch depth exceeds the maximum depth allowed
        - assigns a leaf node if the node does not contain sufficient samples required for a further split
        - otherwise, it proceeds with a further split with the optimal splitting point
    """
    left, right = node['children']
    del (node['children'])
    if left[1].size == 0:
        node['right'] = get_leaf(right[1])
        return
    if right[1].size == 0:
        node['left'] = get_leaf(left[1])
        return
    # check if the current depth exceeds the maximal depth
    if depth >= max_depth:
        node['left'], node['right'] = get_leaf(left[1]), get_leaf(right[1])
        return
    
    # check if the left child has enough samples
    if left[1].size <= min_size:
        node['left']= get_leaf(left[1])
    else:
        # it has enough samples, we further split
        result = get_best_split(left[0], left[1], criterion)
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['left'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['left'] = get_leaf(result_left)
        else:
            node['left'] = result
            split(node['left'], max_depth, min_size, depth+1, criterion)
    
    # check if the right child has enough samples
    if right[1].size <= min_size:
        node['right'] =  get_leaf(right[1])
    else:
        # it has enough samples, we split further
        result = get_best_split(right[0], right[1], criterion)
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['right'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['right'] = get_leaf(result_left[1])
        else:
            node['right'] = result
            split(node['right'], max_depth, min_size, depth+1, criterion)
