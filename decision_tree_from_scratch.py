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
    """
    Iteratively tries every possible split (feature-value combination), finds the one that creates the least impurity.
    Returns the best split along with the data subsets.
    """
    best_index, best_value, best_score, children = None, None, 1, None
    for index in range(len(X[0])): # iterate over each feature column in dataset X
        for value in np.sort(np.unique(X[:, index])): # ensures I consider every possible meaningful split
            groups = split_node(X, y, index, value) # groups hold two subsets (left and right children), each with their corresponding features and labels
            impurity = weighted_impurity([groups[0][1], groups[1][1]], criterion) # uses the labels (y) from each subset
            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups
    
    return {'index': best_index, 'value': best_value, 'children': children}

def get_leaf(labels):
    """
    Takes labels from a node and determines the most common (majority) class
        - np.bincount(labels) counts how many times each class appears in labels
        - .argmax() selects the index (class label) of the most frequent count
        - Returns the most frequent class in the current node
    Returns 
    """
    # Obtain the leaf as the majority of the labels
    return np.bincount(labels).argmax()

def split(node, max_depth, min_size, depth, criterion):
    """
    Repeatedly splits nodes recursively until there's no data left in a node or, it reaches the max allowed depth or, nodes become too small.
        - assigns a leaf node if one of two child nodes is empty
        - assigns a leaf node if the current branch depth exceeds the maximum depth allowed
        - assigns a leaf node if the node does not contain sufficient samples required for a further split
        - otherwise, it proceeds with a further split with the optimal splitting point
    """
    left, right = node['children'] # extracts left and right subsets
    del (node['children']) # delete node from memory
    # If a split results in one child being empty, the other non-empty child becomes a leaf.
	# This prevents useless further splitting because no meaningful data partitioning occurred.
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


def train_tree(X_train, y_train, max_depth, min_size, criterion='gini'):
    X = np.array(X_train)
    y = np.array(y_train)
    root = get_best_split(X, y, criterion)
    split(root, max_depth, min_size, 1, criterion)

    return root

# Test the decision tree
X_train = [['tech', 'professional'],
           ['fashion', 'student'],
           ['fashion', 'professional'],
           ['sports', 'student'],
           ['tech', 'student'],
           ['tech', 'retired'],
           ['sports', 'professional']]
y_train = [1, 0, 0, 0, 1, 0, 1]
tree = train_tree(X_train, y_train, 2, 2)

# Displaying the tree

CONDITION = {'numerical': {'yes': '>=', 'no': '<'},
             'categorical': {'yes': 'is', 'no': 'is not'}}

def visualize_tree(node, depth=0):
    if isinstance(node, dict):
        if node['value'].dtype.kind in ['i', 'f']:
            condition = CONDITION['numerical']
        else:
            condition = CONDITION['categorical']
        print('{}|- X{} {} {}'.format(depth * ' ', node['index'] + 1, condition['no'], node['value']))

        if 'left' in node:
            visualize_tree(node['left'], depth+1)
        print('{}|- X{} {} {}'.format(depth * ' ', node['index'] + 1, condition['yes'], node['value']))

        if 'right' in node:
            visualize_tree(node['right'], depth+1)
    else:
        print(f"{depth * ' '}[{node}]")
    
visualize_tree(tree)

