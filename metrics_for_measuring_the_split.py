import matplotlib.pyplot as plt
import numpy as np

def gini_impurity_example():
    # The fraction of the positive class varies from 0 to 1
    pos_fraction = np.linspace(0.00, 1.00, 1000)

    # Gini impurity
    gini = 1 - pos_fraction**2 - (1-pos_fraction)**2

    plt.plot(pos_fraction, gini)
    plt.ylim(0, 1)
    plt.xlabel('Positive Fraction')
    plt.ylabel('Gini Impurity')
    plt.show()

# Display Gini Impurity Example
# gini_impurity_example()

def gini_impurity(labels):
    # When the set is empty, it is also pure
    if len(labels) == 0:
        return 0
    # Count the occurences of each label
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions**2)

# Test gini impurity function:
# print(f'{gini_impurity([1, 1, 0, 1, 0]):.4f}')

def entropy_changes_example():
    pos_fraction = np.linspace(0.001, 0.999, 1000)
    ent = - (pos_fraction * np.log2(pos_fraction) + ( 1 - pos_fraction) * np.log2(1 - pos_fraction))

    plt.plot(pos_fraction, ent)
    plt.xlabel('Positive Fraction')
    plt.ylabel('Entropy')
    plt.ylim(0, 1)
    plt.show()

# Display Entropy Example
# entropy_changes_example()

def entropy(labels):
    if len(labels) == 0:
        return 0

    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return - np.sum(fractions * np.log2(fractions))

# Test entropy function
# print(f'{entropy([1, 1, 0, 1, 0]):.4f}')

criterion_function = {'gini': gini_impurity,
                      'entropy': entropy}
def weighted_impurity(groups, criterion='gini'):
    """
    Calculate weighted impurity of children after a split
    @param groups: list of children, and a child consists a list of class labels
    @param criterion: metric to measure the quality of a split,
                        'gini' for gini impurity or
                        'entropy' for information gain
    @return: float, weighted impurity
    """
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * criterion_function[criterion](group)
    
    return weighted_sum

# Test weighted_impurity function
# children_1 = [[1, 0, 1], [0, 1]]
# children_2 = [[1, 1], [0, 0, 1]]
# print(f"Gini Impurity of #1 split: {weighted_impurity(children_1):.4f}")
# print(f"Entropy of #2 split: {weighted_impurity(children_2, 'entropy'):.4f}")