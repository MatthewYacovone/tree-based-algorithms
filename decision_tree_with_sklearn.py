from sklearn.tree import DecisionTreeClassifier

# Example data
X_train_n = [[6, 7],
             [2, 4],
             [7, 2],
             [3, 6],
             [4, 7],
             [5, 2],
             [1, 6],
             [2, 0],
             [6, 3],
             [4, 1]]
y_train_n = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

tree_sk = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=2)
tree_sk.fit(X_train_n, y_train_n)

# Visualize the tree
from sklearn.tree import export_graphviz
export_graphviz(tree_sk, out_file='tree.dot', feature_names=['X1', 'X2'], impurity=False, filled=True, class_names=['0', '1'])

