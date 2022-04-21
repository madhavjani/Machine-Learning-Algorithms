from collections import Counter
import numpy as np

def entropy(y):
    h = np.bincount(y)
    ps = h / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def leaf_n(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, sample_split=2, depth=100, features=None):
        self.sample_split = sample_split
        self.depth = depth
        self.features = features
        self.root = None

    def fit(self, x, y):
        self.features = x.shape[1] if not self.features else min(self.features, x.shape[1])
        self.root = self.grow(x, y)

    def predict(self, x):
        return np.array([self.traverse(x, self.root) for x in x])

    def grow(self, x, y, depth=0):
        sample_size, feature_size = x.shape
        n_labels = len(np.unique(y))
        if (
            depth >= self.depth
            or n_labels == 1
            or sample_size < self.sample_split
        ):
            leaf_value = self.common_label(y)
            return Node(value=leaf_value)

        feat_index = np.random.choice(feature_size, self.features, replace=False)

        feat_b, thresh_b = self.criteria(x, y, feat_index)

        left_index, right_index = self._split(x[:, feat_b], thresh_b)
        left = self.grow(x[left_index, :], y[left_index], depth + 1)
        right = self.grow(x[right_index, :], y[right_index], depth + 1)
        return Node(feat_b, thresh_b, left, right)

    def criteria(self, X, y, feat_index):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_index:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def gain(self, y, X_column, split_thresh):
        parent_entropy = entropy(y)

        left_index, right_index = self._split(X_column, split_thresh)

        if len(left_index) == 0 or len(right_index) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_index), len(right_index)
        e_l, e_r = entropy(y[left_index]), entropy(y[right_index])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_index = np.argwhere(X_column <= split_thresh).flatten()
        right_index = np.argwhere(X_column > split_thresh).flatten()
        return left_index, right_index

    def traverse(self, x, node):
        if node.leaf_n():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse(x, node.left)
        return self.traverse(x, node.right)

    def common_label(self, y):
        counter = Counter(y)
        most_common = -1
        if len(counter) != 0:
            most_common = counter.most_common(1)[0][0]
        return most_common

