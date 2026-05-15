import numpy as np


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]

        if self.max_features is None:
            self.max_features = self.n_features

        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X])

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0

        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / m

        return 1 - np.sum(probabilities ** 2)

    def _best_split(self, X, y):
        m, n = X.shape

        if m <= 1:
            return None, None

        feature_indices = np.random.choice(
            n,
            self.max_features,
            replace=False
        )

        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                y_left = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)

                weighted_gini = (
                    len(y_left) * gini_left +
                    len(y_right) * gini_right
                ) / m

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        node = {
            "predicted_class": predicted_class
        }

        if (
            depth < self.max_depth and
            len(y) >= self.min_samples_split and
            len(np.unique(y)) > 1
        ):
            feature, threshold = self._best_split(X, y)

            if feature is not None:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                node["feature"] = feature
                node["threshold"] = threshold
                node["left"] = self._grow_tree(
                    X[left_mask],
                    y[left_mask],
                    depth + 1
                )
                node["right"] = self._grow_tree(
                    X[right_mask],
                    y[right_mask],
                    depth + 1
                )

        return node

    def _predict_row(self, row, node):
        if "feature" not in node:
            return node["predicted_class"]

        if row[node["feature"]] <= node["threshold"]:
            return self._predict_row(row, node["left"])
        else:
            return self._predict_row(row, node["right"])