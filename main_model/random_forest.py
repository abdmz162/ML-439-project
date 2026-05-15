
import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_estimators=10,
        max_depth=10,
        min_samples_split=2,
        max_features=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]

        indices = np.random.choice(
            n_samples,
            size=n_samples,
            replace=True
        )

        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []

        n_features = X.shape[1]

        if self.max_features is None:
            max_features = int(np.sqrt(n_features))
        else:
            max_features = self.max_features

        for _ in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )

            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([
            tree.predict(X) for tree in self.trees
        ])

        tree_predictions = np.swapaxes(tree_predictions, 0, 1)

        predictions = []

        for preds in tree_predictions:
            values, counts = np.unique(preds, return_counts=True)
            predictions.append(values[np.argmax(counts)])

        return np.array(predictions)