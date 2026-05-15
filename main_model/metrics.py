import numpy as np


class Metrics:

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return tn, fp, fn, tp

    @staticmethod
    def precision(y_true, y_pred):
        tn, fp, fn, tp = Metrics.confusion_matrix(y_true, y_pred)

        denominator = tp + fp

        if denominator == 0:
            return 0

        return tp / denominator

    @staticmethod
    def recall(y_true, y_pred):
        tn, fp, fn, tp = Metrics.confusion_matrix(y_true, y_pred)

        denominator = tp + fn

        if denominator == 0:
            return 0

        return tp / denominator

    @staticmethod
    def specificity(y_true, y_pred):
        tn, fp, fn, tp = Metrics.confusion_matrix(y_true, y_pred)

        denominator = tn + fp

        if denominator == 0:
            return 0

        return tn / denominator

    @staticmethod
    def f1_score(y_true, y_pred):
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)

        denominator = precision + recall

        if denominator == 0:
            return 0

        return 2 * (precision * recall) / denominator