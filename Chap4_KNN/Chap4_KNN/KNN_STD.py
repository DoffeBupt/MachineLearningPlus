from collections import Counter
import numpy as np


class KNNClassifier:
    def __init__(self, k):
        assert k >= 1, "k boom"
        self.k = k
        self._X_train = None  # 私有变量
        self._y_train = None

    def fit(self, X_train, y_train):
        assert 1 <= self.k <= X_train.shape[0], "k must be valid"
        assert X_train.shape[0] == y_train.shape[0], "dataset boom"
        self._X_train = X_train
        self._y_train = y_train
        return self  # SKlearn标准

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None, 'you should fit first'
        assert X_predict.shape[1] == self._X_train.shape[1], "预测的boom"

        y_predict = [self._predict(predict_x) for predict_x in X_predict]
        return np.array(y_predict)

    def _predict(self, predict_x):
        assert self._X_train.shape[1] == predict_x.shape[0], "x boom"
        distances = [(np.sum((x_train - predict_x) ** 2)) ** 0.5 for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k
