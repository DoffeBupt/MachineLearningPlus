import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据传入的数据来设置均值与方差"""
        assert X.ndim == 2, "只处理二维数据"

        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        # 均值归一化处理
        assert X.ndim == 2, '只处理二维'
        assert self.mean_ is not None and self.scale_ is not None, '先去fit'
        assert X.shape[1] == len(self.scale_), '数值的特征数不对'

        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_) / self.scale_
        return resX
