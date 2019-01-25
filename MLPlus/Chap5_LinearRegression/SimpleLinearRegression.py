import numpy as np


class SimpleLinearRegression1:

    def __init__(self):
        """初始化模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """训练模型，适应参数->简单线性回归模型"""
        assert x_train.ndim == 1, "只处理一维模型"
        assert len(x_train) == len(y_train), "训练集数据集大小不匹配"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0  # 分子
        d = 0.0  # 分母
        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """对于一维数组进行预测，返回一维结果"""
        assert x_predict.ndim == 1, "只处理一维数组"
        assert self.a_ is not None and self.b_ is not None, "先fit"
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """对于给定的单个x进行预测"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression"
