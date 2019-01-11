import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], '数据集炸了'
    assert 0.0 <= test_ratio <= 1, '测试集比例炸了'

    if seed:
        np.random.seed(seed)

    shuffle_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)  # 强制类型转换防止索引爆炸
    test_indexes = shuffle_indexes[:test_size]  # 测试集
    trian_indexes = shuffle_indexes[test_size:]  # 训练集

    X_train = X[trian_indexes]
    X_test = X[test_indexes]

    y_train = y[trian_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
