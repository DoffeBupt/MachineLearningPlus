import numpy as np

def accuracy_score(y_true, y_predict):
    assert y_true[0] == y_predict[0], '给的数据炸了'
    return sum(y_true == y_predict) / len(y_true)
