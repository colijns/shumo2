







import numpy as np


def _check_pair(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f'y_true 与 y_pred 长度不一致：{y_true.shape} vs {y_pred.shape}')
    return y_true, y_pred


def rmse(y_true, y_pred):









    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):









    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):










    y_true, y_pred = _check_pair(y_true, y_pred)
    mask = y_true != 0
    if not np.any(mask):
        raise ValueError('y_true 全为 0，无法计算 MAPE')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def r2_score(y_true, y_pred):









    y_true, y_pred = _check_pair(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        raise ValueError('y_true 为常数序列，R² 无定义')
    return float(1.0 - ss_res / ss_tot)


if __name__ == '__main__':
    np.random.seed(42)


    n = 20
    y_true = np.linspace(10, 100, n) + np.random.randn(n) * 3.0
    y_pred = y_true + np.random.randn(n) * 5.0

    print('===== 预测精度评价指标 =====')
    print(f'样本数 n   = {n}')
    print(f'RMSE       = {rmse(y_true, y_pred):.4f}')
    print(f'MAE        = {mae(y_true, y_pred):.4f}')
    print(f'MAPE       = {mape(y_true, y_pred):.4f} %')
    print(f'R²         = {r2_score(y_true, y_pred):.4f}')
    print('\nmetrics demo 运行完毕')
