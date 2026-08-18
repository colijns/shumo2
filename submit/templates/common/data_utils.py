







import numpy as np
import pandas as pd


def min_max_normalize(X, axis=0):









    X = np.asarray(X, dtype=float)
    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)
    span = x_max - x_min
    span = np.where(span == 0, 1.0, span)
    return (X - x_min) / span


def z_score_standardize(X, axis=0):









    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=axis, keepdims=True)
    sigma = np.std(X, axis=axis, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma


def _take_param(param, j, name):

    if param is None:
        raise ValueError(f'第 {j} 个指标需要提供 {name} 参数，但当前为 None')
    return param[j]


def forward_transform(X, types, bounds=None, best=None):















    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = X.copy()
    n, m = X.shape
    if len(types) != m:
        raise ValueError(f'types 长度（{len(types)}）必须与指标数（{m}）一致')

    for j in range(m):
        t = types[j]
        col = X[:, j]
        if t == 'benefit':
            pass
        elif t == 'cost':

            X[:, j] = col.max() - col
        elif t == 'interval':

            a, b = _take_param(bounds, j, 'bounds')
            M = max(a - col.min(), col.max() - b)
            if M <= 0:
                X[:, j] = 1.0
            else:
                X[:, j] = np.where(col < a, 1 - (a - col) / M,
                                   np.where(col > b, 1 - (col - b) / M, 1.0))
        elif t == 'mid':

            t0 = _take_param(best, j, 'best')
            M = np.max(np.abs(col - t0))
            X[:, j] = 1.0 if M == 0 else 1 - np.abs(col - t0) / M
        else:
            raise ValueError(f"未知指标类型 '{t}'，仅支持 'benefit'/'cost'/'interval'/'mid'")
    return X


def fill_missing(df, method='mean'):













    df = df.copy()
    if method == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif method == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif method == 'ffill':
        return df.ffill().bfill()
    elif method == 'interpolate':
        return df.interpolate().bfill().ffill()
    else:
        raise ValueError("method 仅支持 'mean'、'median'、'ffill'、'interpolate'")


if __name__ == '__main__':
    np.random.seed(42)


    df = pd.DataFrame({
        '疗效(效益型)': [80.0, 92.0, np.nan, 88.0, 76.0],
        '费用(成本型)': [5000.0, 4200.0, 6100.0, np.nan, 5800.0],
        '住院天数(区间型)': [9.0, 7.0, 12.0, 6.0, 10.0],
        '血药浓度(中间型)': [90.0, 85.0, 70.0, 95.0, 60.0],
    })
    print('===== 1. 原始数据（含缺失值）=====')
    print(df)


    print('\n===== 2. 缺失值填充结果对比 =====')
    for mtd in ['mean', 'median', 'ffill', 'interpolate']:
        filled = fill_missing(df, method=mtd)
        print(f"\n-- method='{mtd}' --")
        print(filled.round(4))


    df_filled = fill_missing(df, method='mean')



    X_pos = forward_transform(df_filled.values,
                              types=['benefit', 'cost', 'interval', 'mid'],
                              bounds={2: (6, 8)},
                              best={3: 90})
    print('\n===== 3. 正向化后（全部越大越好）=====')
    print(pd.DataFrame(X_pos, columns=df.columns).round(4))


    print('\n===== 4. 极差归一化（[0,1]）=====')
    print(pd.DataFrame(min_max_normalize(X_pos, axis=0), columns=df.columns).round(4))
    print('\n===== 5. Z-score 标准化（均值0 方差1）=====')
    print(pd.DataFrame(z_score_standardize(X_pos, axis=0), columns=df.columns).round(4))
    print('\ndata_utils demo 运行完毕')
