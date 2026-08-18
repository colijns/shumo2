import os
import pandas as pd
def load_table(path, sheet_name=0, **kwargs):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path, **kwargs)
    elif ext in ('.xlsx', '.xls'):
        return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    else:
        raise ValueError(f'不支持的文件扩展名: {ext}（仅支持 .csv/.xlsx/.xls）')
def export_result(obj, path, **kwargs):
    df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        df.to_csv(path, index=False, **kwargs)
    elif ext in ('.xlsx', '.xls'):
        df.to_excel(path, index=False, **kwargs)
    else:
        raise ValueError(f'不支持的文件扩展名: {ext}（仅支持 .csv/.xlsx/.xls）')
    return os.path.abspath(path)
if __name__ == '__main__':
    import numpy as np
    df = pd.DataFrame({'方案': ['A', 'B', 'C'], '得分': [0.8, 0.6, 0.9]})
    p = export_result(df, 'io_utils_demo.csv')
    print(f'已导出：{p}')
    print('读回：')
    print(load_table(p))
    os.remove(p)
    print('\nio_utils demo 运行完毕')
