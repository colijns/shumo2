import os
import matplotlib
import matplotlib.pyplot as plt
def set_chinese_style():
    if matplotlib.get_backend().lower() != 'agg':
        matplotlib.use('Agg')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
    plt.rcParams['axes.unicode_minus'] = False
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid']:
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
def save_fig(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f'图片已保存到：{os.path.abspath(path)}（dpi={dpi}）')
if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)
    set_chinese_style()
    x = np.arange(1, 13)
    y = 20 + 10 * np.sin(x / 2.0) + np.random.randn(12) * 1.5
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, 'o-', color='#1f77b4', linewidth=1.8, label='月销售额（万元）')
    ax.axhline(y.mean(), color='gray', linestyle='--', linewidth=1, label='平均值')
    ax.set_title('中文学术风格示例折线图')
    ax.set_xlabel('月份')
    ax.set_ylabel('销售额（万元）')
    ax.legend(loc='best')
    save_fig(fig, 'plot_style_demo.png', dpi=300)
    plt.close(fig)
    print('plot_style demo 运行完毕')
