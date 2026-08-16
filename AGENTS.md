# AGENTS.md

本文件为 ZCode 在本仓库工作时的指引。内容与 `CLAUDE.md` 对应，修改约定时两份同步更新。

## 项目概述

数学建模比赛仓库。`templates/` 为完整代码模板库（不可直接修改，复制出来用）；各题代码按题号目录（`Q1/`、`Q2/`…）组织，从 templates 复制编写。

## 环境与命令

- **Python 环境**: conda env `math` — 每个新 shell 先 `conda activate math`；非交互 shell 用全路径 `E:\Software\Scoop\apps\miniconda3\current\envs\math\python.exe`
- **安装依赖**: `pip install -r requirements.txt`（仅列 numpy/pandas/matplotlib/seaborn/openpyxl/pulp；scipy、statsmodels、networkx、scikit-learn 等模板实际用到但未列出，需用时补装）
- **跑 demo**: `python <模块路径>`（每个 `.py` 自带合成数据 demo）
- **跑测试**: `python -m pytest tests/ -q`（全量）；`python -m pytest tests/ -k <关键词>`（筛选）
- **GBK 编码坑**: 先 `conda activate math` 再 `python xxx.py`，不要用 `conda run -n math python xxx.py`

## 目录结构

```
shumo3/
├── templates/               # 模板库（只复制不修改），速查表见 templates/MODULES.md，详细指南 USAGE.md
├── docs/
│   ├── 题目.md              # 赛题原文
│   ├── 问题*.md             # 各题分析笔记
│   └── appendix/            # AI 使用留存附录（hook 自动 + skill 维护，README 有纪律）
├── attachment/              # 赛题附件数据（不入库）
├── Q1/ Q2/ ...              # 各题代码目录（开始解题时创建）
└── .claude/                 # hooks 留存机制 + skill 原始副本（不入库）
```

## 模板使用铁律

1. **templates/ 只复制不修改**：解题代码放题号目录，从相关模块复制
2. **单文件自包含**：每个模板模块不 import 其他算法模块，复制单文件即可用
3. **函数返回 dict**：语义化键（如 `{'scores', 'ranking', 'weights_used'}`），不返回裸 tuple
4. **随机一律固定 seed**：比赛代码传固定 seed（如 `seed=42`）
5. **绘图用 savefig 不用 plt.show**：模板已设 `matplotlib.use('Agg')` + 中文字体，headless 友好；论文配图用 `08_plot_templates/`，png+pdf 双格式
6. **数据 IO 用 `common/io_utils.py` 的 `load_table` / `export_result`**（按扩展名自动识别 csv/xlsx）；预处理用 `common/data_utils.py`；精度评价（RMSE/MAE/MAPE/R²）用 `common/metrics.py`
7. **引入路径**: `sys.path.append('01_evaluation')` 后 `from topsis import entropy_topsis`

## 项目级 skills（`.zcode/skills/`）

- **competition-record** — AI 使用留存流程（绘图注释头、图配套信息、修改痕迹、AI 工具表、声明草稿等 9 步），references/ 下有各步骤模板
- **math-modeling-report** — 论文写作流程

两者从 `.claude/skills/` 镜像而来；更新内容时两处同步改。

## AI 使用留存（强制）

比赛组委会要求附录留存 AI 使用全程，机制已就位（详见 `docs/appendix/README.md`）：

- 用户 prompt 与会话 transcript：项目级 hook 自动归档（Claude Code 侧 `.claude/settings.json` + `.claude/hooks/record_interaction.py`，写入 `docs/appendix/interaction_logs/`）
- 绘图代码注释头、图配套信息、修改痕迹、AI 工具表、声明草稿等：按 **competition-record** skill 步骤手动执行
- 纪律：只增不改、时间戳 `YYYY-MM-DD HH:MM`、三处工具信息一致

## 当前赛题

**A题：低空经济背景下的多无人机协同巡检路径优化**（多目标路径优化 / VRP 类）。原文见 `docs/题目.md`，附件在 `attachment/`：`附件1.xlsx`/`附件2.xlsx` 为数据，`result1.xlsx`~`result3.xlsx` 为问题 1~3 的输出格式样例。

## 工作流

1. 判题型 → 扫 `templates/MODULES.md` 选模块
2. `conda activate math` 后跑 demo 确认模块功能
3. 复制模板到题号目录 → demo 数据换成赛题数据（经 `load_table` 读取）
4. 出论文配图 → `08_plot_templates/`（按 competition-record skill：双格式 + 图配套信息）
5. 固定所有随机 seed，跑测试确认接口未破坏
