# Q4 结果文件说明

- `question4_result.csv`、`question4_boundary.csv`、`question4_points.csv`、`question4_verify.csv`：原 Q4 快速近似几何的搜索与独立验证结果。
- `question4_unified_a_summary.json`、`question4_unified_a_curve.csv`：Q4 退化为纯 A 时，与 Q3 统一的严格实体复核结果；最终保守推荐 (N_A=619,N_B=0) 以 Q3 的 (M=10000) 正式结果为准。
- `question4_strict_mixed_exploratory_M300_v1.json`、`question4_strict_mixed_exploratory_M300_v1.csv`：旧 11 点集合在固定 750 根 A 序列下的 (M=300) 探索性回归结果，仅用于程序检查，不得作为最终报告数据或全局最优证明。

新版正式模型只检验 8 个含 B 候选：快速搜索的全部 6 个含 B 跨线点，以及 (598,62)、(617,1) 两个对照点；纯 A 统一引用 Q3 的 (M=10000) 结果，不在 Q4 重复抽样。正式样本量预先固定为 (M=4000)，同时报告逐点 Wilson 区间与按 16 个内/外概率区间校正的 Bonferroni 联合判定。

新版严格混合大样本尚未完成，完成后才生成 `question4_strict_mixed_summary.json` 和 `question4_strict_mixed_candidates.csv`。本地检查点文件名含 `checkpoint`，不提交到 Git；程序采用临时文件原子替换、上一版 `.bak` 备份和损坏回退，可在以后继续运行。
