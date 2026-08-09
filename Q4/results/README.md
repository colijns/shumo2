# Q4 结果文件说明

- `question4_result.csv`、`question4_boundary.csv`、`question4_points.csv`、`question4_verify.csv`：原 Q4 快速近似几何的搜索与独立验证结果。
- `question4_unified_a_summary.json`、`question4_unified_a_curve.csv`：Q4 退化为纯 A 时，与 Q3 统一的严格实体复核结果；最终保守推荐 (N_A=619,N_B=0) 以 Q3 的 (M=10000) 正式结果为准。
- `question4_strict_mixed_summary.json`、`question4_strict_mixed_candidates.csv`：固定先生成 750 根 A、再截取候选前缀的 (M=300) 探索性回归结果，仅用于程序检查，不得作为最终报告数据或全局最优证明。

严格混合大样本尚未完成。其本地检查点文件名含 `checkpoint`，不提交到 Git；程序采用临时文件原子替换、上一版 `.bak` 备份和损坏回退，可在以后继续运行。11 个候选的正式筛选应同时报告逐点 Wilson 区间与按 22 个内/外概率区间校正的 Bonferroni 联合判定。
