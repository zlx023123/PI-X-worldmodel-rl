# 后续世界模型与强化学习扩展

推荐按风险和可验证性排序：

1. 完成 ACT 与 π0-FAST 的真实硬件 baseline；
2. 用真实未来帧作为 oracle subgoal，验证目标图条件是否有收益；
3. 做成功演示检索式 subgoal，建立无需生成模型的基线；
4. 再训练轻量 latent future predictor；
5. 最后评估生成式、多视角世界模型及真假目标混合训练；
6. 收集成功、失败和人工干预轨迹，先做 reward-weighted BC/AWR；
7. 有可靠 reward/quality 后再做 IQL 类离线价值学习；
8. residual RL 和在线 RL 放在安全仿真、动作屏蔽和人工监管成熟之后。

`WorldModel.predict_subgoal()` 与 `OfflineRLTrainer.fit()` 已作为 Protocol 预留。当前 metadata 可保存
quality、mistake、speed、intervention、reward、subtask 与 subgoal source，但这些字段在第一阶段
不得改变 π0-FAST 训练目标。

