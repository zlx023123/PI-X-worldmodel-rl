# 第一阶段架构

核心数据流如下：

```text
Teleoperator/Policy
        │ action chunk
        ▼
ActionAdapter → SafetyFilter → ActionChunkExecutor → BaseRobot
                                              │
BaseCamera → CameraManager ─┐                  │ state
                            ├→ ObservationBuilder
BaseRobot ──────────────────┘
        │
        ├→ EpisodeRecorder → Validator → Splitter/Normalization
        │                                │
        └────────────────────────────────┴→ LeRobot adapter → ACT/π0-FAST
```

内部数据格式与 LeRobot 版本解耦，保证没有 LeRobot 时仍可采集、检查和评测。硬件、策略和安全
分别抽象；模型不能直接持有机器人 SDK。`ActionAdapter` 处理语义与维度，`SafetyFilter` 处理
运行时约束，两者不能合并，以便安全逻辑独立测试和审计。

`future/` 只有 Protocol，不参与第一阶段运行依赖。

