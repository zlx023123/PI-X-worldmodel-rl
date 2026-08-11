# LeRobot 集成边界

本适配器在 2026-08-01 对 LeRobot 0.6.0 wheel 源码做过核验，限定 `>=0.6,<0.7`。

字段映射：

| 内部字段 | LeRobot 字段 |
|---|---|
| `images.front` | `observation.images.front` |
| `images.wrist` | `observation.images.wrist`（可选） |
| `state` | `observation.state` |
| `action` | `action` |
| `task` | `task` |
| episode/frame index | LeRobot writer 生成 |
| timestamp | LeRobot 按 `frame_index / fps` 生成 |

转换器调用官方 `LeRobotDataset.create()`、`add_frame()`、`save_episode()` 和 `finalize()`，不复制
数据集实现。目标目录非空时拒绝覆盖。升级 LeRobot 后应检查 `lerobot_dataset.py`、policy factory、
训练配置、processor pipeline 和数据版本，再调整兼容窗口。

