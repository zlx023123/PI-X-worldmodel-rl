# 数据采集与内部格式

Mock 采集命令见 README。每个 episode 单独保存：

```text
episode_000000/
  metadata.json
  steps.npz
  images/front/frame_000000.png
  images/wrist/frame_000000.png  # 可选
```

`steps.npz` 包含 `state`、`action`、`timestamp` 和逐帧可选 `success`。`metadata.json` 包含 task、
episode index、维度、相机列表、机器人类型和未来 metadata 字段。记录器拒绝覆盖同名 episode。

采集顺序是读取图像和 state、获得 teleop action、通过 SafetyFilter、记录 observation 与安全
action、再发送动作。真机时必须由时间同步策略替换当前简单的同步读取，并记录相机/机器人
时钟偏移。转换或训练前必须运行 `inspect`。

