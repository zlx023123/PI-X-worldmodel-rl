# 真实机械臂适配

新适配器必须实现 `BaseRobot` 的完整生命周期，并把厂商 SDK 的字典/消息转换成固定维度
`numpy.ndarray`。建议按以下门槛推进：

1. 只读连接、状态维度、单位和关节顺序测试；
2. 校准加载和版本检查；
3. `stop()` 与实体急停联动测试；
4. 单关节极小动作、低速度、无负载测试；
5. 通信断开、超时、SDK 异常和重连测试；
6. 全部动作先过 `ActionAdapter` 与 `SafetyFilter`；
7. 记录实际执行动作，而不是只记录模型原始输出。

主要文件是 `robots/adapter_template.py`、`robots/base.py`、`robots/safety.py`、
`policies/action_adapter.py` 与新的 `configs/robot/<model>.yaml`。相机在 `cameras/` 中独立适配。

