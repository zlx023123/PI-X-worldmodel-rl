# π0-FAST 微调

1. 安装 `.[lerobot]` 并运行 `doctor`；
2. 校验内部数据并转换为 LeRobot 格式；
3. 用 `download_models.py --dry-run` 复核模型 ID 和许可证；
4. 明确确认后下载 base 与 tokenizer；
5. 运行 `train-pi0fast --dry-run` 检查最终命令；
6. 用少量 steps 做显存和数据字段冒烟测试；
7. 再启动正式实验，保存配置、版本和统计文件。

默认配置使用 `lerobot/pi0fast-base`、`lerobot/fast-action-tokenizer`、bfloat16、gradient
checkpointing、chunk size 10 和 batch size 1。RTX 4090 24GB 上必须实测显存；必要时优先减小
相机数量/分辨率和 batch，再核验当前 LeRobot 官方 PEFT 支持。不要未经测量承诺吞吐或显存占用。

训练前还应核对 state/action 维度、相机 key、任务文本、归一化方式、absolute/delta 语义和夹爪
范围。模型训练成功并不等于可直接真机执行，部署仍需 `ActionAdapter` 和 `SafetyFilter`。

