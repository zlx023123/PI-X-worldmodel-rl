# pi0fast-worldmodel-rl

面向单机械臂的视觉—语言—动作（VLA）训练与部署工程骨架。第一阶段围绕
Hugging Face LeRobot、ACT 和公开的 π0-FAST，建立从数据采集到安全部署评测的最小闭环。

> 安全警告：本仓库的 Mock 测试不能证明真实机械臂安全。接入硬件前必须完成机械限位、
> 软件限位、速度/力矩限制、通信 watchdog、实体急停和隔离区域验证。任何模型输出都不得
> 绕过 `SafetyFilter`。

## 项目目标与阶段边界

第一阶段实现：MockRobot/MockCamera 数据采集、内部数据格式、数据校验、episode 级划分、
归一化统计、LeRobot 数据转换、ACT/π0-FAST 训练入口、离线推理、动作块执行、安全过滤、
Mock rollout 和报告输出。

本项目不是完整复现 π0.7。第一阶段只是 π0-FAST 单机械臂工程闭环；大型视觉世界模型、
自主 rollout 数据闭环、完整离线 RL、在线 RL 和 residual RL 均未实现。项目不声称拥有
原始 π0.7 权重或训练数据。

## 代码与开源依赖边界

本仓库自行实现硬件抽象、内部数据格式、安全层、动作空间适配、CLI、评测和测试。
ACT、π0-FAST、FAST tokenizer 和 `LeRobotDataset` 的核心实现不在本仓库中复制；需要时通过
外部 LeRobot 包调用。

2026-08-01 已按 LeRobot 0.6.0 官方包源码核验：

- 训练命令：`lerobot-train`；
- 数据写入：`LeRobotDataset.create/add_frame/save_episode/finalize`；
- ACT：`lerobot.policies.act.modeling_act.ACTPolicy`；
- π0-FAST：`lerobot.policies.pi0_fast.modeling_pi0_fast.PI0FastPolicy`；
- 策略类型：`pi0_fast`；
- 权重：`lerobot/pi0fast-base`；
- tokenizer：`lerobot/fast-action-tokenizer`。

适配器限定已核验范围 `LeRobot >=0.6,<0.7`。升级 LeRobot 后必须重新运行 `doctor`、测试和
转换冒烟测试。参考：[LeRobot 仓库](https://github.com/huggingface/lerobot)、
[π0-FAST 模型卡](https://huggingface.co/lerobot/pi0fast-base)。

## 目录结构

```text
configs/                 YAML 配置与模型 ID
src/pi0fast_wm_rl/
  robots/                机器人接口、Mock 与安全层
  cameras/               Mock/USB 相机和多相机管理
  teleoperation/         Mock/键盘动作源
  data/                  记录、读取、校验、划分、统计、LeRobot 转换
  policies/              MockPolicy、动作映射、ACT/π0-FAST 延迟适配
  training/              依赖检查和 lerobot-train 命令生成
  inference/             观测构建、动作块执行、rollout、服务边界
  evaluation/            离线指标、rollout 汇总和报告
  future/                世界模型与离线 RL 的 Protocol（无模型实现）
  utils/                 配置、日志、随机种子和设备检查
scripts/                 下载、检查、数据、训练与评测入口
tests/                   无网络、无 GPU、无硬件测试
docs/                    中文设计与操作文档
data/                    本地数据（Git 忽略）
models/                  权重、微调结果和统计（Git 忽略）
outputs/                 日志、checkpoint、视频和报告（Git 忽略）
```

## Ubuntu 22.04 / Python 3.12 安装

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip git ffmpeg
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
pi0fast-wm-rl doctor
```

Mock 闭环不需要 LeRobot、GPU、OpenCV 或模型权重。训练环境再显式安装：

```bash
pip install -e ".[lerobot]"
pi0fast-wm-rl doctor
```

LeRobot 0.6.0 要求 Python ≥3.12、PyTorch ≥2.7。应根据 Ubuntu 22.04、RTX 4090 和服务器
CUDA 驱动，从 PyTorch 官方渠道选择匹配的 wheel；不要仅依赖系统自带 CUDA toolkit。
完整说明见 [docs/environment_setup.md](docs/environment_setup.md)。

## Mock 模式快速开始

```bash
pi0fast-wm-rl record \
  --robot mock --camera mock --teleop mock \
  --task configs/task/pick_place.yaml --episodes 10

pi0fast-wm-rl inspect --dataset data/raw/pick_place
pi0fast-wm-rl split --dataset data/raw/pick_place \
  --train 0.8 --val 0.1 --test 0.1 --seed 42
pi0fast-wm-rl norm-stats --dataset data/raw/pick_place
pi0fast-wm-rl rollout \
  --robot mock --camera mock --policy mock --episodes 3 --dry-run
```

Mock 采集不会按真实控制周期等待，因此可快速生成可重复数据。默认每个 episode 为 200 帧。
再次运行会追加新 episode，不覆盖已有数据。

## 数据格式、检查和转换

内部格式优先可检查性：`images/<camera>/*.png`、`steps.npz`、`metadata.json`。每条记录包含
front 图像、可选 wrist 图像、state、action、task、timestamp、episode/frame index 和可选
success。转换前先运行 `inspect`。

```bash
pi0fast-wm-rl convert-lerobot \
  --dataset data/raw/pick_place \
  --output data/processed/lerobot_pick_place \
  --repo-id local/pick_place --fps 10
```

转换需要 `.[lerobot]`，使用 LeRobot 0.6 的真实 writer API，不复制 `LeRobotDataset`。
LeRobot 会按 `frame_index / fps` 生成目标数据集 timestamp；原始内部时间戳在转换前由校验器检查。
更多内容见 [docs/data_collection.md](docs/data_collection.md) 和
[docs/lerobot_integration.md](docs/lerobot_integration.md)。

## 权重目录与下载

公开权重只允许放在：

```text
models/pretrained/pi0fast-base/
models/pretrained/fast-action-tokenizer/
```

这些目录及 `.safetensors/.bin/.pt/.pth/.ckpt` 均被 Git 忽略。先做 dry-run：

```bash
python scripts/download_models.py --model all --dry-run
python scripts/download_models.py --model all
python scripts/download_models.py --model all --verify
```

真实下载前脚本会展示模型 ID、revision 和目标目录，并要求输入 `yes`；令牌仅从
`HF_TOKEN` 环境变量读取。下载后写入目录级 SHA-256 完整性记录。使用权重前仍需自行阅读
各模型卡和许可证。

## ACT 与 π0-FAST 训练

先将数据转换为 LeRobot 格式，再检查训练计划：

```bash
pi0fast-wm-rl train-act --config configs/policy/act.yaml --dry-run
pi0fast-wm-rl train-pi0fast --config configs/policy/pi0fast_bc.yaml --dry-run
```

dry-run 总是校验配置，并报告数据、依赖和权重是否就绪，同时打印最终 `lerobot-train`
命令；依赖缺失不会伪装成可训练。去掉 `--dry-run` 后，只有所有 preflight 项通过才会启动
官方训练进程。

RTX 4090 只有 24GB 显存。π0-FAST 配置默认使用 bfloat16、batch size 1 和 gradient
checkpointing，但实际可行性仍取决于图像数量、分辨率、chunk 和 LeRobot 版本；遇到 OOM
应先降低视觉输入和 batch，随后评估官方 PEFT/LoRA 路径。见
[docs/pi0fast_finetuning.md](docs/pi0fast_finetuning.md)。

## 推理、评测与报告

```bash
pi0fast-wm-rl eval-offline --dataset data/raw/pick_place
```

默认 MockPolicy 评测输出 action MSE、推理延迟和 episode/frame 数量。报告写入
`outputs/reports/report.json` 与 `report.md`。真实任务 success rate 使用 episode success 标签，
但第一阶段不自动判断真实任务成功。

`ActionChunkExecutor` 支持固定 control Hz、只执行 chunk 前 N 步、提前停止、dry-run、延迟记录
和裁剪计数。每一步都调用独立 `SafetyFilter`。

## 真实机械臂适配

确定 SO-101、xArm、Franka 或其他硬件后，主要修改：

1. 复制 `configs/robot/robot_template.yaml` 并填写关节、校准和 watchdog 参数；
2. 参照 `robots/adapter_template.py` 实现 `BaseRobot`；
3. 在 `policies/action_adapter.py` 明确模型维度、关节顺序、夹爪范围和 absolute/delta 模式；
4. 配置 `USBCamera` 或新增相机适配器；
5. 为硬件急停、断连、超时和限位增加集成测试；
6. 先 dry-run、低速、无负载验证，再逐步放开动作范围。

详见 [docs/real_robot_adapter.md](docs/real_robot_adapter.md) 和
[docs/safety.md](docs/safety.md)。

## 当前未完成内容

- 尚未选择或接入真实机械臂、真实相机和实体遥操作器；
- 未下载权重，未在本项目数据上实际训练 ACT 或 π0-FAST；
- 未用真实权重执行 ACT/π0-FAST 端到端推理冒烟测试；
- 未实现带认证/TLS 的网络 policy server；
- 未实现 π0.7 世界模型、自动奖励、离线 RL、在线 RL 或 residual RL；
- LeRobot 0.6 之外的版本需重新核验接口。

## 路线图

推荐顺序：真实硬件只读状态与实体急停 → 小规模遥操作数据 → ACT baseline → π0-FAST
微调/部署 → 真实未来帧 oracle subgoal → 成功轨迹检索式 subgoal → 轻量 latent future
predictor → 失败/干预数据闭环 → reward-weighted BC/AWR → IQL 类离线价值学习 → 最后才评估
residual/在线 RL。详见 [docs/future_extensions.md](docs/future_extensions.md)。

## 常见问题

**没有 LeRobot 为什么 Mock 模式能运行？**  模型和 LeRobot 只在对应适配器中延迟导入。

**`doctor` 显示 CUDA 不可用怎么办？**  Mock 流程不受影响；训练前检查 NVIDIA 驱动、PyTorch
wheel 和 `CUDA_VISIBLE_DEVICES`。

**为什么训练 dry-run 显示 `ready: false`？**  查看 `next_steps`，通常是尚未转换数据、未显式
下载权重或未安装 `.[lerobot]`。

**能否提交采集数据或权重？**  默认不能；相关目录和常见权重扩展名已被 `.gitignore` 排除。

**可以直接把 Mock 安全参数用于真机吗？**  不可以。限位、速度、动作模式和 watchdog 必须基于
具体机器人手册、末端工具和现场风险评估重新确定。

