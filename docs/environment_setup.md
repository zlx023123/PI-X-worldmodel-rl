# 环境安装

目标开发/训练环境为 Ubuntu 22.04、Python 3.12、RTX 4090 24GB。先只安装 Mock 依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
pi0fast-wm-rl doctor
```

训练时再执行 `pip install -e ".[lerobot]"`。项目核验基线 LeRobot 0.6.0 要求 Python ≥3.12、
PyTorch ≥2.7；其包会引入训练与 π0 系列依赖。服务器应先确认 NVIDIA 驱动可用，再从 PyTorch
官方安装页选择和驱动兼容的 wheel，最后用以下代码确认：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

不要在 `.env` 中提交 token。需要访问受限模型时导出 `HF_TOKEN`。`external/lerobot/` 仅适用于
需要调试上游源码的 editable install，不是仓库必需内容。

