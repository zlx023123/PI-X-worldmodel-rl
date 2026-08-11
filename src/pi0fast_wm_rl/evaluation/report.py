"""Write machine-readable and concise Markdown evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(metrics: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write report.json and report.md."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "report.json"
    markdown_path = root / "report.md"
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# 评测报告", "", f"- Episode 数量：{metrics.get('episode_count', 'N/A')}"]
    for key in ("frame_count", "action_mse", "completion_rate", "safety_clipped_steps"):
        if key in metrics:
            lines.append(f"- `{key}`：{metrics[key]}")
    lines.extend(["", "完整结构化指标见 `report.json`。", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
