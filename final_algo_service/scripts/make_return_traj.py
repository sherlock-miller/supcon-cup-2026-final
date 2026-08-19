#!/usr/bin/env python3
"""生成任务1逆轨迹（拍照位 → 初始位）
=====================================
轨迹1（task1.light_1_*.json）是 初始位→拍照位 的单程轨迹。
执行轨迹回放后臂停在按钮位，下一轮轨迹1回放起点错误。
此脚本: 反转轨迹1采样点 → 回初始位轨迹（拍照位→初始位）。
用法:
  python scripts/make_return_traj.py <轨迹1.json> [输出.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAJ_DIR = ROOT / "现场配置" / "轨迹"


def make_return_traj(src_path: Path, out_path: Path) -> dict:
    d = json.load(open(src_path, encoding="utf-8"))
    samples = d["samples"]

    # 反转采样序列（时间重排，关节角序列反向）
    rev = []
    n = len(samples)
    for i, s in enumerate(reversed(samples)):
        rev.append({"t": i / max(n - 1, 1), "joints": list(s["joints"])})

    # 交换 start/end 位姿
    d_out = dict(d)
    d_out["samples"] = rev
    d_out["sample_count"] = len(rev)
    d_out["start_pose"], d_out["end_pose"] = d["end_pose"], d["start_pose"]
    d_out["recording_method"] = "reversed_from_" + str(Path(src_path).name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d_out, f, ensure_ascii=False, indent=1)
    return d_out


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        TRAJ_DIR / "task1.light_1_20260819_153239.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        TRAJ_DIR / "task1_return_home.json"
    d = make_return_traj(src, out)
    print(f"已生成逆轨迹: {out}")
    print(f"  {d['sample_count']} 采样: "
          f"start={d['start_pose']['z']:.3f}m → end={d['end_pose']['z']:.3f}m")
