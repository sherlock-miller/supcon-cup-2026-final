#!/usr/bin/env python3
"""
标定结果注入 vision_manager.py
================================
读取 现场配置/camera_intrinsics.npz + handeye_matrix.npz，
把真实标定参数注入 vision/vision_manager.py 的 pixel_to_arm_coord。

修复记录（2026-08-19 审核）:
- 旧正则匹配 "# TODO: 现场标定后替换" 与实际占位注释不符 → 注入静默失败
- 旧模板是退化实现（忽略末端旋转），改为只替换占位参数块，
  保留 vision_manager.py 中完整的去畸变+变换链算法
"""
import json
import logging
import re
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("apply_calibration")

ROOT = Path(__file__).parent.parent
CALIB_DIR = ROOT / "现场配置"
VISION_MANAGER = ROOT / "vision" / "vision_manager.py"


def main():
    mtx_file = CALIB_DIR / "camera_intrinsics.npz"
    he_file = CALIB_DIR / "handeye_matrix.npz"

    if not mtx_file.exists():
        logger.error(f"缺少内参文件: {mtx_file}（先跑 interactive_calib.py --mode camera）")
        return

    d = np.load(mtx_file)
    mtx, dist = d["mtx"], d["dist"]
    intrinsics = {
        "fx": float(mtx[0, 0]), "fy": float(mtx[1, 1]),
        "cx": float(mtx[0, 2]), "cy": float(mtx[1, 2]),
        "dist_coeffs": [float(x) for x in dist.ravel()],
    }

    handeye = None
    if he_file.exists():
        hd = np.load(he_file)
        handeye = {
            "R_cam2gripper": [[float(x) for x in row] for row in hd["R_cam2gripper"]],
            "t_cam2gripper": [float(x) for x in hd["t_cam2gripper"].ravel()],
        }
        logger.info("检测到手眼矩阵文件，将一并注入")
    else:
        logger.warning("无手眼矩阵文件，保持单位矩阵（仅注入内参）")

    source = VISION_MANAGER.read_text(encoding="utf-8")

    # 生成替换代码（只替换占位参数块，保留完整算法）
    if handeye:
        R_line = f"        R_cam2gripper = np.array({handeye['R_cam2gripper']})"
        t_line = f"        t_cam2gripper = np.array({handeye['t_cam2gripper']}).reshape(3)"
    else:
        R_line = "        R_cam2gripper = np.eye(3)"
        t_line = "        t_cam2gripper = np.zeros(3)"

    new_code = (
        "        # ===== 标定参数（apply_calibration.py 注入）=====\n"
        f"        fx, fy = {intrinsics['fx']:.4f}, {intrinsics['fy']:.4f}    # 相机内参\n"
        f"        cx, cy = {intrinsics['cx']:.4f}, {intrinsics['cy']:.4f}    # 光心\n"
        f"        dist_coeffs = {intrinsics['dist_coeffs']}  # 畸变系数\n"
        "\n"
        "        # 手眼矩阵：相机 → 末端\n"
        f"{R_line}\n"
        f"{t_line}\n"
    )

    # 替换占位参数块（从标定参数注释到 t_cam2gripper 占位行）
    pattern = re.compile(
        r"        # ===== 标定参数.*?t_cam2gripper = np\.zeros\(3\)\n",
        re.DOTALL,
    )
    new_source, count = pattern.subn(new_code, source)

    if count == 0:
        logger.error("未找到占位参数块，请检查 vision_manager.py 的 pixel_to_arm_coord")
        return

    VISION_MANAGER.write_text(new_source, encoding="utf-8")

    logger.info(f"✅ 标定结果已注入 {VISION_MANAGER.name}")
    logger.info(f"   fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}, "
                f"cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
    if handeye:
        logger.info("   手眼矩阵已注入")

    # 顺带更新 config.json 存档
    config = {}
    if mtx_file.exists():
        config["camera_intrinsics"] = {**intrinsics,
                                       "image_size": [int(x) for x in d["image_size"]]}
    if handeye:
        config["handeye"] = handeye
    out = CALIB_DIR / "calibration.json"
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"配置已存档: {out}")


if __name__ == "__main__":
    main()
