#!/usr/bin/env python3
"""
标定结果注入工具
================
将 calibration.json 的标定结果注入 vision_manager.py，
使 pixel_to_arm_coord() 使用真实标定矩阵。

用法: python apply_calibration.py
"""
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("apply-calib")

CALIB_DIR = Path(__file__).parent.parent / "现场配置"
VISION_MANAGER = Path(__file__).parent.parent / "vision" / "vision_manager.py"


def main():
    calib_file = CALIB_DIR / "calibration.json"
    if not calib_file.exists():
        logger.error(f"未找到 {calib_file}")
        logger.info("请先运行 calibrate.py 完成标定")
        return

    with open(calib_file, "r", encoding="utf-8") as f:
        calib = json.load(f)

    intrinsics = calib.get("camera_intrinsics", {})
    handeye = calib.get("handeye", {})

    if not intrinsics:
        logger.error("标定文件缺少内参")
        return

    # 读取 vision_manager.py
    with open(VISION_MANAGER, "r", encoding="utf-8") as f:
        source = f.read()

    # 生成替换代码
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    new_code = f'''        # ===== 标定结果（{calib_file.name} 生成）=====
        fx, fy = {fx:.4f}, {fy:.4f}    # 相机内参
        cx, cy = {cx:.4f}, {cy:.4f}    # 光心
        dist_coeffs = {intrinsics.get("dist_coeffs", [])}
'''
    if handeye:
        R = handeye["R_cam2gripper"]
        t = handeye["t_cam2gripper"]
        new_code += f'''        # 手眼矩阵（相机→末端）
        R_cam2gripper = {R}
        t_cam2gripper = {t}
        import numpy as _np
        _R = _np.array(R_cam2gripper)
        _t = _np.array(t_cam2gripper).reshape(3)
'''
    else:
        new_code += '''        # 手眼矩阵（未标定，使用单位矩阵）
        import numpy as _np
        _R = _np.eye(3)
        _t = _np.zeros(3)
'''

    new_code += f'''        # 深度图 → 相机系 3D 坐标
        z_cam = depth_value / 1000.0  # mm → m
        x_cam = (pixel_x - cx) * z_cam / fx
        y_cam = (pixel_y - cy) * z_cam / fy

        # 相机系 → 末端系 → 基座系
        point_cam = _np.array([x_cam, y_cam, z_cam])
        point_gripper = _R @ point_cam + _t

        # 末端系 → 基座系（需要当前机械臂位姿，此处由上层传入时叠加）
        # 简化：此处假设末端与基座无旋转（现场按需修正）
        x_arm = point_gripper[0]
        y_arm = point_gripper[1]
        z_arm = point_gripper[2]

        logger.debug(f"像素 ({{pixel_x}},{{pixel_y}})+{{depth_value}}mm → 基坐标 ({{x_arm:.3f}},{{y_arm:.3f}},{{z_arm:.3f}})")
        return (x_arm, y_arm, z_arm)
'''

    # 替换函数体
    pattern = re.compile(
        r"        # TODO: 现场标定后替换.*?return \(x_arm, y_arm, z_arm\)\n",
        re.DOTALL,
    )
    new_source, count = pattern.subn(new_code, source)

    if count == 0:
        logger.error("未找到需要替换的代码段，请检查 vision_manager.py")
        return

    with open(VISION_MANAGER, "w", encoding="utf-8") as f:
        f.write(new_source)

    logger.info(f"✅ 标定结果已注入 {VISION_MANAGER}")
    logger.info(f"   fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    if handeye:
        logger.info("   手眼矩阵已注入")


if __name__ == "__main__":
    main()
