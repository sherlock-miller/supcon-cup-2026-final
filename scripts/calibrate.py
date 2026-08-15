#!/usr/bin/env python3
"""
手眼标定 + 相机内参标定 — 半自动化脚本
======================================
现场最耗时的工作，本脚本将其压缩到 15-20 分钟。

流程:
  阶段A: 相机内参标定（棋盘格 20 张）
  阶段B: 手眼标定（机械臂带相机移动 12 个位姿拍照）
  阶段C: 输出标定结果 → 生成标定配置文件

用法:
  python calibrate.py --mode camera    # 仅内参
  python calibrate.py --mode handeye   # 仅手眼
  python calibrate.py --mode all       # 全流程（推荐）

相机类型: Gemini335 (Orbbec) — 装在机械臂末端（eye-in-hand）
棋盘格: 需打印 9x7 或 8x6 棋盘格，格宽需实测（mm）
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibrate")

# ============================================================
# 配置
# ============================================================
CALIB_DIR = Path(__file__).parent.parent / "现场配置"
CALIB_DIR.mkdir(exist_ok=True)

CHESSBOARD = (8, 6)       # 内角点数 (宽, 高) — 需与打印的棋盘格一致
SQUARE_SIZE_MM = 25.0     # 棋盘格边长 mm — 需实测
CAMERA_IMAGES = 20        # 内参标定照片数
HANDEYE_POSES = 12        # 手眼标定位姿数


def get_camera():
    """获取相机"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from vision.camera import CameraWrapper
    cam = CameraWrapper()
    cam.initialize()
    return cam


def get_arm():
    """获取机械臂"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from hardware.arm_client import ArmClient
    arm = ArmClient()
    if not arm.check_connection():
        logger.warning("机械臂连接失败，手眼标定将无法进行")
    return arm


def detect_chessboard(image) -> Optional[np.ndarray]:
    """检测棋盘格角点"""
    import cv2
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
    if ret:
        # 亚像素精化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners
    return None


def calibrate_camera(cam, num_images: int = CAMERA_IMAGES) -> Tuple[np.ndarray, np.ndarray]:
    """
    阶段A: 相机内参标定
    引导用户在不同位置/角度拍摄棋盘格
    """
    import cv2

    logger.info("=" * 60)
    logger.info("阶段A: 相机内参标定")
    logger.info("=" * 60)
    logger.info(f"需要拍摄 {num_images} 张棋盘格照片")
    logger.info("要求:")
    logger.info("  1. 棋盘格完整出现在画面中")
    logger.info("  2. 每次改变角度和距离（覆盖画面不同区域）")
    logger.info("  3. 保持棋盘格静止时拍摄")
    logger.info("")

    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

    objpoints = []
    imgpoints = []

    save_dir = CALIB_DIR / "标定照片"
    save_dir.mkdir(exist_ok=True)

    while len(objpoints) < num_images:
        input(f"按 Enter 拍摄第 {len(objpoints)+1}/{num_images} 张（q 提前结束）: ")
        image = cam.capture()
        corners = detect_chessboard(image)

        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
            image.save(save_dir / f"calib_{len(objpoints):02d}.png")
            logger.info(f"  ✅ 第 {len(objpoints)} 张成功（角点检测通过）")
        else:
            logger.warning("  ❌ 未检测到棋盘格，重试（调整角度/距离/光照）")

    if len(objpoints) < 10:
        raise RuntimeError(f"有效照片不足（{len(objpoints)} < 10），无法标定")

    logger.info("计算内参...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints,
        (640, 480),  # 需与相机分辨率一致
        None, None,
    )
    logger.info(f"重投影误差: {ret:.3f} px (建议 < 0.5)")
    logger.info(f"内参矩阵:\n{mtx}")
    logger.info(f"畸变系数: {dist.ravel()}")

    # 保存
    np.savez(
        CALIB_DIR / "camera_intrinsics.npz",
        mtx=mtx, dist=dist,
        ret=ret, image_size=(640, 480),
    )
    logger.info(f"内参已保存: {CALIB_DIR / 'camera_intrinsics.npz'}")
    return mtx, dist


def calibrate_handeye(cam, arm, mtx: np.ndarray, dist: np.ndarray, num_poses: int = HANDEYE_POSES):
    """
    阶段B: 手眼标定 (eye-in-hand)
    机械臂移动到不同位姿，相机拍棋盘格，记录位姿对。
    求解 AX = XB。
    """
    import cv2

    logger.info("=" * 60)
    logger.info("阶段B: 手眼标定 (eye-in-hand)")
    logger.info("=" * 60)
    logger.info(f"需要 {num_poses} 组位姿对")
    logger.info("要求:")
    logger.info("  1. 棋盘格固定放置在操作台上")
    logger.info("  2. 每次机械臂移动到不同位姿后拍照")
    logger.info("  3. 位姿差异尽量大（平移+旋转都变化）")
    logger.info("")

    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    save_dir = CALIB_DIR / "手眼标定照片"
    save_dir.mkdir(exist_ok=True)

    # 预设位姿（右臂安全工作域内）
    preset_poses = [
        # (x, y, z, roll, pitch, yaw) — 每 4 个一组不同 Z 高度
        (0.275, -0.28, 0.44, -3.141, -1.552, 3.141),
        (0.275, -0.24, 0.44, -3.141, -1.452, 3.141),
        (0.275, -0.20, 0.44, -3.141, -1.352, 3.141),
        (0.275, -0.16, 0.44, -3.141, -1.252, 3.141),
        (0.275, -0.12, 0.46, -3.141, -1.152, 3.141),
        (0.275, -0.08, 0.46, -3.141, -1.052, 3.141),
        (0.275, -0.04, 0.46, -3.141, -0.952, 3.141),
        (0.275, -0.28, 0.48, -2.941, -1.352, 2.941),
        (0.275, -0.24, 0.48, -2.841, -1.252, 2.841),
        (0.275, -0.20, 0.48, -2.741, -1.152, 2.741),
        (0.275, -0.16, 0.50, -2.641, -1.052, 2.641),
        (0.275, -0.12, 0.50, -2.541, -0.952, 2.541),
    ]

    for i, pose in enumerate(preset_poses[:num_poses]):
        x, y, z, roll, pitch, yaw = pose
        logger.info(f"\n位姿 {i+1}/{num_poses}: ({x:.3f}, {y:.3f}, {z:.3f})")

        # 移动机械臂
        input("  按 Enter 移动机械臂...")
        try:
            arm.move_linear(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, speed=0.15)
            arm.wait_until_idle()
        except Exception as e:
            logger.error(f"  移动失败: {e}")
            skip = input("  跳过此位姿? (y/n): ").strip().lower()
            if skip == 'y':
                continue
            return

        # 拍照
        input("  按 Enter 拍照...")
        image = cam.capture()
        corners = detect_chessboard(image)

        if corners is None:
            logger.warning("  ❌ 未检测到棋盘格，请调整棋盘格位置或位姿")
            continue

        # solvePnP: 棋盘格在相机坐标系下的位姿
        ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
        R_target2cam_mat, _ = cv2.Rodrigues(rvec)
        t_target2cam_vec = tvec

        # 机械臂末端在基坐标系下的位姿
        arm_pose = arm.get_pose()
        p = arm_pose.get("pose", {})
        if not p:
            logger.warning("  无法读取机械臂位姿")
            continue

        # roll/pitch/yaw → 旋转矩阵
        R_g2b, _ = cv2.Rodrigues(np.array([p["roll"], p["pitch"], p["yaw"]]))
        t_g2b = np.array([p["x"], p["y"], p["z"]]).reshape(3, 1)

        R_gripper2base.append(R_g2b)
        t_gripper2base.append(t_g2b)
        R_target2cam.append(R_target2cam_mat)
        t_target2cam.append(t_target2cam_vec)

        image.save(save_dir / f"handeye_{i+1:02d}.png")
        logger.info(f"  ✅ 记录位姿对 {i+1}")

    if len(R_gripper2base) < 6:
        raise RuntimeError(f"有效位姿对不足（{len(R_gripper2base)} < 6）")

    # 手眼标定求解
    logger.info("\n求解手眼标定矩阵 AX=XB ...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    logger.info(f"相机→末端旋转矩阵:\n{R_cam2gripper}")
    logger.info(f"相机→末端平移向量 (m): {t_cam2gripper.ravel()}")

    # 保存
    np.savez(
        CALIB_DIR / "handeye_matrix.npz",
        R_cam2gripper=R_cam2gripper,
        t_cam2gripper=t_cam2gripper,
    )
    logger.info(f"手眼矩阵已保存: {CALIB_DIR / 'handeye_matrix.npz'}")
    return R_cam2gripper, t_cam2gripper


def generate_config_files():
    """
    阶段C: 生成标定配置文件
    将标定结果注入 vision_manager.py 的坐标变换函数
    """
    logger.info("=" * 60)
    logger.info("阶段C: 生成标定配置")
    logger.info("=" * 60)

    intrinsics_file = CALIB_DIR / "camera_intrinsics.npz"
    handeye_file = CALIB_DIR / "handeye_matrix.npz"

    if not intrinsics_file.exists():
        logger.warning("缺少内参文件，仅生成手眼配置")
        return

    data = np.load(intrinsics_file)
    mtx = data["mtx"]
    dist = data["dist"]

    config = {
        "camera_intrinsics": {
            "fx": float(mtx[0, 0]),
            "fy": float(mtx[1, 1]),
            "cx": float(mtx[0, 2]),
            "cy": float(mtx[1, 2]),
            "dist_coeffs": dist.tolist(),
            "image_size": [640, 480],
        },
    }

    if handeye_file.exists():
        he = np.load(handeye_file)
        config["handeye"] = {
            "R_cam2gripper": he["R_cam2gripper"].tolist(),
            "t_cam2gripper": he["t_cam2gripper"].tolist(),
        }

    with open(CALIB_DIR / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"标定配置已生成: {CALIB_DIR / 'calibration.json'}")
    logger.info("")
    logger.info("下一步: 将此文件内容填入 vision/vision_manager.py 的 pixel_to_arm_coord()")
    logger.info("或运行: python apply_calibration.py 自动注入")


def main():
    parser = argparse.ArgumentParser(description="手眼标定半自动化脚本")
    parser.add_argument("--mode", choices=["camera", "handeye", "all"], default="all",
                        help="标定模式 (默认 all)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("汪汪队决赛 — 手眼标定工具")
    logger.info("=" * 60)

    cam = get_camera()
    arm = get_arm()

    mtx = dist = None
    if args.mode in ("camera", "all"):
        mtx, dist = calibrate_camera(cam)

    if args.mode in ("handeye", "all"):
        if mtx is None:
            # 尝试加载已有内参
            intrinsics_file = CALIB_DIR / "camera_intrinsics.npz"
            if intrinsics_file.exists():
                data = np.load(intrinsics_file)
                mtx, dist = data["mtx"], data["dist"]
                logger.info("已加载现有内参")
            else:
                logger.error("缺少内参，请先运行 --mode camera")
                sys.exit(1)
        calibrate_handeye(cam, arm, mtx, dist)

    generate_config_files()


if __name__ == "__main__":
    main()
