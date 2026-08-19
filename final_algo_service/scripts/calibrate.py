#!/usr/bin/env python3
"""
手眼标定 + 相机内参标定工具
==========================
支持两种手眼采样方式：
  1. manual: 人工拖动机械臂到合适姿态后采样（推荐，避免自动运动磕碰）
  2. auto:   使用预设位姿自动运动采样（兼容旧流程）

流程:
  阶段A: 相机内参标定
  阶段B: 手眼标定（eye-in-hand）
  阶段C: 输出标定结果 → 生成标定配置文件

用法:
  python calibrate.py --mode handeye --pattern charuco
  python calibrate.py --mode handeye --pattern charuco --collection-mode manual
  python calibrate.py --mode camera --pattern charuco

相机类型: Gemini335 (Orbbec) — 装在机械臂末端（eye-in-hand）
ChArUco 板: 使用 scripts/标定标记/标定标记全集.pdf 第1页
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# ChArUco 标定板参数（与 scripts/gen_markers.py 生成的第1页一致）
CHARUCO_SQUARES = (5, 7)      # 方格数 (宽, 高)
CHARUCO_SQUARE_MM = 30.0      # 方格边长 mm
CHARUCO_MARKER_MM = 22.0      # 内嵌 ArUco 标记边长 mm
CHARUCO_DICT_NAME = "DICT_6X6_250"  # cv2.aruco 字典名（函数内解析，避免模块级依赖 cv2）
MIN_VALID_HANDEYE_SAMPLES = 6


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


def load_camera_intrinsics() -> Tuple[np.ndarray, np.ndarray]:
    """加载已有相机内参。"""
    intrinsics_file = CALIB_DIR / "camera_intrinsics.npz"
    if not intrinsics_file.exists():
        raise FileNotFoundError(f"缺少内参文件: {intrinsics_file}")
    data = np.load(intrinsics_file)
    logger.info(f"已加载现有内参: {intrinsics_file}")
    return data["mtx"], data["dist"]


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


def get_charuco_board():
    """构建 ChArUco 标定板对象（与 gen_markers.py 第1页一致）"""
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, CHARUCO_DICT_NAME))
    return cv2.aruco.CharucoBoard(
        CHARUCO_SQUARES, CHARUCO_SQUARE_MM, CHARUCO_MARKER_MM, dictionary)


def detect_charuco(image):
    """检测 ChArUco 角点 + 标记角点。返回 (charuco_corners, charuco_ids)
    或 (None, None)。抗遮挡，板可部分可见。"""
    import cv2
    board = get_charuco_board()
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < 6:
        return None, None
    return charuco_corners, charuco_ids


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """XYZ 固定轴欧拉角 → 旋转矩阵。"""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx


def solve_target_pose(image, mtx: np.ndarray, dist: np.ndarray, pattern: str):
    """
    求解标定板在相机坐标系下的位姿。

    Returns:
        (R_target2cam, t_target2cam, detected_points)
        或 (None, None, 0)
    """
    import cv2

    if pattern == "charuco":
        corners, ids = detect_charuco(image)
        if corners is None or ids is None:
            return None, None, 0

        board = get_charuco_board()
        board_points = board.getChessboardCorners()
        object_points = np.asarray(
            [board_points[int(i)] for i in ids.flatten()],
            dtype=np.float32,
        ).reshape(-1, 1, 3)
        image_points = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)

        if len(object_points) < 6:
            return None, None, len(object_points)

        ret, rvec, tvec = cv2.solvePnP(
            object_points, image_points, mtx, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ret:
            return None, None, len(object_points)

        R_target2cam_mat, _ = cv2.Rodrigues(rvec)
        return R_target2cam_mat, tvec, len(object_points)

    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
        * SQUARE_SIZE_MM
    )
    corners = detect_chessboard(image)
    if corners is None:
        return None, None, 0

    ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
    if not ret:
        return None, None, len(corners)
    R_target2cam_mat, _ = cv2.Rodrigues(rvec)
    return R_target2cam_mat, tvec, len(corners)


def read_arm_sample(arm) -> Tuple[Optional[Dict[str, float]], Any]:
    """读取当前末端位姿与关节信息，作为手眼采样输入和追溯日志。"""
    status = arm.get_status()
    pose_resp = arm.get_pose()
    pose = pose_resp.get("pose", {})
    joints = (
        status.get("right_joints")
        or status.get("joints")
        or status.get("left_joints")
    )

    required_keys = {"x", "y", "z", "roll", "pitch", "yaw"}
    if not pose or not required_keys.issubset(pose):
        return None, joints

    normalized_pose = {
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "z": float(pose["z"]),
        "roll": float(pose["roll"]),
        "pitch": float(pose["pitch"]),
        "yaw": float(pose["yaw"]),
    }
    return normalized_pose, joints


def append_jsonl(path: Path, record: Dict[str, Any]):
    """追加写入 JSONL 日志，便于现场追溯每组样本。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def calibrate_camera(cam, num_images: int = CAMERA_IMAGES,
                     pattern: str = "chessboard") -> Tuple[np.ndarray, np.ndarray]:
    """
    阶段A: 相机内参标定
    引导用户在不同位置/角度拍摄标定板
    pattern: "chessboard"（棋盘格）或 "charuco"（ChArUco，推荐，抗遮挡）
    """
    import cv2

    logger.info("=" * 60)
    logger.info(f"阶段A: 相机内参标定 (pattern={pattern})")
    logger.info("=" * 60)
    logger.info(f"需要拍摄 {num_images} 张标定板照片")
    logger.info("要求:")
    if pattern == "charuco":
        logger.info("  1. ChArUco 板至少 6 个标记可见（可部分遮挡，可拍局部）")
    else:
        logger.info("  1. 棋盘格完整出现在画面中")
    logger.info("  2. 每次改变角度和距离（覆盖画面不同区域）")
    logger.info("  3. 保持标定板静止时拍摄")
    logger.info("")

    if pattern == "charuco":
        board = get_charuco_board()
        charuco_corners_list = []
        charuco_ids_list = []
        img_size = None
    else:
        objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM
        objpoints = []
        imgpoints = []

    save_dir = CALIB_DIR / "标定照片"
    save_dir.mkdir(exist_ok=True)

    if pattern == "charuco":
        while len(charuco_corners_list) < num_images:
            input(f"按 Enter 拍摄第 {len(charuco_corners_list)+1}/{num_images} 张（q 提前结束）: ")
            image = cam.capture()
            gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
            corners, ids = detect_charuco(image)
            if corners is not None:
                charuco_corners_list.append(corners)
                charuco_ids_list.append(ids)
                img_size = gray.shape[::-1]
                image.save(save_dir / f"charuco_{len(charuco_corners_list):02d}.png")
                logger.info(f"  ✅ 第 {len(charuco_corners_list)} 张成功（{len(ids)} 个角点）")
            else:
                logger.warning("  ❌ 未检测到足够 ChArUco 角点，重试（调整角度/距离/光照）")
        if len(charuco_corners_list) < 8:
            raise RuntimeError(f"有效照片不足（{len(charuco_corners_list)} < 8），无法标定")
        logger.info("计算内参（ChArUco）...")
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charuco_corners_list, charuco_ids_list, board,
            img_size, None, None)
    else:
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


def calibrate_handeye(
    cam,
    arm,
    mtx: np.ndarray,
    dist: np.ndarray,
    num_poses: int = HANDEYE_POSES,
    pattern: str = "chessboard",
    collection_mode: str = "manual",
):
    """
    阶段B: 手眼标定 (eye-in-hand)
    记录不同视角下的末端位姿与标定板位姿对。
    求解 AX = XB。
    """
    logger.info("=" * 60)
    logger.info(
        f"阶段B: 手眼标定 (eye-in-hand, pattern={pattern}, collection={collection_mode})"
    )
    logger.info("=" * 60)
    logger.info(f"需要 {num_poses} 组位姿对")
    logger.info("要求:")
    if pattern == "charuco":
        logger.info("  1. ChArUco 板固定放置在操作台上，尽量保持平整")
        logger.info("  2. 每次至少识别到 6 个角点，最好 >10 个")
    else:
        logger.info("  1. 棋盘格固定放置在操作台上")
    if collection_mode == "manual":
        logger.info("  2. 进入示教模式后，人工拖动机械臂到不同位姿再采样")
        logger.info("  3. 脚本直接读取 /api/pose 作为末端位姿")
        logger.info("  4. 同时保存 /api/status 中的 right_joints 作为追溯日志")
    else:
        logger.info("  2. 每次机械臂移动到不同位姿后拍照")
        logger.info("  3. 位姿差异尽量大（平移+旋转都变化）")
    logger.info("  5. 标定板在图像中尽量覆盖不同区域、角度和距离")
    logger.info("")

    if collection_mode not in {"manual", "auto"}:
        raise ValueError(f"不支持的 collection_mode: {collection_mode}")

    if collection_mode == "manual":
        logger.warning("手动采样模式不会发送任何 move 指令，只会切换示教模式并读取当前位姿")
        input("按 Enter 进入示教模式（零力矩拖动），准备开始手动摆位...")
        arm.teach_mode(enable=True)

    sample_log_path = CALIB_DIR / "handeye_samples.jsonl"
    if sample_log_path.exists():
        sample_log_path.unlink()

    metadata = {
        "pattern": pattern,
        "collection_mode": collection_mode,
        "charuco_board": {
            "source_pdf": str(Path(__file__).parent / "标定标记" / "标定标记全集.pdf"),
            "page": 1,
            "squares": list(CHARUCO_SQUARES),
            "square_mm": CHARUCO_SQUARE_MM,
            "marker_mm": CHARUCO_MARKER_MM,
            "dictionary": CHARUCO_DICT_NAME,
        },
    }
    append_jsonl(sample_log_path, {"type": "session", **metadata})

    logger.info(f"样本日志将保存到: {sample_log_path}")
    logger.info("")

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

    try:
        attempt = 0
        while len(R_gripper2base) < num_poses:
            attempt += 1
            sample_idx = len(R_gripper2base) + 1

            if collection_mode == "manual":
                logger.info(f"\n样本 {sample_idx}/{num_poses}: 请人工拖动机械臂到新姿态")
                cmd = input("  按 Enter 采样，输入 s 跳过，输入 q 结束: ").strip().lower()
                if cmd == "q":
                    break
                if cmd == "s":
                    logger.info("  已跳过本次采样")
                    continue
            else:
                pose = preset_poses[len(R_gripper2base) % len(preset_poses)]
                x, y, z, roll, pitch, yaw = pose
                logger.info(f"\n位姿 {sample_idx}/{num_poses}: ({x:.3f}, {y:.3f}, {z:.3f})")
                input("  按 Enter 移动机械臂...")
                try:
                    arm.move_linear(
                        x=x, y=y, z=z,
                        roll=roll, pitch=pitch, yaw=yaw,
                        speed=0.15,
                    )
                    arm.wait_until_idle()
                except Exception as e:
                    logger.error(f"  移动失败: {e}")
                    skip = input("  跳过此位姿? (y/n): ").strip().lower()
                    if skip == "y":
                        continue
                    raise
                input("  按 Enter 拍照...")

            image = cam.capture()
            R_target2cam_mat, t_target2cam_vec, detected_points = solve_target_pose(
                image, mtx, dist, pattern
            )

            if R_target2cam_mat is None:
                if pattern == "charuco":
                    logger.warning(
                        f"  ❌ 未检测到足够 ChArUco 角点（当前 {detected_points} 个），"
                        "请调整板位置/角度/距离"
                    )
                else:
                    logger.warning("  ❌ 未检测到棋盘格，请调整棋盘格位置或位姿")
                append_jsonl(sample_log_path, {
                    "type": "failed_sample",
                    "attempt": attempt,
                    "sample_index": sample_idx,
                    "detected_points": int(detected_points),
                    "reason": "target_not_detected",
                })
                continue

            pose, joints = read_arm_sample(arm)
            if pose is None:
                logger.warning("  ❌ 无法从 /api/pose 读取完整末端位姿，已丢弃本次样本")
                append_jsonl(sample_log_path, {
                    "type": "failed_sample",
                    "attempt": attempt,
                    "sample_index": sample_idx,
                    "detected_points": int(detected_points),
                    "reason": "pose_unavailable",
                    "right_joints": joints,
                })
                continue

            image_path = save_dir / f"handeye_{sample_idx:02d}.png"
            image.save(image_path)

            R_g2b = rpy_to_rotation_matrix(pose["roll"], pose["pitch"], pose["yaw"])
            t_g2b = np.array([pose["x"], pose["y"], pose["z"]]).reshape(3, 1)

            R_gripper2base.append(R_g2b)
            t_gripper2base.append(t_g2b)
            R_target2cam.append(R_target2cam_mat)
            t_target2cam.append(t_target2cam_vec)

            append_jsonl(sample_log_path, {
                "type": "sample",
                "attempt": attempt,
                "sample_index": sample_idx,
                "image_path": str(image_path),
                "detected_points": int(detected_points),
                "arm_pose": pose,
                "right_joints": joints,
                "target_rvec_matrix": R_target2cam_mat.tolist(),
                "target_tvec_m": t_target2cam_vec.reshape(3).tolist(),
            })
            logger.info(
                "  ✅ 记录位姿对 %s（检测点数: %s, pose=(%.3f, %.3f, %.3f)）",
                sample_idx,
                detected_points,
                pose["x"], pose["y"], pose["z"],
            )
    finally:
        if collection_mode == "manual":
            try:
                arm.teach_mode(enable=False)
                logger.info("已退出示教模式")
            except Exception as e:
                logger.warning(f"退出示教模式失败，请现场手动确认: {e}")

    if len(R_gripper2base) < MIN_VALID_HANDEYE_SAMPLES:
        raise RuntimeError(
            f"有效位姿对不足（{len(R_gripper2base)} < {MIN_VALID_HANDEYE_SAMPLES}）"
        )

    # 手眼标定求解
    import cv2
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
    append_jsonl(sample_log_path, {
        "type": "result",
        "valid_samples": len(R_gripper2base),
        "R_cam2gripper": R_cam2gripper.tolist(),
        "t_cam2gripper": t_cam2gripper.reshape(3).tolist(),
    })
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
    parser.add_argument("--pattern", choices=["chessboard", "charuco"],
                        default="charuco",
                        help="标定板类型 (默认 charuco；使用标定标记全集.pdf 第1页)"
                             "——charuco 板由 scripts/gen_markers.py 生成")
    parser.add_argument("--collection-mode", choices=["manual", "auto"],
                        default="manual",
                        help="手眼采样模式：manual=人工拖动后读取 /api/pose，auto=按预设位姿自动运动")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("汪汪队决赛 — 手眼标定工具")
    logger.info("=" * 60)

    cam = get_camera()
    arm = get_arm()

    mtx = dist = None
    if args.mode in ("camera", "all"):
        mtx, dist = calibrate_camera(cam, pattern=args.pattern)

    if args.mode in ("handeye", "all"):
        if mtx is None:
            try:
                mtx, dist = load_camera_intrinsics()
            except FileNotFoundError:
                logger.error("缺少内参，请先运行 --mode camera")
                sys.exit(1)
        calibrate_handeye(
            cam, arm, mtx, dist,
            pattern=args.pattern,
            collection_mode=args.collection_mode,
        )

    generate_config_files()


if __name__ == "__main__":
    main()
