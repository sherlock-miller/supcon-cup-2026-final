#!/usr/bin/env python3
"""
交互式手眼标定程序（手掰示教 + 空格触发）
==========================================
流程:
  阶段1 相机内参标定: 标定纸固定或手拿移动，按空格拍 20 张
  阶段2 手眼标定:     手掰机械臂到不同位姿，按空格采集 25 组
  阶段3 保存结果 → calibration.json（可注入 vision_manager）

用法:
  python interactive_calib.py --mode camera    # 仅内参
  python interactive_calib.py --mode handeye   # 仅手眼（需已有内参）
  python interactive_calib.py --mode all       # 全流程

交互:
  空格 = 采集一帧（成功才计数）
  q    = 退出
  （不支持 q 时按 Ctrl+C）

标定板: ChArUco 5x7 方格, 方格30mm, 标记22mm, DICT_6X6_250
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("interactive-calib")

sys.path.insert(0, str(Path(__file__).parent.parent))

CALIB_DIR = Path(__file__).parent.parent / "现场配置"
CALIB_DIR.mkdir(exist_ok=True)

# ChArUco 标定板参数（《标定标记全集.pdf》第1页）
CHARUCO_SQUARES = (5, 7)
CHARUCO_SQUARE_MM = 30.0
CHARUCO_MARKER_MM = 22.0
CHARUCO_DICT_NAME = "DICT_6X6_250"

CAMERA_IMAGES = 20
HANDEYE_POSES = 25

# 机械臂 API
ARM_BASE_URL = os.getenv("ARM_BASE_URL", "http://192.168.0.22:8087")


# ============================================================
# 相机（Orbbec Gemini 335 RGB）
# ============================================================
class Gemini335:
    """最小 RGB 采集封装（pyorbbecsdk v2）"""

    def __init__(self):
        import pyorbbecsdk as obs
        self.obs = obs
        self._pipeline = None
        self._ctx = obs.Context()
        dev_list = self._ctx.query_devices()
        if dev_list.get_count() == 0:
            raise RuntimeError("未检测到 Orbbec 相机")
        self._device = dev_list.get_device_by_index(0)
        self._sensor_list = self._device.get_sensor_list()

    def start(self):
        obs = self.obs
        self._pipeline = obs.Pipeline(self._device)
        profiles = self._pipeline.get_stream_profile_list(obs.OBSensorType.COLOR_SENSOR)
        # 优先 RGB 格式
        color_profile = None
        for p in profiles:
            if p.get_format() == obs.OBFormat.RGB:
                color_profile = p
                break
        if color_profile is None:
            for p in profiles:
                color_profile = p
                break
        config = obs.Config()
        config.enable_stream(color_profile)
        self._pipeline.start(config)
        logger.info(f"相机流已启动: RGB {color_profile.get_width()}x{color_profile.get_height()}")

    def grab_rgb(self) -> Image.Image:
        """取一帧 RGB → PIL Image"""
        obs = self.obs
        frames = self._pipeline.wait_for_frames(2000)
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("取 RGB 帧失败")
        width, height = color_frame.get_width(), color_frame.get_height()
        data = color_frame.get_data()
        fmt = color_frame.get_format()
        if fmt == obs.OBFormat.RGB:
            arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        elif fmt == obs.OBFormat.MJPG:
            import cv2
            arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            arr = arr[..., ::-1]  # BGR→RGB
        else:
            raise RuntimeError(f"不支持的格式: {fmt}")
        return Image.fromarray(arr)

    def stop(self):
        if self._pipeline:
            self._pipeline.stop()


# ============================================================
# 实时预览窗口（后台线程 + 检测叠加 + 空格采集）
# ============================================================
import queue
import threading


class PreviewWindow:
    """实时相机预览：叠加 ChArUco 检测结果，空格=采集 q/Esc=退出"""

    def __init__(self, cam: "Gemini335"):
        self.cam = cam
        self.events = queue.Queue()
        self.latest = {"image": None, "corners": None, "ids": None,
                       "marker_corners": None, "marker_ids": None}
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import cv2
        win_name = "标定预览 - 空格=采集 q=退出"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        try:
            while not self._stop:
                try:
                    image = self.cam.grab_rgb()
                except Exception:
                    time.sleep(0.1)
                    continue

                # 检测 ChArUco（含 marker 角点用于绘制）
                board = get_charuco_board()
                gray = cv2.cvtColor(np.array(image.convert("RGB")),
                                    cv2.COLOR_RGB2GRAY)
                detector = cv2.aruco.CharucoDetector(board)
                corners, ids, m_corners, m_ids = detector.detectBoard(gray)
                # OpenCV 5.0 要求 corners/ids 数量一致（检测偶发不一致 → 裁剪对齐）
                if (corners is not None and ids is not None
                        and len(corners) != len(ids)):
                    n = min(len(corners), len(ids))
                    corners, ids = corners[:n], ids[:n]
                ok = corners is not None and ids is not None and len(ids) >= 6

                self.latest = {"image": image, "corners": corners, "ids": ids,
                               "marker_corners": m_corners, "marker_ids": m_ids}

                # 绘制（BGR）——自定义绘制，绕开 OpenCV 5.0 draw 函数的形状断言
                disp = np.array(image.convert("RGB"))[..., ::-1].copy()
                try:
                    if ok:
                        # 角点（5.0 返回 (N,2) 或 (N,1,2)，统一 flatten 后画圆）
                        pts = np.asarray(corners).reshape(-1, 2).astype(int)
                        for (x, y) in pts:
                            cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
                        # 标记框
                        if m_corners is not None:
                            for mc in m_corners:
                                mm = np.asarray(mc).reshape(-1, 2).astype(int)
                                if len(mm) >= 3:
                                    cv2.polylines(disp, [mm], True,
                                                  (255, 0, 255), 2)
                        cv2.putText(disp, f"OK {len(ids)} pts - SPACE=采集",
                                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                    (0, 255, 0), 3)
                    else:
                        cv2.putText(disp, f"detecting... ({0 if ids is None else len(ids)} pts)",
                                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                    (0, 0, 255), 3)
                        cv2.putText(disp, "调整标定板位置/角度/光照",
                                    (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 0, 255), 2)
                except cv2.error:
                    cv2.putText(disp, "draw err", (15, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(win_name, disp)

                key = cv2.waitKey(30) & 0xFF
                if key == 32:  # 空格
                    self.events.put(("capture", dict(self.latest)))
                elif key in (ord('q'), ord('Q'), 27):  # q / Esc
                    self.events.put(("quit", None))
                    break
        finally:
            cv2.destroyAllWindows()

    def wait_event(self):
        return self.events.get()

    def stop(self):
        self._stop = True


# ============================================================
# ChArUco 检测
# ============================================================
def get_charuco_board():
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, CHARUCO_DICT_NAME))
    return cv2.aruco.CharucoBoard(
        CHARUCO_SQUARES, CHARUCO_SQUARE_MM, CHARUCO_MARKER_MM, dictionary)


def _calibrate_charuco_compat(corners_list, ids_list, board, img_size):
    """OpenCV 4/5 双版本 Charuco 内参标定兼容层。

    - OpenCV 4.x: cv2.aruco.calibrateCameraCharuco 存在
    - OpenCV 5.x: 该函数被移除，用 board.matchImagePoints 转棋盘格式
      + cv2.calibrateCamera 标准标定
    """
    import cv2
    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        return cv2.aruco.calibrateCameraCharuco(
            corners_list, ids_list, board, img_size, None, None)
    obj_points, img_points = [], []
    for corners, ids in zip(corners_list, ids_list):
        op, ip = board.matchImagePoints(corners, ids)
        obj_points.append(op)
        img_points.append(ip)
    return cv2.calibrateCamera(obj_points, img_points, img_size, None, None)


def _save_handeye_progress(save_dir, R_g2b, t_g2b, R_t2c, t_t2c):
    """位姿对即时存档（JSON）——支持中断后离线重算"""
    import json
    data = {
        "R_g2b": [m.tolist() for m in R_g2b],
        "t_g2b": [v.ravel().tolist() for v in t_g2b],
        "R_t2c": [m.tolist() for m in R_t2c],
        "t_t2c": [v.ravel().tolist() for v in t_t2c],
    }
    with open(save_dir / "handeye_poses.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def gray_size(image: Image.Image):
    """图像尺寸 (w, h)——ChArUco 标定需要"""
    import cv2
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return gray.shape[::-1]


def detect_charuco(image):
    import cv2
    board = get_charuco_board()
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None, None, gray.shape[::-1]
    return corners, ids, gray.shape[::-1]


# ============================================================
# 机械臂位姿
# ============================================================
def get_arm_pose() -> Optional[dict]:
    import requests
    try:
        r = requests.get(f"{ARM_BASE_URL}/api/pose", timeout=3)
        r.raise_for_status()
        return r.json().get("pose")
    except Exception as e:
        logger.warning(f"读取机械臂位姿失败: {e}")
        return None


def rpy_to_R(roll, pitch, yaw) -> np.ndarray:
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


# ============================================================
# 阶段1: 相机内参标定
# ============================================================
def calibrate_camera(cam: Gemini335) -> Tuple[np.ndarray, np.ndarray]:
    import cv2
    logger.info("=" * 60)
    logger.info("阶段1: 相机内参标定（预览窗口按空格拍 20 张）")
    logger.info("=" * 60)
    logger.info("操作: 在预览窗口中观察标定板识别情况（绿点=已识别）")
    logger.info("      移动/倾斜标定纸，画面显示 OK 时按【空格】采集")
    logger.info("      要求: 画面覆盖不同区域/角度/距离")
    logger.info("")

    board = get_charuco_board()
    corners_list, ids_list = [], []
    img_size = None
    save_dir = CALIB_DIR / "标定照片"
    save_dir.mkdir(exist_ok=True)

    pw = PreviewWindow(cam)
    pw.start()
    try:
        while len(corners_list) < CAMERA_IMAGES:
            logger.info(f"  等待采集 [{len(corners_list)}/{CAMERA_IMAGES}]（预览窗口按空格）...")
            ev, data = pw.wait_event()
            if ev == "quit":
                logger.info("用户退出")
                break
            image, corners, ids = data["image"], data["corners"], data["ids"]
            if corners is None or ids is None:
                logger.warning("  ❌ 当前帧角点不足，继续调整...")
                continue
            corners_list.append(corners)
            ids_list.append(ids)
            img_size = gray_size(image)
            image.save(save_dir / f"charuco_{len(corners_list):02d}.png")
            logger.info(f"  ✅ 第 {len(corners_list)} 张（{len(ids)} 角点）")
    finally:
        pw.stop()

    if len(corners_list) < 8:
        raise RuntimeError(f"有效照片不足 {len(corners_list)} < 8")

    logger.info("解算内参...")
    ret, mtx, dist, _, _ = _calibrate_charuco_compat(
        corners_list, ids_list, board, img_size)
    logger.info(f"重投影误差: {ret:.4f} px（建议 <0.5）")
    logger.info(f"内参:\n{mtx}")
    np.savez(CALIB_DIR / "camera_intrinsics.npz", mtx=mtx, dist=dist,
             ret=ret, image_size=img_size)
    logger.info(f"已保存: {CALIB_DIR / 'camera_intrinsics.npz'}")
    return mtx, dist


# ============================================================
# 阶段2: 手眼标定（手掰示教）
# ============================================================
def calibrate_handeye(cam: Gemini335, mtx: np.ndarray, dist: np.ndarray,
                      num_poses: int = HANDEYE_POSES):
    import cv2
    logger.info("=" * 60)
    logger.info(f"阶段2: 手眼标定（手掰示教 + 空格采集 {num_poses} 组）")
    logger.info("=" * 60)
    logger.info("操作: 1. 手掰机械臂到一个新位姿（标定纸在相机视野内）")
    logger.info("      2. 手离开机械臂，预览窗口显示 OK 后按【空格】")
    logger.info("      3. 位姿差异尽量大（平移+旋转都变）")
    logger.info("")

    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []
    save_dir = CALIB_DIR / "手眼标定照片"
    save_dir.mkdir(exist_ok=True)

    pw = PreviewWindow(cam)
    pw.start()
    try:
        while len(R_g2b) < num_poses:
            logger.info(f"  等待采集 [{len(R_g2b)}/{num_poses}]（预览窗口按空格）...")
            ev, data = pw.wait_event()
            if ev == "quit":
                logger.info("用户退出")
                break

            image, corners, ids = data["image"], data["corners"], data["ids"]
            if corners is None or ids is None:
                logger.warning("  ❌ 当前帧角点不足，继续调整...")
                continue

            # 2. 解标定板在相机系的位姿
            # ⚠️ 必须用 board.matchImagePoints 把 charuco 角点映射到棋盘对象点
            # （charuco ID ≠ 棋盘角点索引，直接索引 getChessboardCorners 会错位
            #   —— 审核发现的严重 bug: PnP 仍收敛但矩阵全错）
            board = get_charuco_board()
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            obj_pts = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 1, 3)
            img_pts = np.asarray(img_pts, dtype=np.float32).reshape(-1, 1, 2)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                logger.warning("  ❌ PnP 解算失败，重试")
                continue
            R_target2cam, _ = cv2.Rodrigues(rvec)
            t_target2cam = tvec.reshape(3, 1) / 1000.0  # mm→m

            # 3. 读机械臂末端位姿
            pose = get_arm_pose()
            if pose is None:
                logger.warning("  ❌ 机械臂位姿读取失败，重试")
                continue
            R_g2b_mat = rpy_to_R(pose["roll"], pose["pitch"], pose["yaw"])
            t_g2b_vec = np.array([pose["x"], pose["y"], pose["z"]]).reshape(3, 1)

            R_g2b.append(R_g2b_mat)
            t_g2b.append(t_g2b_vec)
            R_t2c.append(R_target2cam)
            t_t2c.append(t_target2cam)

            image.save(save_dir / f"handeye_{len(R_g2b):02d}.png")
            # 位姿即时存档（JSON）——采集中断/解算失败时可离线重算，不必重拍
            _save_handeye_progress(save_dir, R_g2b, t_g2b, R_t2c, t_t2c)
            logger.info(f"  ✅ 第 {len(R_g2b)} 组（角点 {len(ids)}，末端 "
                        f"({pose['x']:.3f},{pose['y']:.3f},{pose['z']:.3f})）")
    finally:
        pw.stop()

    if len(R_g2b) < 8:
        raise RuntimeError(f"有效位姿对不足 {len(R_g2b)} < 8")

    logger.info("解算手眼矩阵 AX=XB（TSAI 法）...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
    logger.info(f"R_cam2gripper:\n{R_cam2gripper}")
    logger.info(f"t_cam2gripper (m): {t_cam2gripper.ravel()}")
    np.savez(CALIB_DIR / "handeye_matrix.npz",
             R_cam2gripper=R_cam2gripper, t_cam2gripper=t_cam2gripper)
    logger.info(f"已保存: {CALIB_DIR / 'handeye_matrix.npz'}")
    return R_cam2gripper, t_cam2gripper


# ============================================================
# 阶段3: 生成配置
# ============================================================
def generate_config():
    mtx_file = CALIB_DIR / "camera_intrinsics.npz"
    he_file = CALIB_DIR / "handeye_matrix.npz"
    config = {}
    if mtx_file.exists():
        d = np.load(mtx_file)
        mtx, dist = d["mtx"], d["dist"]
        config["camera_intrinsics"] = {
            "fx": float(mtx[0, 0]), "fy": float(mtx[1, 1]),
            "cx": float(mtx[0, 2]), "cy": float(mtx[1, 2]),
            "dist_coeffs": dist.tolist(),
            "image_size": list(d["image_size"]),
        }
    if he_file.exists():
        d = np.load(he_file)
        config["handeye"] = {
            "R_cam2gripper": d["R_cam2gripper"].tolist(),
            "t_cam2gripper": d["t_cam2gripper"].tolist(),
        }
    out = CALIB_DIR / "calibration.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"配置已生成: {out}")
    logger.info("下一步: python scripts/apply_calibration.py 注入代码")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["camera", "handeye", "all"],
                        default="all")
    parser.add_argument("--poses", type=int, default=HANDEYE_POSES)
    args = parser.parse_args()

    cam = Gemini335()
    cam.start()
    try:
        mtx = dist = None
        if args.mode in ("camera", "all"):
            mtx, dist = calibrate_camera(cam)
        if args.mode in ("handeye", "all"):
            if mtx is None:
                f = CALIB_DIR / "camera_intrinsics.npz"
                if f.exists():
                    d = np.load(f)
                    mtx, dist = d["mtx"], d["dist"]
                    logger.info("已加载现有内参")
                else:
                    logger.error("缺少内参，请先 --mode camera")
                    sys.exit(1)
            calibrate_handeye(cam, mtx, dist, num_poses=args.poses)
        generate_config()
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
