#!/usr/bin/env python3
"""
交互式手眼标定程序（手掰示教 + 空格触发）
==========================================
流程:
  阶段1 相机内参标定: 相机固定，手拿标定纸移动/倾斜，按空格拍 15 张
  阶段2 手眼标定:     标定纸固定，手掰机械臂到不同位姿，按空格采集 15 组
  阶段3 保存结果 → calibration.json（可注入 vision_manager）

用法:
  python interactive_calib.py --mode camera            # 仅内参
  python interactive_calib.py --mode handeye           # 仅手眼（需已有内参）
  python interactive_calib.py --mode all               # 全流程
  python interactive_calib.py --mode camera --images 15
  python interactive_calib.py --mode handeye --poses 15

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

CAMERA_IMAGES = 15
HANDEYE_POSES = 15

# 粗略初始内参（仅用于内参采集时的标定板位姿多样性判断，不做精度用途）
GUESS_MTX = np.array([[600, 0, 320],
                      [0, 600, 240],
                      [0, 0, 1]], dtype=np.float64)

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
    from pathlib import Path
    save_dir = Path(save_dir)
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


def set_teach_mode(enable: bool) -> bool:
    """切换机械臂示教模式（官方文档 §3.12）。

    enable=True: 电机零力矩，手臂可自由拖动（手掰示教的前提）。
    不开示教模式时电机锁死，机械臂根本掰不动——这是 25 组位姿
    全相同事故的根因（掰的只能是相机/标定纸，关节角纹丝不动）。
    """
    import requests
    try:
        r = requests.post(f"{ARM_BASE_URL}/api/teach_mode",
                          json={"enable": enable}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"示教模式切换失败: {e}")
        return False


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
def calibrate_camera(cam: Gemini335, num_images: int = CAMERA_IMAGES
                     ) -> Tuple[np.ndarray, np.ndarray]:
    import cv2
    logger.info("=" * 60)
    logger.info(f"阶段1: 相机内参标定（预览窗口按空格拍 {num_images} 张）")
    logger.info("=" * 60)
    logger.info("操作: 相机固定不动，手持标定纸【移动/倾斜/旋转】")
    logger.info("      画面显示 OK（绿点）时按【空格】采集")
    logger.info("      要求: 标定板位姿多样性——不同位置/角度/距离各覆盖")
    logger.info("      （与手眼标定相反: 内参是板动相机不动）")
    logger.info("")
    logger.info("标定板参数: ChArUco 5x7 方格30mm 标记22mm DICT_6X6_250")
    logger.info("  ⚠️ 确认打印纸是 1:1 且方格实际尺寸=30mm（用尺量一下）")
    logger.info("")

    board = get_charuco_board()
    corners_list, ids_list = [], []
    img_size = None
    pnp_poses = []      # 每张的 (rvec, tvec)——用于视角多样性检查
    same_streak = 0     # 连续位姿不变计数
    save_dir = CALIB_DIR / "标定照片"
    save_dir.mkdir(exist_ok=True)

    pw = PreviewWindow(cam)
    pw.start()
    try:
        while len(corners_list) < num_images:
            logger.info(f"  等待采集 [{len(corners_list)}/{num_images}]（预览窗口按空格）...")
            ev, data = pw.wait_event()
            if ev == "quit":
                logger.info("用户退出")
                break
            image, corners, ids = data["image"], data["corners"], data["ids"]
            if corners is None or ids is None:
                logger.warning("  ❌ 当前帧角点不足，继续调整...")
                continue

            # 视角多样性检查: 粗略内参解 PnP → 与上一张比较板子位姿
            # 板子不动 = 内参解算退化（与手眼位姿不变同理）
            try:
                obj_pts, img_pts = board.matchImagePoints(corners, ids)
                obj_pts = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 1, 3)
                img_pts = np.asarray(img_pts, dtype=np.float32).reshape(-1, 1, 2)
                ok_pnp, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, GUESS_MTX, None,
                    flags=cv2.SOLVEPNP_ITERATIVE)
            except Exception:
                ok_pnp = False
            if ok_pnp and pnp_poses:
                d_t = float(np.linalg.norm(tvec - pnp_poses[-1][1]))
                R_a, _ = cv2.Rodrigues(pnp_poses[-1][0])
                R_b, _ = cv2.Rodrigues(rvec)
                d_r = float(np.degrees(np.arccos(np.clip(
                    (np.trace(R_b @ R_a.T) - 1) / 2, -1, 1))))
                if d_t < 0.02 and d_r < 5.0:
                    same_streak += 1
                    logger.warning(
                        f"  ⚠️ 标定板位姿与上张几乎相同（Δ位置 {d_t*1000:.0f}mm, "
                        f"Δ角度 {d_r:.1f}°）——请移动/倾斜标定纸！")
                    if same_streak >= 3:
                        logger.error(
                            "连续 3 张位姿未变化，已中止采集。\n"
                            "内参标定要求: 手持标定纸移动/倾斜/旋转，\n"
                            "覆盖画面不同区域、不同角度、不同距离。\n"
                            "确认后重新运行本脚本。")
                        break
                else:
                    same_streak = 0
                    logger.info(f"    Δ位置 {d_t*1000:.0f}mm, Δ角度 {d_r:.1f}° ✓")
            if ok_pnp:
                pnp_poses.append((rvec, tvec))

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
    logger.info("正在开启示教模式（电机零力矩）——机械臂必须能自由拖动！")
    if not set_teach_mode(True):
        logger.warning("⚠️ 示教模式开启失败！若机械臂掰不动，先手动确认示教模式。")
    logger.info("操作: 1. 掰【机械臂本体】（不是相机/标定纸）到新位姿")
    logger.info("      2. 手离开机械臂，预览窗口显示 OK 后按【空格】")
    logger.info("      3. 位姿差异尽量大（平移+旋转都变）")
    logger.info("")

    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []
    save_dir = CALIB_DIR / "手眼标定照片"
    save_dir.mkdir(exist_ok=True)
    same_streak = 0  # 连续位姿不变计数

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

            # 3.5 与上一组对比——位姿不变=机械臂没动（掰错对象或示教未开）
            if len(R_g2b) >= 1:
                d_t = float(np.linalg.norm(t_g2b_vec - t_g2b[-1]))
                d_r = float(np.degrees(np.arccos(np.clip(
                    (np.trace(R_g2b_mat @ R_g2b[-1].T) - 1) / 2, -1, 1))))
                if d_t < 0.005 and d_r < 1.0:
                    same_streak += 1
                    logger.warning(
                        f"  ⚠️ 位姿与上组几乎相同（Δ平移 {d_t*1000:.1f}mm, "
                        f"Δ旋转 {d_r:.2f}°）——机械臂可能没动！")
                    if same_streak >= 3:
                        logger.error(
                            "连续 3 组位姿未变化，已中止采集。\n"
                            "请确认: ① 掰的是机械臂本体（银色臂杆），不是相机\n"
                            "        ② 示教模式已开启（臂可自由拖动）\n"
                            "确认后重新运行本脚本。")
                        break
                else:
                    same_streak = 0
                    logger.info(f"    Δ平移 {d_t*1000:.1f}mm, Δ旋转 {d_r:.1f}° ✓")

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
        logger.info("退出示教模式（恢复电机位置控制）...")
        set_teach_mode(False)

    if len(R_g2b) < 8:
        raise RuntimeError(f"有效位姿对不足 {len(R_g2b)} < 8")

    # 位姿多样性检查——位姿无变化时 AX=XB 不可解（审核+现场实测教训）
    from handeye_tsai import check_pose_diversity
    diverse_ok, diversity_report = check_pose_diversity(R_g2b, t_g2b)
    logger.info(f"位姿多样性: {diversity_report}")
    if not diverse_ok:
        raise RuntimeError(
            f"位姿多样性不足，解算结果无效。请重新采集："
            f"必须掰动机械臂本体（不同角度+位置），而不是只移动标定纸。\n"
            f"  检测: {diversity_report}")

    logger.info("解算手眼矩阵 AX=XB（TSAI 法, OpenCV 4/5 双版本兼容）...")
    from handeye_tsai import calibrate_handeye_compat
    R_cam2gripper, t_cam2gripper = calibrate_handeye_compat(
        R_g2b, t_g2b, R_t2c, t_t2c)
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
            "image_size": [int(x) for x in d["image_size"]],
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
    parser.add_argument("--images", type=int, default=CAMERA_IMAGES,
                        help="内参标定照片数（默认 15）")
    parser.add_argument("--poses", type=int, default=HANDEYE_POSES,
                        help="手眼标定位姿组数（默认 15）")
    args = parser.parse_args()

    cam = Gemini335()
    cam.start()
    try:
        mtx = dist = None
        if args.mode in ("camera", "all"):
            mtx, dist = calibrate_camera(cam, num_images=args.images)
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
