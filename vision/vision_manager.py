"""
视觉管理模块
============
统一管理 CLIP 分类器、EasyOCR、Grounding DINO 检测器、相机。
复用初赛代码，适配决赛接口。

注意：Gemini335 深度相机需要奥比中光 SDK。
目前使用 OpenCV 作为占位接口，现场需替换为实际 SDK。
"""
import logging
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class VisionManager:
    """视觉模块统一管理"""

    def __init__(self):
        self._classifier = None
        self._detector = None
        self._ocr = None
        self._camera = None
        self._initialized = False

    def initialize(self):
        """初始化所有视觉模型（首次调用时加载）"""
        if self._initialized:
            return

        logger.info("初始化视觉模型...")

        # CLIP 分类器（通用零样本：形状 + 场景）
        try:
            from vision.classifier import get_shape_classifier
            self._classifier = get_shape_classifier()
            logger.info("CLIP 形状分类器就绪")
        except Exception as e:
            logger.error(f"CLIP 加载失败: {e}")

        # Grounding DINO 检测器（复用初赛代码）
        try:
            from vision.detector import GroundingDinoDetector
            self._detector = GroundingDinoDetector()
            logger.info("Grounding DINO 检测器就绪")
        except Exception as e:
            logger.error(f"Grounding DINO 加载失败: {e}")

        # EasyOCR
        try:
            from vision.ocr_engine import OCREngine
            self._ocr = OCREngine()
            logger.info("EasyOCR 就绪")
        except Exception as e:
            logger.error(f"EasyOCR 加载失败: {e}")

        # 相机（占位：现场需替换为 Gemini335 SDK）
        try:
            from vision.camera import CameraWrapper
            self._camera = CameraWrapper()
            logger.info("相机接口就绪")
        except Exception as e:
            logger.error(f"相机初始化失败: {e}")

        self._initialized = True
        logger.info("视觉模型初始化完成")

    # ================================================================
    # 拍照
    # ================================================================

    def capture_image(self) -> Image.Image:
        """拍照并返回 PIL Image"""
        if self._camera is None:
            raise RuntimeError("相机未初始化")
        return self._camera.capture()

    def capture_with_depth(self) -> Tuple[Image.Image, np.ndarray]:
        """拍照并返回 RGB + Depth"""
        if self._camera is None:
            raise RuntimeError("相机未初始化")
        return self._camera.capture_with_depth()

    # ================================================================
    # 任务1 专用：灯亮检测
    # ================================================================

    def detect_lit_light(
        self,
        image: Image.Image,
    ) -> Optional[Dict[str, Any]]:
        """
        检测开关面板上哪个灯亮了。

        策略：
        1. 先用 Grounding DINO 检测三个灯的区域
        2. 对每个灯区域计算亮度/颜色
        3. 返回亮灯的位置信息

        返回: {"label": "light_1", "switch_type": "button", "center": (x, y)}
              或 None 如果没有检测到亮灯
        """
        # 检测灯的位置
        class_names = ["indicator light", "lit light", "LED light"]
        if self._detector:
            detections = self._detector.predict(image, {"class_names": class_names})
        else:
            detections = {"targets": []}

        # 颜色分析：亮灯 vs 灭灯
        targets = detections.get("targets", [])
        img_array = np.array(image.convert("RGB"))

        best_light = None
        best_brightness = 0

        for target in targets:
            cx = int(target["cx"])
            cy = int(target["cy"])
            # 提取灯区域的颜色
            x1 = max(0, cx - 20)
            y1 = max(0, cy - 20)
            x2 = min(img_array.shape[1], cx + 20)
            y2 = min(img_array.shape[0], cy + 20)
            region = img_array[y1:y2, x1:x2]

            # 计算亮度
            brightness = float(np.mean(region))

            # 判断颜色（红/黄/绿）
            r_mean = float(np.mean(region[:, :, 0]))
            g_mean = float(np.mean(region[:, :, 1]))
            b_mean = float(np.mean(region[:, :, 2]))

            if r_mean > g_mean and r_mean > b_mean:
                color = "red"
            elif g_mean > r_mean and g_mean > b_mean:
                color = "green"
            else:
                color = "yellow"

            brightness_score = max(r_mean, g_mean, b_mean)
            if brightness_score > best_brightness:
                best_brightness = brightness_score
                best_light = {
                    "cx": cx,
                    "cy": cy,
                    "brightness": brightness,
                    "color": color,
                }

        if best_light and best_brightness > 100:  # 亮度阈值
            # 根据灯的 Y 坐标判断是哪个灯（从上到下：灯1, 灯2, 灯3）
            img_h = image.height
            light_cy = best_light["cy"]
            if light_cy < img_h * 0.3:
                light_id = "light_1"
            elif light_cy < img_h * 0.6:
                light_id = "light_2"
            else:
                light_id = "light_3"

            from config import SWITCH_PANEL
            switch_type = SWITCH_PANEL["switch_type"].get(light_id, "button")
            return {
                "light_id": light_id,
                "switch_type": switch_type,
                "pixel": (best_light["cx"], best_light["cy"]),
            }

        return None

    # ================================================================
    # 任务2 专用：数字识别
    # ================================================================

    def detect_cube_numbers(
        self, image: Image.Image
    ) -> List[Dict[str, Any]]:
        """
        检测四个长方体上的数字。

        策略：
        1. Grounding DINO 检测四个长方体区域
        2. 对每个区域用 EasyOCR 识别数字
        3. 返回 [(number, position), ...] 按数字排序

        返回: [{"number": 1, "cx": 320, "cy": 240}, ...]
        """
        # 检测长方体
        if self._detector:
            detections = self._detector.predict(
                image,
                {"class_names": ["cube", "block", "rectangular block"]},
            )
        else:
            detections = {"targets": []}

        targets = detections.get("targets", [])
        results = []

        img_array = np.array(image.convert("RGB"))

        for target in targets:
            cx = int(target["cx"])
            cy = int(target["cy"])
            # 裁剪方块区域做 OCR
            x1 = max(0, cx - 80)
            y1 = max(0, cy - 80)
            x2 = min(img_array.shape[1], cx + 80)
            y2 = min(img_array.shape[0], cy + 80)
            crop = Image.fromarray(img_array[y1:y2, x1:x2])

            # OCR 识别数字
            if self._ocr:
                ocr_result = self._ocr.predict(crop)
                text = ocr_result.get("text", "").strip()
            else:
                text = ""

            # 提取数字
            import re
            numbers = re.findall(r'[1-4]', text)
            if numbers:
                num = int(numbers[0])
                results.append({
                    "number": num,
                    "cx": cx,
                    "cy": cy,
                    "raw_text": text,
                })

        # 按数字排序
        results.sort(key=lambda x: x["number"])
        return results

    # ================================================================
    # 任务3 专用：形状分类
    # ================================================================

    def classify_shape(self, image: Image.Image) -> str:
        """
        识别几何体形状。

        策略：CLIP 零样本分类（常见几何体形状）
        """
        from config import SHAPE_LABELS

        if self._classifier:
            result = self._classifier.predict(image)
            label = result.get("label", SHAPE_LABELS[0])
        else:
            label = SHAPE_LABELS[0]

        logger.info(f"形状分类结果: {label}")
        return label

    def detect_and_classify_shapes(
        self, image: Image.Image
    ) -> List[Dict[str, Any]]:
        """
        检测所有几何体并分类。

        策略：
        1. Grounding DINO 检测所有几何体
        2. 裁剪每个几何体区域
        3. CLIP 分类每个几何体的形状

        返回: [{"shape": "长方体", "cx": 320, "cy": 240}, ...]
        """
        # 检测几何体
        if self._detector:
            detections = self._detector.predict(
                image,
                {"class_names": [
                    "cube", "cuboid", "cylinder", "sphere",
                    "polyhedron", "geometric shape", "block"
                ]},
            )
        else:
            detections = {"targets": []}

        targets = detections.get("targets", [])
        results = []

        img_array = np.array(image.convert("RGB"))

        for target in targets:
            cx = int(target["cx"])
            cy = int(target["cy"])
            # 裁剪区域
            x1 = max(0, cx - 60)
            y1 = max(0, cy - 60)
            x2 = min(img_array.shape[1], cx + 60)
            y2 = min(img_array.shape[0], cy + 60)
            crop = Image.fromarray(img_array[y1:y2, x1:x2])

            # CLIP 分类
            shape = self.classify_shape(crop)

            results.append({
                "shape": shape,
                "cx": cx,
                "cy": cy,
                "confidence": target.get("score", 0.0),
            })

        return results

    # ================================================================
    # 坐标变换（像素 → 机械臂基坐标系）
    # ================================================================

    def pixel_to_arm_coord(
        self,
        pixel_x: float,
        pixel_y: float,
        depth_value: float,
        arm_pose: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float, float]:
        """
        像素坐标 + 深度 → 机械臂基坐标系 3D 坐标（完整 eye-in-hand 链路）

        变换链：
          像素 (u,v) + 深度 d
            → 相机坐标系 3D 点（相机内参 + 去畸变）
            → 机械臂末端坐标系（手眼矩阵 X = T_cam2gripper）
            → 基座坐标系（当前末端位姿 = 机械臂正运动学）

        Args:
            pixel_x, pixel_y: 像素坐标
            depth_value: 深度值 (mm)
            arm_pose: 机械臂当前末端位姿（可选；不传则读取 /api/pose）

        Returns:
            (x, y, z) 基座系坐标 (m)
        """
        # ===== 标定参数（calibrate.py 完成后由 apply_calibration.py 自动替换）=====
        fx, fy = 600.0, 600.0    # 相机内参（占位，待标定）
        cx, cy = 320.0, 240.0    # 光心（占位）
        dist_coeffs = []          # 畸变系数（占位）

        # 手眼矩阵：相机 → 末端（占位，待标定）
        R_cam2gripper = np.eye(3)
        t_cam2gripper = np.zeros(3)

        # ===== 1. 去畸变 + 相机系 3D 坐标 =====
        z_cam = depth_value / 1000.0  # mm → m
        if dist_coeffs:
            import cv2
            mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            dist = np.array(dist_coeffs)
            undistorted = cv2.undistortPoints(
                np.array([[[pixel_x, pixel_y]]], dtype=np.float32),
                mtx, dist, P=mtx,
            )[0][0]
            x_cam = float(undistorted[0]) * z_cam
            y_cam = float(undistorted[1]) * z_cam
        else:
            x_cam = (pixel_x - cx) * z_cam / fx
            y_cam = (pixel_y - cy) * z_cam / fy

        point_cam = np.array([x_cam, y_cam, z_cam])

        # ===== 2. 相机系 → 末端系（手眼矩阵） =====
        point_gripper = R_cam2gripper @ point_cam + t_cam2gripper

        # ===== 3. 末端系 → 基座系（末端当前位姿） =====
        if arm_pose is None:
            # 尝试读取机械臂位姿
            try:
                from hardware.arm_client import ArmClient
                arm = ArmClient()
                arm_pose = arm.get_pose().get("pose")
            except Exception:
                arm_pose = None

        if arm_pose:
            # 末端旋转矩阵（roll/pitch/yaw → R）
            roll, pitch, yaw = (
                arm_pose["roll"], arm_pose["pitch"], arm_pose["yaw"]
            )
            # XYZ 固定轴欧拉角 → 旋转矩阵
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
            R_g2b = Rz @ Ry @ Rx
            t_g2b = np.array([
                arm_pose["x"], arm_pose["y"], arm_pose["z"],
            ])
            point_base = R_g2b @ point_gripper + t_g2b
        else:
            # 无位姿信息：假设末端在默认位（近似）
            point_base = point_gripper

        x_arm, y_arm, z_arm = point_base

        logger.debug(
            f"像素 ({pixel_x},{pixel_y}) + {depth_value}mm → 基坐标 "
            f"({x_arm:.3f}, {y_arm:.3f}, {z_arm:.3f})"
        )
        return (float(x_arm), float(y_arm), float(z_arm))
