"""
Gemini335 深度相机接口封装
==========================
支持两种模式：
1. 直连模式：本机 USB 直连（Orbbec SDK / OpenCV 兜底）
2. 远程模式：通过相机服务 HTTP 获取（Docker 部署时使用）

通过环境变量 CAMERA_SERVER_URL 切换：
  - 不设置 → 直连模式
  - http://host.docker.internal:5002 → 远程模式
"""
import logging
import os
import base64
import io
from typing import Tuple, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CAMERA_SERVER_URL = os.getenv("CAMERA_SERVER_URL", "")

# 尝试导入 Orbbec SDK
try:
    import pyorbbecsdk as obs
    HAS_ORBBEC = True
except ImportError:
    HAS_ORBBEC = False

if not HAS_ORBBEC:
    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False


class CameraWrapper:
    """Gemini335 深度相机接口（直连或远程）"""

    def __init__(self):
        self._device = None
        self._pipeline = None
        self._initialized = False
        self._remote = bool(CAMERA_SERVER_URL)
        self._remote_url = CAMERA_SERVER_URL.rstrip("/")

    def initialize(self):
        """初始化相机"""
        if self._initialized:
            return

        if self._remote:
            self._init_remote()
        elif HAS_ORBBEC:
            self._init_orbbec()
        elif HAS_CV2:
            self._init_cv2()
        else:
            logger.error("无可用的相机后端")

        self._initialized = True

    def _init_remote(self):
        """远程模式：检查相机服务可达性"""
        import requests
        try:
            resp = requests.get(f"{self._remote_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info(f"相机服务就绪 (远程: {self._remote_url})")
            else:
                raise RuntimeError(f"相机服务异常: {resp.status_code}")
        except Exception as e:
            logger.error(f"相机服务连接失败: {e}")
            raise

    def _init_orbbec(self):
        """使用 Orbbec SDK 初始化 Gemini335"""
        try:
            import pyorbbecsdk as obs
            ctx = obs.Context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                raise RuntimeError("未检测到 Orbbec 设备")

            self._device = devices[0]
            self._pipeline = obs.Pipeline(self._device)

            # 配置 RGB + Depth 流
            config = obs.Config()
            config.enable_video_stream(
                obs.OB_STREAM_COLOR,
                640, 480, 30,
                obs.OB_FORMAT_RGB888,
            )
            config.enable_video_stream(
                obs.OB_STREAM_DEPTH,
                640, 480, 30,
                obs.OB_FORMAT_Y16,
            )
            self._pipeline.start(config)
            logger.info("Gemini335 相机初始化成功 (Orbbec SDK)")

        except Exception as e:
            logger.error(f"Orbbec 相机初始化失败: {e}")
            raise

    def _init_cv2(self):
        """使用 OpenCV 初始化普通 USB 相机（占位）"""
        import cv2
        self._cv2_cap = cv2.VideoCapture(0)
        if not self._cv2_cap.isOpened():
            raise RuntimeError("无法打开相机 (CV2)")
        # 设置分辨率
        self._cv2_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cv2_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info("相机初始化成功 (OpenCV 占位)")

    # ================================================================
    # 拍照
    # ================================================================

    def capture(self) -> Image.Image:
        """拍照，返回 RGB PIL Image"""
        self.initialize()

        if self._remote:
            rgb, _ = self._capture_remote()
            return rgb
        elif HAS_ORBBEC:
            return self._capture_orbbec()
        elif HAS_CV2:
            return self._capture_cv2()
        else:
            # 返回黑色占位图
            return Image.new("RGB", (640, 480), (0, 0, 0))

    def capture_with_depth(self) -> Tuple[Image.Image, np.ndarray]:
        """拍照，返回 (RGB Image, Depth ndarray in mm)"""
        self.initialize()

        if self._remote:
            return self._capture_remote()
        elif HAS_ORBBEC:
            return self._capture_with_depth_orbbec()
        elif HAS_CV2:
            rgb = self._capture_cv2()
            depth = np.zeros((480, 640), dtype=np.uint16)
            return rgb, depth
        else:
            return Image.new("RGB", (640, 480)), np.zeros((480, 640), dtype=np.uint16)

    def _capture_remote(self) -> Tuple[Image.Image, np.ndarray]:
        """远程模式：通过 HTTP 从相机服务获取图像"""
        import requests
        resp = requests.get(f"{self._remote_url}/capture", timeout=10)
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"相机服务返回失败: {data.get('message')}")

        # RGB
        rgb_bytes = base64.b64decode(data["rgb"])
        rgb = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")

        # Depth
        if "depth" in data:
            depth_bytes = base64.b64decode(data["depth"])
            depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(data["depth_shape"])
        else:
            depth = np.zeros((480, 640), dtype=np.uint16)

        return rgb, depth

    def _capture_orbbec(self) -> Image.Image:
        """Orbbec 拍照"""
        frames = self._pipeline.wait_for_frames(5000)
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("获取彩色帧失败")

        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        data = data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
        return Image.fromarray(data)

    def _capture_with_depth_orbbec(self) -> Tuple[Image.Image, np.ndarray]:
        """Orbbec 拍照 + 深度"""
        frames = self._pipeline.wait_for_frames(5000)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            raise RuntimeError("获取帧失败")

        # RGB
        color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        color_data = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
        rgb = Image.fromarray(color_data)

        # Depth (mm)
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

        return rgb, depth_data

    def _capture_cv2(self) -> Image.Image:
        """OpenCV 拍照（占位）"""
        import cv2
        ret, frame = self._cv2_cap.read()
        if not ret:
            raise RuntimeError("CV2 拍照失败")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    # ================================================================
    # 清理
    # ================================================================

    def close(self):
        """释放相机资源"""
        if HAS_ORBBEC and self._pipeline:
            self._pipeline.stop()
        elif HAS_CV2 and hasattr(self, '_cv2_cap'):
            self._cv2_cap.release()
        self._initialized = False

    def __del__(self):
        self.close()
