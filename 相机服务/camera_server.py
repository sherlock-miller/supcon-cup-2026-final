"""
相机采集服务 — 宿主机独立进程（Win11 现场）
============================================
Docker 容器无法直接访问 USB 相机（Windows Docker Desktop 限制），
因此相机服务作为独立进程跑在宿主机上，通过 HTTP 对外提供图像。

架构:
  算法服务(Docker 容器) ──HTTP──> 相机服务(宿主机 :5002) ──USB──> Gemini335

用法（宿主机 Windows 原生 Python）:
  pip install fastapi uvicorn opencv-python pyorbbecsdk
  python camera_server.py

接口:
  GET  /health      → {"success": true}
  GET  /capture     → {"success": true, "rgb": "base64...", "depth": "base64..."}
  GET  /capture_rgb → {"success": true, "image": "base64..."}
"""
import base64
import io
import logging
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("camera-server")

app = FastAPI(title="汪汪队相机服务", version="1.0.0")

# ============================================================
# 相机初始化
# ============================================================
_camera = None


def get_camera():
    """懒加载相机（Orbbec SDK 优先，OpenCV 兜底）"""
    global _camera
    if _camera is None:
        # 尝试 Orbbec Gemini335
        try:
            import pyorbbecsdk as obs
            ctx = obs.Context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                raise RuntimeError("未检测到 Orbbec 设备")

            class OrbbecCam:
                def __init__(self):
                    self.pipeline = obs.Pipeline(devices[0])
                    config = obs.Config()
                    config.enable_video_stream(
                        obs.OB_STREAM_COLOR, 640, 480, 30, obs.OB_FORMAT_RGB888)
                    config.enable_video_stream(
                        obs.OB_STREAM_DEPTH, 640, 480, 30, obs.OB_FORMAT_Y16)
                    self.pipeline.start(config)

                def capture(self):
                    frames = self.pipeline.wait_for_frames(3000)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if color_frame is None:
                        raise RuntimeError("获取彩色帧失败")
                    rgb = np.frombuffer(
                        color_frame.get_data(), dtype=np.uint8
                    ).reshape((color_frame.get_height(), color_frame.get_width(), 3)).copy()
                    depth = None
                    if depth_frame is not None:
                        depth = np.frombuffer(
                            depth_frame.get_data(), dtype=np.uint16
                        ).reshape((depth_frame.get_height(), depth_frame.get_width())).copy()
                    return rgb, depth

            _camera = OrbbecCam()
            logger.info("Orbbec Gemini335 相机就绪")

        except ImportError:
            logger.warning("pyorbbecsdk 未安装，使用 OpenCV")
            import cv2
            class CV2Cam:
                def __init__(self):
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        raise RuntimeError("无法打开相机")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                def capture(self):
                    ret, frame = self.cap.read()
                    if not ret:
                        raise RuntimeError("拍照失败")
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None

            _camera = CV2Cam()
            logger.info("OpenCV 相机就绪（注意：无深度！）")

        except Exception as e:
            logger.error(f"相机初始化失败: {e}")
            raise
    return _camera


def encode_image(img: np.ndarray) -> str:
    """numpy 图像 → base64 JPEG"""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_depth(depth: np.ndarray) -> str:
    """深度图 → base64 (原始 uint16 小端字节流)"""
    return base64.b64encode(depth.tobytes()).decode("ascii")


# ============================================================
# API
# ============================================================
@app.get("/health")
def health():
    return {"success": True, "message": "camera ready"}


@app.get("/capture")
def capture():
    """拍照：返回 RGB + Depth"""
    try:
        cam = get_camera()
        rgb, depth = cam.capture()
        result = {"success": True, "rgb": encode_image(rgb)}
        if depth is not None:
            result["depth"] = encode_depth(depth)
            result["depth_shape"] = list(depth.shape)
        return result
    except Exception as e:
        logger.error(f"拍照失败: {e}")
        return {"success": False, "message": str(e)}


@app.get("/capture_rgb")
def capture_rgb():
    """只拍 RGB"""
    try:
        cam = get_camera()
        rgb, _ = cam.capture()
        return {"success": True, "image": encode_image(rgb)}
    except Exception as e:
        return {"success": False, "message": str(e)}


if __name__ == "__main__":
    logger.info("汪汪队相机服务启动于 :5002")
    uvicorn.run(app, host="0.0.0.0", port=5002, log_level="info")
