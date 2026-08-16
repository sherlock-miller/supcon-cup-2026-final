"""
模拟硬件 — 用于本地开发测试
===========================
当没有真实机械臂/灵巧手/相机时，
提供 Mock 实现来测试整体流程。
"""
import logging
import time
from typing import Dict, Any, Tuple, List, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger("mock")


class MockArmClient:
    """模拟机械臂客户端"""

    def __init__(self):
        self._x = 0.275
        self._y = -0.16
        self._z = 0.48
        self._enabled = True
        self._moving = False

    def check_connection(self) -> bool:
        return True

    def enable(self):
        self._enabled = True
        logger.info("[MOCK] 机械臂使能")

    def disable(self):
        self._enabled = False
        logger.info("[MOCK] 机械臂失能")

    def get_status(self) -> Dict:
        return {"moving": self._moving, "right_joints": {}}

    def get_pose(self) -> Dict:
        return {
            "arm": "right",
            "pose": {"x": self._x, "y": self._y, "z": self._z}
        }

    def is_moving(self) -> bool:
        return self._moving

    def wait_until_idle(self, timeout=30, interval=0.3):
        time.sleep(0.1)

    def move_linear(self, x, y, z, roll=None, pitch=None, yaw=None,
                    speed=0.12, plan_only=False, check_workspace=True):
        self._moving = True
        logger.info(f"[MOCK] 直线运动 → ({x:.3f}, {y:.3f}, {z:.3f}) speed={speed}")
        time.sleep(0.5)  # 模拟运动时间
        self._x, self._y, self._z = x, y, z
        self._moving = False
        return {"success": True, "message": "Cartesian execution finished for right_arm"}

    def move_to_safe_height(self, speed=0.2):
        logger.info(f"[MOCK] 提到安全高度")
        self._z = 0.52
        time.sleep(0.3)

    def move_joints(self, joints, speed=0.2):
        logger.info(f"[MOCK] 关节运动 → {[f'{j:.2f}' for j in joints]}")
        time.sleep(0.5)
        return {"success": True, "message": "Joint motion executed"}

    def move_home(self):
        logger.info("[MOCK] 回 home 位姿")
        return self.move_joints([0.0, 0.5, 0.0, -1.0, -0.1, -1.0, 0.0])

    def emergency_stop(self):
        logger.info("[MOCK] 紧急停止")


class MockHandClient:
    """模拟灵巧手客户端"""

    def __init__(self):
        self._position = 0.0

    def check_connection(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return True

    def set_position(self, positions=None, value=0.0):
        self._position = value if positions is None else (sum(positions) / len(positions))
        logger.info(f"[MOCK] 灵巧手位置 → {self._position:.2f}")
        return {"success": True}

    def grasp(self, strength=0.6):
        self._position = strength
        logger.info(f"[MOCK] 灵巧手抓取 (strength={strength})")

    def release(self):
        self._position = 0.0
        logger.info("[MOCK] 灵巧手张开")

    def close(self):
        self._position = 1.0
        logger.info("[MOCK] 灵巧手完全闭合")

    def grasp_object(self, object_type="cube"):
        logger.info(f"[MOCK] 灵巧手按 {object_type} 策略抓取")


class MockVisionManager:
    """模拟视觉模块"""

    def __init__(self):
        self._initialized = False

    def initialize(self):
        self._initialized = True
        logger.info("[MOCK] 视觉模型初始化完成")

    def capture_image(self) -> Image.Image:
        logger.info("[MOCK] 拍照")
        # 生成模拟图像
        img = Image.new("RGB", (640, 480), (50, 50, 50))
        return img

    def capture_with_depth(self) -> Tuple[Image.Image, np.ndarray]:
        rgb = self.capture_image()
        depth = np.ones((480, 640), dtype=np.uint16) * 1000  # 1m
        return rgb, depth

    def detect_lit_light(self, image) -> Optional[Dict]:
        """模拟：随机返回一个亮灯"""
        import random
        light_id = random.choice(["light_1", "light_2", "light_3"])
        switch_types = {"light_1": "button", "light_2": "toggle", "light_3": "button"}
        logger.info(f"[MOCK] 检测到亮灯: {light_id}")
        return {
            "light_id": light_id,
            "switch_type": switch_types[light_id],
            "pixel": (320, 240),
        }

    def detect_cube_numbers(self, image) -> List[Dict]:
        """模拟：返回四个数字"""
        logger.info("[MOCK] 识别长方体数字: 1, 2, 3, 4")
        return [
            {"number": 1, "cx": 160, "cy": 240, "raw_text": "1"},
            {"number": 2, "cx": 320, "cy": 240, "raw_text": "2"},
            {"number": 3, "cx": 480, "cy": 240, "raw_text": "3"},
            {"number": 4, "cx": 160, "cy": 360, "raw_text": "4"},
        ]

    def detect_and_classify_shapes(self, image) -> List[Dict]:
        """模拟：返回四个几何体"""
        logger.info("[MOCK] 识别几何体: 长方体, 正方体, 圆柱体, 多面体")
        return [
            {"shape": "长方体", "cx": 160, "cy": 240, "confidence": 0.95},
            {"shape": "正方体", "cx": 320, "cy": 240, "confidence": 0.92},
            {"shape": "圆柱体", "cx": 480, "cy": 240, "confidence": 0.88},
            {"shape": "多面体", "cx": 320, "cy": 360, "confidence": 0.85},
        ]

    def pixel_to_arm_coord(self, px, py, depth, arm_pose=None):
        return (0.275, -0.16, 0.48)
