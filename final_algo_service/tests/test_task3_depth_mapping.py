import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tasks.task3_shapes import execute_shape_task


class RecordingArm:
    def __init__(self):
        self.moves = []
        self.safe_height_calls = 0

    def check_connection(self):
        return True

    def enable(self):
        return None

    def wait_until_idle(self):
        return None

    def get_pose(self):
        return {
            "pose": {
                "x": 0.275,
                "y": -0.16,
                "z": 0.48,
                "roll": -3.141,
                "pitch": -1.552,
                "yaw": 3.141,
            }
        }

    def move_linear(self, x, y, z, roll=None, pitch=None, yaw=None, speed=0.12):
        self.moves.append((round(x, 3), round(y, 3), round(z, 3), speed))
        return {"success": True}

    def move_to_safe_height(self):
        self.safe_height_calls += 1


class RecordingHand:
    def __init__(self):
        self.actions = []

    def release(self):
        self.actions.append("release")

    def grasp_object(self, object_type="cube"):
        self.actions.append(f"grasp:{object_type}")


class DepthVision:
    def __init__(self, depth_map, pick_point):
        self.depth_map = depth_map
        self.pick_point = pick_point
        self.capture_with_depth_called = 0
        self.pixel_calls = []

    def initialize(self):
        return None

    def capture_with_depth(self):
        self.capture_with_depth_called += 1
        return Image.new("RGB", (640, 480), (0, 0, 0)), self.depth_map

    def capture_image(self):
        raise AssertionError("任务3不应再走 capture_image")

    def detect_and_classify_shapes(self, image):
        return [{
            "shape": "长方体",
            "cx": 100,
            "cy": 120,
            "bbox": [90, 110, 130, 150],
            "confidence": 0.95,
        }]

    def pixel_to_arm_coord(self, px, py, depth, arm_pose=None):
        self.pixel_calls.append((px, py, depth, arm_pose))
        return self.pick_point


def test_task3_uses_depth_and_transformed_pick_point():
    arm = RecordingArm()
    hand = RecordingHand()
    depth_map = np.zeros((480, 640), dtype=np.uint16)
    depth_map[110:150, 90:130] = 1234
    vision = DepthVision(depth_map, (0.311, -0.141, 0.467))

    ok, _ = execute_shape_task(arm=arm, hand=hand, vision=vision)

    assert ok is True
    assert vision.capture_with_depth_called == 1
    assert vision.pixel_calls
    assert vision.pixel_calls[0][2] == 1234.0
    assert (0.311, -0.141, 0.52, 0.12) in arm.moves
    assert (0.311, -0.141, 0.467, 0.08) in arm.moves
    assert "grasp:cube" in hand.actions


def test_task3_skips_shape_without_valid_depth():
    arm = RecordingArm()
    hand = RecordingHand()
    depth_map = np.zeros((480, 640), dtype=np.uint16)
    vision = DepthVision(depth_map, (0.4, -0.1, 0.5))

    ok, msg = execute_shape_task(arm=arm, hand=hand, vision=vision)

    assert ok is False
    assert "所有几何体分拣均失败" in msg
    assert vision.capture_with_depth_called == 1
    assert vision.pixel_calls == []
    assert hand.actions == []
