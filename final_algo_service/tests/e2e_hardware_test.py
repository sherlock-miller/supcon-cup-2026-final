#!/usr/bin/env python3
"""
端到端控制程序验证
==================
Mock 视觉识别 + 真实硬件控制代码 → 模拟硬件服务器(8087/8088)

验证点:
1. 三任务完整执行不崩溃
2. 指令序列合理性（先使能→提安全高度→移动→抓取→...）
3. 全部位姿在安全工作域内（模拟服务器校验）
4. 灵巧手位置值合法（10维 0-1）
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["ARM_BASE_URL"] = "http://127.0.0.1:8087"
os.environ["HAND_BASE_URL"] = "http://127.0.0.1:8088"
os.environ["CAMERA_BASE_URL"] = "http://127.0.0.1:9999"  # 相机也 mock

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

from unittest.mock import patch, MagicMock

# Mock 视觉识别结果
def mock_capture_with_depth():
    import numpy as np
    img = MagicMock()
    depth = np.full((480, 640), 0.5, dtype=np.float32)
    return img, depth

def mock_detect_lit_light(image):
    return {"light_id": "light_1", "switch_type": "toggle",
            "pixel": (320.0, 240.0), "color": "red",
            "confidence": 0.95, "method": "hsv"}

def mock_pixel_to_arm_coord(px, py, depth, arm_pose=None):
    # 简化: 像素→机械臂坐标占位变换（在家验证流程用）
    x = 0.30 + (px - 320) * 0.0005
    y = -0.16 + (240 - py) * 0.0005
    z = float(depth) if isinstance(depth, (int, float)) else 0.48
    return x, y, z

def mock_detect_switch_type(image, pixel):
    return "toggle"  # 拨杆

def mock_detect_cube_numbers(image):
    return [
        {"number": 1, "cx": 0.3, "cy": 0.5, "conf": 0.9, "bbox": [200, 300, 400, 500]},
        {"number": 2, "cx": 0.4, "cy": 0.5, "conf": 0.9, "bbox": [280, 300, 480, 500]},
        {"number": 3, "cx": 0.5, "cy": 0.5, "conf": 0.9, "bbox": [360, 300, 560, 500]},
        {"number": 4, "cx": 0.6, "cy": 0.5, "conf": 0.9, "bbox": [440, 300, 640, 500]},
    ]

def mock_detect_shapes(image):
    return [
        {"shape": "长方体", "confidence": 0.9, "bbox": [100, 200, 300, 400], "cx": 0.25, "cy": 0.5},
        {"shape": "正方体", "confidence": 0.9, "bbox": [250, 200, 450, 400], "cx": 0.45, "cy": 0.5},
    ]

patches = [
    patch("vision.vision_manager.VisionManager.capture_with_depth", mock_capture_with_depth),
    patch("vision.vision_manager.VisionManager.detect_lit_light", mock_detect_lit_light),
    patch("vision.vision_manager.VisionManager.pixel_to_arm_coord", mock_pixel_to_arm_coord),
    patch("vision.vision_manager.VisionManager.detect_cube_numbers", mock_detect_cube_numbers),
    patch("vision.vision_manager.VisionManager.detect_and_classify_shapes", mock_detect_shapes),
]

results = {}
for p in patches:
    p.start()

from tasks import task1_switch, task2_cubes, task3_shapes
from hardware.arm_client import ArmClient
from hardware.hand_client import HandClient

# 真实硬件 client（指向模拟服务器）+ Mock 视觉
arm = ArmClient(base_url="http://127.0.0.1:8087")
hand = HandClient(base_url="http://127.0.0.1:8088")
vision = MagicMock()
vision.capture_with_depth = mock_capture_with_depth
vision.detect_lit_light = mock_detect_lit_light
vision.pixel_to_arm_coord = mock_pixel_to_arm_coord
vision.detect_cube_numbers = mock_detect_cube_numbers
vision.detect_and_classify_shapes = mock_detect_shapes

print("=" * 60)
print("任务1: 拨按开关")
print("=" * 60)
try:
    ok1, msg1 = task1_switch.execute_switch_task(arm=arm, hand=hand, vision=vision)
    results["task1"] = {"ok": ok1, "message": str(msg1)[:80]}
    print(f"  结果: ok={ok1} msg={str(msg1)[:60]}")
except Exception as e:
    results["task1"] = {"ok": False, "exception": str(e)[:120]}
    print(f"  ❌ 异常: {e}")

print()
print("=" * 60)
print("任务2: 长方体有序转运")
print("=" * 60)
try:
    ok2, msg2 = task2_cubes.execute_cube_task(arm=arm, hand=hand, vision=vision)
    results["task2"] = {"ok": ok2, "message": str(msg2)[:80]}
    print(f"  结果: ok={ok2} msg={str(msg2)[:60]}")
except Exception as e:
    results["task2"] = {"ok": False, "exception": str(e)[:120]}
    print(f"  ❌ 异常: {e}")

print()
print("=" * 60)
print("任务3: 几何体无序分拣")
print("=" * 60)
try:
    ok3, msg3 = task3_shapes.execute_shape_task(arm=arm, hand=hand, vision=vision)
    results["task3"] = {"ok": ok3, "message": str(msg3)[:80]}
    print(f"  结果: ok={ok3} msg={str(msg3)[:60]}")
except Exception as e:
    results["task3"] = {"ok": False, "exception": str(e)[:120]}
    print(f"  ❌ 异常: {e}")

for p in patches:
    p.stop()

print()
print("=" * 60)
print("端到端执行汇总")
print("=" * 60)
for k, v in results.items():
    status = "✅" if v.get("ok") else ("⚠️ " + v.get("message", "")[:30])
    print(f"  {k}: {status}")

# 统计指令日志
log_file = "/tmp/fake_hardware_log.jsonl"
if os.path.exists(log_file):
    lines = [json.loads(l) for l in open(log_file, encoding="utf-8")]
    arm_posts = [l for l in lines if l["who"] == "arm" and l["method"] == "POST"]
    hand_posts = [l for l in lines if l["who"] == "hand" and l["method"] == "POST"]
    failed = [l for l in lines if l.get("body", {}).get("success") is False]
    print(f"\n指令统计: 机械臂POST={len(arm_posts)}, 灵巧手POST={len(hand_posts)}, 总请求={len(lines)}")
    print(f"机械臂指令序列:")
    for l in arm_posts:
        b = l["body"]
        if l["path"] == "/api/end_effector":
            r = b.get("right", {})
            print(f"  [{l['ts']}] end_effector → ({r.get('x'):.3f}, {r.get('y'):.3f}, {r.get('z'):.3f})")
        else:
            print(f"  [{l['ts']}] {l['path']} {json.dumps(b, ensure_ascii=False)[:60]}")
    print(f"灵巧手指令序列:")
    for l in hand_posts:
        b = l["body"]
        pos = b.get("position", [])
        print(f"  [{l['ts']}] {l['path']} position={[f'{p:.1f}' for p in pos]}")
