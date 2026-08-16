#!/usr/bin/env python3
"""
硬件自检脚本
============
开赛前运行，全面检查硬件连接状态。

用法: python hardware_check.py
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hwcheck")

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_all():
    results = []
    print("\n" + "=" * 60)
    print("  汪汪队决赛 — 硬件自检")
    print("=" * 60)

    # 1. 机械臂
    print("\n[1/5] 机械臂 (FTArm B9)...")
    try:
        from hardware.arm_client import ArmClient
        arm = ArmClient()
        if arm.check_connection():
            status = arm.get_status()
            joints = status.get("right_joints")
            if joints:
                print(f"  ✅ 机械臂在线, {len(joints)} 个关节就绪")
            else:
                print("  ⚠️  机械臂在线但关节未就绪")
            results.append(("机械臂", "OK" if joints else "WARN"))
        else:
            print("  ❌ 机械臂连接失败")
            results.append(("机械臂", "FAIL"))
    except Exception as e:
        print(f"  ❌ 检查异常: {e}")
        results.append(("机械臂", "FAIL"))

    # 2. 灵巧手
    print("\n[2/5] 灵巧手...")
    try:
        from hardware.hand_client import HandClient
        hand = HandClient()
        if hand.check_connection():
            print("  ✅ 灵巧手在线")
            results.append(("灵巧手", "OK"))
        else:
            print("  ⚠️  灵巧手未响应（请确认 HTTP 桥接已启动）")
            results.append(("灵巧手", "WARN"))
    except Exception as e:
        print(f"  ❌ 检查异常: {e}")
        results.append(("灵巧手", "FAIL"))

    # 3. 相机
    print("\n[3/5] Gemini335 相机...")
    try:
        from vision.camera import CameraWrapper, HAS_ORBBEC
        cam = CameraWrapper()
        cam.initialize()
        img = cam.capture()
        print(f"  ✅ 相机正常, 分辨率 {img.size}, 后端: {'Orbbec SDK' if HAS_ORBBEC else 'OpenCV'}")
        results.append(("相机", "OK"))
    except Exception as e:
        print(f"  ⚠️  相机初始化失败: {e}")
        results.append(("相机", "WARN"))

    # 4. 视觉模型
    print("\n[4/5] 视觉模型...")
    try:
        from vision.classifier import get_classifier
        clf = get_classifier()
        if clf.model is not None:
            print("  ✅ CLIP 分类器就绪")
        else:
            print("  ❌ CLIP 加载失败")
            results.append(("CLIP", "FAIL"))
    except Exception as e:
        print(f"  ⚠️  CLIP 检查跳过: {e}")

    try:
        from vision.ocr_engine import get_ocr_engine
        ocr = get_ocr_engine()
        if ocr.available:
            print("  ✅ EasyOCR 就绪")
        else:
            print("  ⚠️  EasyOCR 不可用（数字识别降级）")
    except Exception:
        pass

    # 5. 算法服务
    print("\n[5/5] 算法服务 (本机 5000 端口)...")
    try:
        import requests
        r = requests.get("http://127.0.0.1:5000/api/health", timeout=3)
        if r.status_code == 200:
            print(f"  ✅ 服务在线: {r.json()}")
        else:
            print(f"  ⚠️  服务返回 {r.status_code}")
    except Exception:
        print("  ⚠️  本机服务未启动（比赛中由竞赛软件直接调用，此检查仅供参考）")

    # 汇总
    print("\n" + "=" * 60)
    print("  自检汇总:")
    for name, status in results:
        icon = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(status, "?")
        print(f"  {icon} {name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    check_all()
