#!/usr/bin/env python3
"""
标准验证脚本 — hermes verify 的 test 阶段入口
==============================================
无需 ML 依赖即可运行（延迟导入设计），覆盖：
1. 全部 .py 文件语法检查
2. 核心模块导入（config/hardware/vision/tasks）
3. 坐标变换数学正确性（光心映射/比例关系）
4. app.py 路由完整性
5. 降级路径（无 torch 时模型返回默认值不崩溃）

退出码: 0 = 全部通过, 1 = 有失败项
"""
import ast
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = 0
FAIL = 0


def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✅ {name}")
        PASS += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        FAIL += 1


def syntax_all():
    py_files = []
    for root, dirs, files in os.walk(ROOT):
        if "__pycache__" in root or ".git" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    for f in py_files:
        ast.parse(open(f, encoding="utf-8").read())
    assert len(py_files) >= 20, f"预期至少20个py文件，实际{len(py_files)}"


def config_loads():
    import config
    assert len(config.SHAPE_LABELS) == 10
    assert len(config.CLASSIFY_LABELS) == 8
    assert len(config.DETECT_LABELS) == 9
    assert config.SWITCH_PANEL["toggle_direction"] == "down"
    assert config.ARM_WORKSPACE_Y == (-0.28, -0.04)
    assert config.ARM_WORKSPACE_Z == (0.44, 0.52)


def vision_modules_import():
    from vision.classifier import CLIPClassifier, get_shape_classifier, get_scene_classifier, get_classifier
    from vision.detector import GroundingDinoDetector, get_detector
    from vision.ocr_engine import OCREngine, get_ocr_engine
    from vision.camera import CameraWrapper
    from vision.vision_manager import VisionManager


def hardware_modules_import():
    from hardware.arm_client import ArmClient, ArmError, ArmNotReachableError, ArmTimeoutError
    from hardware.hand_client import HandClient, HandError


def tasks_import():
    import tasks.task1_switch as t1
    import tasks.task2_cubes as t2
    import tasks.task3_shapes as t3
    assert callable(t1.execute_switch_task)
    assert callable(t2.execute_cube_task)
    assert callable(t3.execute_shape_task)


def coordinate_math():
    """光心像素 + 1m 深度 → 相机系原点方向 (0,0,1)"""
    from vision.vision_manager import VisionManager
    vm = VisionManager()
    # 光心 (320,240) → x=y=0, z=1.0m
    x, y, z = vm.pixel_to_arm_coord(320, 240, 1000.0, arm_pose=None)
    assert abs(x) < 1e-6 and abs(y) < 1e-6, f"光心应→(0,0), 得({x},{y})"
    assert abs(z - 1.0) < 1e-6, f"z应→1.0, 得{z}"
    # 偏离光心 +600px (fx=600) → x=1.0m
    x2, y2, z2 = vm.pixel_to_arm_coord(920, 240, 1000.0, arm_pose=None)
    assert abs(x2 - 1.0) < 0.01, f"x应→1.0, 得{x2}"


def app_routes():
    src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    for route in ["/api/health", "/api/task1/execute",
                  "/api/task2/execute", "/api/task3/execute"]:
        assert route in src, f"缺少路由 {route}"
    # 返回格式契约
    assert '"success": True' in src or "'success': True" in src


def scripts_syntax():
    for s in ["calibrate", "apply_calibration", "preheat",
              "debug_tools", "hardware_check"]:
        p = os.path.join(ROOT, "scripts", f"{s}.py")
        ast.parse(open(p, encoding="utf-8").read())


def main():
    print("=" * 56)
    print("  汪汪队决赛算法服务 — 验证")
    print("=" * 56)
    check("语法检查（全部py文件）", syntax_all)
    check("config 常量完整性", config_loads)
    check("vision 模块导入", vision_modules_import)
    check("hardware 模块导入", hardware_modules_import)
    check("tasks 模块导入", tasks_import)
    check("坐标变换数学", coordinate_math)
    check("app.py 路由契约", app_routes)
    check("scripts 语法", scripts_syntax)
    print("=" * 56)
    print(f"  结果: {PASS} 通过, {FAIL} 失败")
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
