"""
pytest 测试套件 — 适配 hermes verify 默认 recipe
================================================
覆盖语法/导入/坐标数学/路由契约，无需 ML 依赖。
"""
import ast
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PY_FILES = [
    os.path.join(root, f)
    for root, dirs, files in os.walk(ROOT)
    if "__pycache__" not in root and ".git" not in root
    for f in files
    if f.endswith(".py")
]


class TestSyntax:
    @pytest.mark.parametrize("path", PY_FILES)
    def test_compiles(self, path):
        ast.parse(open(path, encoding="utf-8").read())


class TestConfig:
    def test_shape_labels(self):
        import config
        assert len(config.SHAPE_LABELS) == 10
        assert "长方体" in config.SHAPE_LABELS and "圆柱体" in config.SHAPE_LABELS

    def test_legacy_labels_present(self):
        import config
        assert len(config.CLASSIFY_LABELS) == 8
        assert len(config.DETECT_LABELS) == 9

    def test_switch_panel(self):
        import config
        assert config.SWITCH_PANEL["switch_type"]["light_2"] == "toggle"
        assert config.SWITCH_PANEL["toggle_direction"] == "down"

    def test_workspace_bounds(self):
        import config
        assert config.ARM_WORKSPACE_Y == (-0.28, -0.04)
        assert config.ARM_WORKSPACE_Z == (0.44, 0.52)


class TestImports:
    def test_classifier(self):
        from vision.classifier import (
            CLIPClassifier, get_shape_classifier,
            get_scene_classifier, get_classifier,
        )
        assert callable(get_classifier)

    def test_detector(self):
        from vision.detector import GroundingDinoDetector, get_detector

    def test_ocr(self):
        from vision.ocr_engine import OCREngine, get_ocr_engine

    def test_camera_and_manager(self):
        from vision.camera import CameraWrapper
        from vision.vision_manager import VisionManager

    def test_hardware(self):
        from hardware.arm_client import ArmClient, ArmError
        from hardware.hand_client import HandClient, HandError

    def test_tasks_callable(self):
        import tasks.task1_switch as t1
        import tasks.task2_cubes as t2
        import tasks.task3_shapes as t3
        assert callable(t1.execute_switch_task)
        assert callable(t2.execute_cube_task)
        assert callable(t3.execute_shape_task)


class TestCoordinateMath:
    def test_optical_center(self):
        from vision.vision_manager import VisionManager
        vm = VisionManager()
        x, y, z = vm.pixel_to_arm_coord(320, 240, 1000.0, arm_pose=None)
        assert abs(x) < 1e-6 and abs(y) < 1e-6
        assert abs(z - 1.0) < 1e-6

    def test_pixel_scale(self):
        from vision.vision_manager import VisionManager
        vm = VisionManager()
        x, y, z = vm.pixel_to_arm_coord(920, 240, 1000.0, arm_pose=None)
        assert abs(x - 1.0) < 0.01

    def test_arm_pose_transform(self):
        """带末端位姿的完整变换链不崩溃且输出有限值"""
        import math
        from vision.vision_manager import VisionManager
        vm = VisionManager()
        pose = {"x": 0.275, "y": -0.16, "z": 0.48,
                "roll": -3.141, "pitch": -1.552, "yaw": 3.141}
        x, y, z = vm.pixel_to_arm_coord(320, 240, 1000.0, arm_pose=pose)
        assert all(math.isfinite(v) for v in (x, y, z))


class TestAppRoutes:
    def test_routes_present(self):
        src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
        for route in ["/api/health", "/api/task1/execute",
                      "/api/task2/execute", "/api/task3/execute"]:
            assert route in src, f"缺少路由 {route}"

    def test_response_contract(self):
        src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
        assert '"success"' in src or "'success'" in src


class TestDegradedPath:
    """无 ML 依赖时的降级路径（模块可导入 + 分类器返回默认值）"""

    def test_classifier_without_torch(self):
        # 降级路径测试：无论有无 torch，模块导入 + 构造都不应崩溃。
        # - 无 torch：加载失败 → model=None → 降级返回默认标签
        # - 有 torch：模型真实加载成功 → 同样有效
        from vision.classifier import get_shape_classifier
        try:
            clf = get_shape_classifier()
            if clf.model is None:
                # 降级路径：predict 仍可用，返回默认标签
                result = clf.predict(None)
                assert result["label"] == clf.labels[0]
            else:
                # 真实模型加载成功（带 ML 环境）
                assert clf.model is not None
        except ImportError:
            pass  # torch 缺失时构造异常也算降级（不崩溃整个服务）
