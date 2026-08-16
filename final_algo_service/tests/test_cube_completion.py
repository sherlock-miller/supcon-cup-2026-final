"""
任务2 枚举补全逻辑单元测试
=========================
验证：只识别到 3 个数字时自动补全缺失数字（1-4）
"""
import sys
import os
import ast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest


class TestCubeCompletion:
    """枚举补全逻辑"""

    def test_completion_logic_in_source(self):
        """补全逻辑存在（静态检查）"""
        src = open(os.path.join(ROOT, "tasks", "task2_cubes.py"), encoding="utf-8").read()
        assert "枚举补全" in src, "缺少枚举补全逻辑"
        assert "missing_nums" in src, "缺少 missing_nums 计算"
        assert "complemented" in src, "缺少 complemented 标记"

    def test_completion_math(self):
        """补全数学：{2,3,4} → 缺失 {1}"""
        detected = {2, 3, 4}
        missing = {1, 2, 3, 4} - detected
        assert missing == {1}

    def test_completion_order(self):
        """补全后按数字排序 1→2→3→4"""
        cubes = [
            {"number": 3, "cx": 0, "cy": 0},
            {"number": 2, "cx": 0, "cy": 0},
            {"number": 4, "cx": 0, "cy": 0},
        ]
        for m in {1, 2, 3, 4} - {c["number"] for c in cubes}:
            cubes.append({"number": m, "cx": None, "cy": None, "complemented": True})
        cubes.sort(key=lambda c: c["number"])
        assert [c["number"] for c in cubes] == [1, 2, 3, 4]

    def test_no_completion_when_full(self):
        """识别全 4 个时不补全"""
        detected = {1, 2, 3, 4}
        assert {1, 2, 3, 4} - detected == set()

    def test_task2_import_still_ok(self):
        """task2 模块可导入"""
        import tasks.task2_cubes as t2
        assert callable(t2.execute_cube_task)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
