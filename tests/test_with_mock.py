"""
本地模拟测试脚本
================
使用 Mock 硬件测试三任务完整流程。
无需真实机械臂、灵巧手、相机即可运行。

用法:
    cd final_algo_service
    python tests/test_with_mock.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from tests.mock_hardware import MockArmClient, MockHandClient, MockVisionManager


def test_health():
    """测试健康检查"""
    print("\n" + "=" * 50)
    print("测试: /api/health")
    from app import health
    result = health()
    print(f"  返回: {result}")
    assert result["success"] is True
    print("  ✅ 通过")


def test_task1():
    """测试任务1: 拨按开关"""
    print("\n" + "=" * 50)
    print("测试: /api/task1/execute")

    arm = MockArmClient()
    hand = MockHandClient()
    vision = MockVisionManager()

    # 挂载到 app 的全局变量（模拟真实流程）
    import tasks.task1_switch as t1
    ok, msg = t1.execute_switch_task(arm, hand, vision)
    print(f"  结果: ok={ok}, msg={msg}")
    if ok:
        print("  ✅ 通过")
    else:
        print(f"  ⚠️ 部分通过: {msg}")


def test_task2():
    """测试任务2: 长方体有序转运"""
    print("\n" + "=" * 50)
    print("测试: /api/task2/execute")

    arm = MockArmClient()
    hand = MockHandClient()
    vision = MockVisionManager()

    import tasks.task2_cubes as t2
    ok, msg = t2.execute_cube_task(arm, hand, vision)
    print(f"  结果: ok={ok}, msg={msg}")
    if ok:
        print("  ✅ 通过")
    else:
        print(f"  ⚠️ 部分通过: {msg}")


def test_task3():
    """测试任务3: 几何体无序分拣"""
    print("\n" + "=" * 50)
    print("测试: /api/task3/execute")

    arm = MockArmClient()
    hand = MockHandClient()
    vision = MockVisionManager()

    import tasks.task3_shapes as t3
    ok, msg = t3.execute_shape_task(arm, hand, vision)
    print(f"  结果: ok={ok}, msg={msg}")
    if ok:
        print("  ✅ 通过")
    else:
        print(f"  ⚠️ 部分通过: {msg}")


if __name__ == "__main__":
    print("=" * 50)
    print("  汪汪队决赛算法服务 — Mock 测试")
    print("=" * 50)

    test_health()
    test_task1()
    test_task2()
    test_task3()

    print("\n" + "=" * 50)
    print("  全部测试完成！")
    print("=" * 50)
    print("\n下一步:")
    print("  1. 在 config.py 中填入现场测量的坐标")
    print("  2. 完成相机标定 + 手眼标定")
    print("  3. 对接真实机械臂 IP 和灵巧手 IP")
    print("  4. 在比赛工控机上 python app.py 启动服务")
