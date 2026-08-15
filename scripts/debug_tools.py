#!/usr/bin/env python3
"""
现场调试工具集 — 交互式控制台
=============================
比赛现场调试用。功能：
  1. 机械臂状态查看 + Jog 手动控制
  2. 灵巧手测试
  3. 相机拍照测试
  4. 记录关键位姿（用于填写 config.py）
  5. 模拟竞赛软件调用测试

用法:
  python debug_tools.py
"""
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("debug")

sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.arm_client import ArmClient
from hardware.hand_client import HandClient


def print_menu():
    print("""
╔══════════════════════════════════════════╗
║        汪汪队决赛 — 现场调试工具          ║
╠══════════════════════════════════════════╣
║  1. 机械臂状态         6. 相机拍照测试    ║
║  2. 读取末端位姿(记录坐标) 7. 灵巧手张开   ║
║  3. Jog 直线移动        8. 灵巧手闭合     ║
║  4. Jog 关节移动        9. 灵巧手抓取测试 ║
║  5. 机械臂使能/失能     10. 回安全位      ║
║  11. 模拟竞赛软件调用   0. 退出           ║
╚══════════════════════════════════════════╝
""")


def cmd_arm_status(arm):
    status = arm.get_status()
    pose = arm.get_pose()
    print(f"\n机械臂状态:")
    print(f"  运动中: {status.get('moving', False)}")
    print(f"  MoveIt: {status.get('moveit_available', True)}")
    joints = status.get("right_joints")
    if joints:
        for name, val in joints.items():
            print(f"  {name}: {val:.4f} rad")
    p = pose.get("pose")
    if p:
        print(f"\n末端位姿 (基坐标系):")
        print(f"  x={p['x']:.4f} m, y={p['y']:.4f} m, z={p['z']:.4f} m")
        print(f"  roll={p['roll']:.4f}, pitch={p['pitch']:.4f}, yaw={p['yaw']:.4f} rad")
        print(f"\n  >>> 填入 config.py 用: x={p['x']:.3f}, y={p['y']:.3f}, z={p['z']:.3f}, "
              f"roll={p['roll']:.3f}, pitch={p['pitch']:.3f}, yaw={p['yaw']:.3f}")


def cmd_jog_linear(arm):
    pose = arm.get_pose().get("pose", {})
    x = float(input(f"  X (当前 {pose.get('x', 0.275):.3f}): ") or pose.get("x", 0.275))
    y = float(input(f"  Y (当前 {pose.get('y', -0.16):.3f}): ") or pose.get("y", -0.16))
    z = float(input(f"  Z (当前 {pose.get('z', 0.48):.3f}): ") or pose.get("z", 0.48))
    speed = float(input(f"  速度 (默认 0.12): ") or 0.12)
    confirm = input(f"  确认移动到 ({x:.3f}, {y:.3f}, {z:.3f})? [y/N]: ")
    if confirm.lower() == 'y':
        arm.move_linear(x=x, y=y, z=z, speed=speed)


def cmd_jog_joints(arm):
    print("  输入7个关节角度 (rad, 空格分隔):")
    print("  例: 0.0 0.5 0.0 -1.0 -0.1 -1.0 0.0")
    try:
        joints = [float(v) for v in input("  > ").split()]
        if len(joints) == 7:
            arm.move_joints(joints)
        else:
            print("  需要 7 个角度")
    except ValueError:
        print("  输入格式错误")


def cmd_arm_power(arm):
    choice = input("  1=使能 2=失能(⚠️重力下坠) > ")
    if choice == '1':
        arm.enable()
    elif choice == '2':
        confirm = input("  ⚠️ 确认失能? 手臂会下坠! [y/N]: ")
        if confirm.lower() == 'y':
            arm.disable()


def cmd_camera_test():
    from vision.camera import CameraWrapper
    cam = CameraWrapper()
    cam.initialize()
    img = cam.capture()
    print(f"\n相机测试:")
    print(f"  图像尺寸: {img.size}")
    save_path = Path(__file__).parent.parent / "现场配置" / "camera_test.png"
    save_path.parent.mkdir(exist_ok=True)
    img.save(save_path)
    print(f"  已保存: {save_path}")


def cmd_hand_test(hand, action):
    if action == 'open':
        hand.release()
        print("  灵巧手已张开")
    elif action == 'close':
        hand.close()
        print("  灵巧手已闭合")
    elif action == 'grasp':
        strength = float(input("  抓取力度 0-1 (默认 0.6): ") or 0.6)
        hand.grasp(strength=strength)
        print(f"  已抓取 (力度 {strength})")


def cmd_simulate_competition():
    """模拟竞赛软件调用"""
    import requests
    base = input(f"  Base URL (默认 http://127.0.0.1:5000): ") or "http://127.0.0.1:5000"
    print(f"\n  健康检查:")
    try:
        r = requests.get(f"{base}/api/health", timeout=5)
        print(f"    HTTP {r.status_code}: {r.json()}")
    except Exception as e:
        print(f"    ❌ {e}")

    task = input("  调用哪个任务? (1/2/3): ")
    if task in ('1', '2', '3'):
        print(f"  调用 /api/task{task}/execute ...")
        try:
            r = requests.post(f"{base}/api/task{task}/execute", json={}, timeout=120)
            print(f"    HTTP {r.status_code}: {r.json()}")
        except Exception as e:
            print(f"    ❌ {e}")


def main():
    arm = ArmClient()
    hand = HandClient()

    while True:
        print_menu()
        choice = input("选择操作 > ").strip()

        try:
            if choice == '1':
                cmd_arm_status(arm)
            elif choice == '2':
                cmd_arm_status(arm)
            elif choice == '3':
                cmd_jog_linear(arm)
            elif choice == '4':
                cmd_jog_joints(arm)
            elif choice == '5':
                cmd_arm_power(arm)
            elif choice == '6':
                cmd_camera_test()
            elif choice == '7':
                cmd_hand_test(hand, 'open')
            elif choice == '8':
                cmd_hand_test(hand, 'close')
            elif choice == '9':
                cmd_hand_test(hand, 'grasp')
            elif choice == '10':
                arm.move_to_safe_height()
                print("  已回安全高度")
            elif choice == '11':
                cmd_simulate_competition()
            elif choice == '0':
                print("退出")
                break
            else:
                print("未知选项")
        except Exception as e:
            logger.error(f"操作失败: {e}")

        print()


if __name__ == "__main__":
    main()
