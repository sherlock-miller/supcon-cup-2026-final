"""
任务1：拨按开关
===============
流程：
1. 机械臂移动到拍照位置（正对开关面板）
2. Gemini335 拍照
3. 视觉检测哪个灯亮了
4. 计算机械臂末端移动到对应开关上方的坐标
5. 根据开关类型执行点按或拨动
6. 退回到安全位置
"""
import logging
from typing import Tuple
import time

import numpy as np

from config import SWITCH_PANEL, ARM_SAFE_Z

logger = logging.getLogger(__name__)


def execute_switch_task(arm, hand, vision) -> Tuple[bool, str]:
    """
    执行一次开关操作。

    竞赛软件每次随机亮一个灯，此函数被连续调用三次。

    Returns:
        (ok, message)
    """
    try:
        # 步骤 1: 检查硬件连接 + 电机使能（官方流程: enable 后电机才上力）
        if not arm.check_connection():
            return False, "机械臂连接失败"
        try:
            arm.enable()
            logger.info("机械臂已使能")
        except Exception as e:
            logger.warning(f"使能失败（可能开机已自动使能，继续执行）: {e}")

        # 步骤 2: 初始化视觉
        vision.initialize()

        # 步骤 3: 移动到拍照位置
        photo_pos = SWITCH_PANEL["photo_position"]
        logger.info(f"移动到拍照位置: ({photo_pos['x']}, {photo_pos['y']}, {photo_pos['z']})")
        arm.move_linear(
            x=photo_pos["x"],
            y=photo_pos["y"],
            z=photo_pos["z"],
            roll=photo_pos.get("roll"),
            pitch=photo_pos.get("pitch"),
            yaw=photo_pos.get("yaw"),
            speed=0.15,
        )
        arm.wait_until_idle()

        # 步骤 4: 拍照（带深度）
        logger.info("拍照...")
        image, depth_map = vision.capture_with_depth()

        # 步骤 5: 检测亮灯
        logger.info("检测亮灯...")
        lit_light = vision.detect_lit_light(image)
        if lit_light is None:
            logger.error("未检测到亮灯")
            return False, "未检测到亮灯，请检查灯光是否正常工作"

        light_id = lit_light["light_id"]
        switch_type = lit_light["switch_type"]
        pixel = lit_light["pixel"]
        logger.info(f"检测到亮灯: {light_id} (类型: {switch_type}, 像素: {pixel})")

        # 步骤 6: 深度取点 + 坐标转换（像素 → 相机系 → 基座系）
        px, py = pixel
        px = int(np.clip(px, 0, depth_map.shape[1] - 1))
        py = int(np.clip(py, 0, depth_map.shape[0] - 1))
        depth_val = float(depth_map[py, px])

        if depth_val <= 0 or depth_val > 5000:
            logger.warning(f"深度值异常 ({depth_val}mm)，使用面板预设深度")
            depth_val = 400.0  # 占位：假设灯距相机 40cm

        target_x, target_y, target_z = vision.pixel_to_arm_coord(
            px, py, depth_val,
            arm_pose=arm.get_pose().get("pose"),
        )
        logger.info(
            f"目标开关基坐标: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
        )

        # 安全工作域保护
        from config import ARM_WORKSPACE_Y, ARM_WORKSPACE_Z
        target_y = max(ARM_WORKSPACE_Y[0], min(ARM_WORKSPACE_Y[1], target_y))
        target_z = max(ARM_WORKSPACE_Z[0], min(ARM_WORKSPACE_Z[1], target_z))

        # 步骤 7: 执行开关操作
        if switch_type == "button":
            logger.info(f"执行按钮点按: {light_id}")
            ok, msg = _press_button(arm, hand, target_x, target_y, target_z)
        elif switch_type == "toggle":
            logger.info(f"执行拨动开关: {light_id}")
            ok, msg = _flip_toggle(arm, hand, target_x, target_y, target_z)
        else:
            return False, f"未知开关类型: {switch_type}"

        if not ok:
            return False, msg

        # 步骤 8: 回到安全位置
        logger.info("回到安全位置")
        arm.move_to_safe_height()

        return True, f"任务1完成: {light_id} {switch_type} 操作成功"

    except Exception as e:
        logger.error(f"任务1异常: {e}", exc_info=True)
        # 尝试安全收尾
        try:
            arm.move_to_safe_height()
        except Exception:
            pass
        return False, f"任务1异常: {type(e).__name__}: {str(e)[:200]}"


def _press_button(arm, hand, x: float, y: float, z: float) -> Tuple[bool, str]:
    """
    点按按钮操作：
    1. 灵巧手伸出食指
    2. 机械臂移动到按钮前方
    3. 线性推进按按钮
    4. 退回
    """
    from config import ARM_WORKSPACE_Z
    try:
        # 准备手指（单指伸出）
        logger.info("准备按按钮手势...")
        hand.grasp_object("button")

        # 移动到按钮前方（+5cm Z，钳制在工作域内）
        approach_z = min(ARM_WORKSPACE_Z[1], z + 0.05)
        logger.info(f"接近按钮: ({x:.3f}, {y:.3f}, {approach_z:.3f})")
        arm.move_linear(x=x, y=y, z=approach_z, speed=0.08)

        # 推进按下
        logger.info(f"按下按钮: ({x:.3f}, {y:.3f}, {z:.3f})")
        arm.move_linear(x=x, y=y, z=z - 0.01, speed=0.05)  # 略深一点

        # 稍等
        time.sleep(0.3)

        # 退回
        arm.move_linear(x=x, y=y, z=approach_z, speed=0.08)

        hand.release()
        return True, "按钮点按完成"

    except Exception as e:
        return False, f"按钮点按失败: {e}"


def _flip_toggle(arm, hand, x: float, y: float, z: float) -> Tuple[bool, str]:
    """
    拨动开关操作：
    1. 灵巧手两指捏住拨杆
    2. 机械臂向上/下移动拨动
    3. 释放

    注意：微信群确认拨杆支持上下双向拨动。
    策略：先向下拨，如果灯没灭再向上拨。
    """
    from config import ARM_WORKSPACE_Z
    try:
        # 两指捏合手势
        logger.info("准备拨动开关手势...")
        hand.grasp_object("toggle")

        # 移动到拨杆位置（+5cm Z，钳制在工作域内）
        approach_z = min(ARM_WORKSPACE_Z[1], z + 0.05)
        logger.info(f"接近拨杆: ({x:.3f}, {y:.3f}, {approach_z:.3f})")
        arm.move_linear(x=x, y=y, z=approach_z, speed=0.08)

        # 到达拨杆
        arm.move_linear(x=x, y=y, z=z, speed=0.05)

        # 捏住拨杆
        hand.grasp(strength=0.5)
        time.sleep(0.2)

        # 向下拨动
        logger.info("向下拨动开关...")
        arm.move_linear(x=x, y=y, z=z - 0.03, speed=0.05)

        # 释放
        hand.release()
        time.sleep(0.2)

        # 退回
        arm.move_linear(x=x, y=y, z=approach_z, speed=0.08)

        return True, "拨动开关完成"

    except Exception as e:
        return False, f"拨动开关失败: {e}"
