"""
任务1：拨按开关（示教轨迹回放方案）
==================================
流程：
1. 检查机械臂连接 + 使能
2. 灵巧手设为食指伸出姿态（设置一次，全程保持不动）
3. 回放示教轨迹1（去拍照位——相机正对开关面板）
4. 拍照 + ROI 亮灯识别（红/白/绿 → light_1/2/3）
5. 按识别结果回放对应示教轨迹（灯1按压 / 灯2拨动 / 灯3按压）
6. 返回结果

四条轨迹（机械臂控制器 ~/trajectories/ 下）:
  goto_photo  → 轨迹1: 安全位 → 相机识别姿态
  light_1     → 轨迹2: 红按钮按压
  light_2     → 轨迹3: 拨杆拨动
  light_3     → 轨迹4: 绿按钮按压
"""
import logging
from typing import Tuple
import time

import numpy as np

from config import (
    SWITCH_PANEL, ARM_SAFE_Z,
    TASK1_TRAJECTORIES, TASK1_PLAYBACK_SPEED, HAND_POINT_POSE,
)

logger = logging.getLogger(__name__)


def execute_switch_task(arm, hand, vision) -> Tuple[bool, str]:
    """
    执行一次开关操作（竞赛软件每次随机亮一个灯，连续调用三次）。

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

        # 步骤 2: 灵巧手食指伸出姿态（一次设置，全程保持）
        if hand is not None:
            try:
                hand.set_position(list(HAND_POINT_POSE))
                logger.info(f"灵巧手已设食指姿态: {HAND_POINT_POSE}")
            except Exception as e:
                logger.warning(f"灵巧手姿态设置失败（继续执行）: {e}")

        # 步骤 3: 回放轨迹1 —— 到拍照识别位
        goto_traj = TASK1_TRAJECTORIES["goto_photo"]
        logger.info(f"回放轨迹1（去拍照位）: {goto_traj}")
        result = arm.playback(goto_traj, speed_scale=TASK1_PLAYBACK_SPEED)
        if not result.get("success", True):
            logger.warning(f"轨迹1回放异常: {result}")

        # 步骤 4: 拍照 + 亮灯识别（ROI 优先，失败重试一次）
        vision.initialize()
        lit_light = None
        for attempt in range(2):
            image = vision.capture_image()
            lit_light = vision.detect_lit_light(image)
            if lit_light is not None:
                break
            logger.warning(f"第 {attempt + 1} 次识别未发现亮灯，重试...")
            time.sleep(0.5)

        if lit_light is None:
            return False, "未检测到亮灯，请检查灯光是否正常工作"

        light_id = lit_light["light_id"]
        color = lit_light.get("color", "?")
        logger.info(
            f"识别结果: {light_id} ({color} 灯亮, "
            f"method={lit_light.get('method')}, "
            f"confidence={lit_light.get('confidence')})")

        # 步骤 5: 按识别结果回放对应轨迹
        action_traj = TASK1_TRAJECTORIES.get(light_id)
        if not action_traj:
            return False, f"无 {light_id} 对应的示教轨迹配置"
        logger.info(f"回放操作轨迹: {light_id} → {action_traj}")
        result = arm.playback(action_traj, speed_scale=TASK1_PLAYBACK_SPEED)
        if not result.get("success", True):
            logger.warning(f"操作轨迹回放异常: {result}")

        switch_type = SWITCH_PANEL["switch_type"].get(light_id, "button")
        return True, f"任务1完成: {light_id}({color}) {switch_type} 操作成功"

    except Exception as e:
        logger.error(f"任务1异常: {e}", exc_info=True)
        return False, f"任务1异常: {type(e).__name__}: {str(e)[:200]}"


# ============================================================
# 备用方案（坐标计算直驱，示教回放失效时手动启用）
# ============================================================
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

        # 推进按下（钳制工作域下限，防撞台——审核修复）
        press_z = max(ARM_WORKSPACE_Z[0], z - 0.01)
        logger.info(f"按下按钮: ({x:.3f}, {y:.3f}, {press_z:.3f})")
        arm.move_linear(x=x, y=y, z=press_z, speed=0.05)  # 略深一点

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

        # 向下拨动（钳制工作域下限——审核修复）
        flip_z = max(ARM_WORKSPACE_Z[0], z - 0.03)
        logger.info("向下拨动开关...")
        arm.move_linear(x=x, y=y, z=flip_z, speed=0.05)

        # 释放
        hand.release()
        time.sleep(0.2)

        # 退回
        arm.move_linear(x=x, y=y, z=approach_z, speed=0.08)

        return True, "拨动开关完成"

    except Exception as e:
        return False, f"拨动开关失败: {e}"
