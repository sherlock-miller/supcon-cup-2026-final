"""
任务3：几何体无序分拣
====================
流程：
1. 机械臂移动到俯拍位置（正对散放区域）
2. Gemini335 拍照
3. CLIP 识别每个几何体的形状
4. 逐个抓取并放入对应形状的槽位

决赛更新说明确认：
- 几何体全部竖直摆放（无需姿态调整，统一从上往下抓取）
- 官方不告知形状种类 → CLIP 零样本分类覆盖常见几何体
- 位置标定贴纸可自备（建议打印 4 个 ArUco 码辅助定位）
"""
import logging
from typing import Tuple
import time

import numpy as np

from config import SHAPE_SLOTS, SHAPE_LABELS

logger = logging.getLogger(__name__)


def execute_shape_task(arm, hand, vision) -> Tuple[bool, str]:
    """
    执行几何体无序分拣任务。

    返回 (ok, message)
    """
    try:
        # 步骤 1: 检查硬件连接 + 电机使能
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
        photo_pos = SHAPE_SLOTS["photo_position"]
        logger.info(f"移动到俯拍位置: ({photo_pos['x']}, {photo_pos['y']}, {photo_pos['z']})")
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

        # 步骤 4: 拍照 + 形状识别
        logger.info("拍照...")
        image, depth_map = vision.capture_with_depth()

        logger.info("识别几何体形状...")
        shapes = vision.detect_and_classify_shapes(image)
        logger.info(f"识别结果: {shapes}")

        if len(shapes) == 0:
            return False, "未识别到任何几何体"

        # 步骤 5: 逐个抓取分拣
        pick_area = SHAPE_SLOTS["pick_area"]
        place_slots = SHAPE_SLOTS["place_slots"]

        successful = 0
        for i, shape_info in enumerate(shapes):
            shape = shape_info["shape"]
            logger.info(f"--- 处理第 {i+1}/{len(shapes)} 个: {shape} ---")

            # 检查目标槽位是否存在
            if shape not in place_slots:
                logger.warning(f"形状 '{shape}' 没有对应槽位，跳过")
                continue

            target_slot = place_slots[shape]

            pick_point = _estimate_pick_point(
                shape_info=shape_info,
                depth_map=depth_map,
                vision=vision,
                arm=arm,
            )
            if pick_point is None:
                logger.warning(f"形状 '{shape}' 无有效深度/坐标，跳过")
                continue

            pick_x, pick_y, pick_z = pick_point
            approach_z = min(0.52, max(pick_area["z_approach"], pick_z + 0.05))

            # 放置坐标
            place_x = target_slot["x"]
            place_y = target_slot["y"]
            place_z = target_slot["z"]

            try:
                # 5a. 移到几何体上方
                logger.info(f"接近几何体: ({pick_x:.3f}, {pick_y:.3f}, {approach_z:.3f})")
                arm.move_linear(x=pick_x, y=pick_y, z=approach_z, speed=0.12)

                # 5b. 下降抓取
                hand.release()
                logger.info(f"下降抓取: ({pick_x:.3f}, {pick_y:.3f}, {pick_z:.3f})")
                arm.move_linear(x=pick_x, y=pick_y, z=pick_z, speed=0.08)

                # 5c. 根据形状选择合适的抓取策略
                if "圆柱" in shape or "球" in shape:
                    hand.grasp_object("cylinder")
                else:
                    hand.grasp_object("cube")
                time.sleep(0.3)

                # 5d. 上升
                arm.move_linear(x=pick_x, y=pick_y, z=approach_z, speed=0.08)

                # 5e. 移到目标槽位上方
                logger.info(f"移动到 {shape} 槽位上方: ({place_x:.3f}, {place_y:.3f}, {place_z + 0.05:.3f})")
                arm.move_linear(x=place_x, y=place_y, z=place_z + 0.05, speed=0.12)

                # 5f. 下降放置
                logger.info(f"放置: ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})")
                arm.move_linear(x=place_x, y=place_y, z=place_z, speed=0.08)

                # 5g. 释放
                hand.release()
                time.sleep(0.2)

                # 5h. 上升
                arm.move_linear(x=place_x, y=place_y, z=place_z + 0.05, speed=0.08)

                successful += 1
                logger.info(f"{shape} 分拣完成")

            except Exception as e:
                logger.error(f"处理 {shape} 失败: {e}")
                try:
                    hand.release()
                    arm.move_to_safe_height()
                except Exception:
                    pass

        # 步骤 6: 回到安全位置
        arm.move_to_safe_height()

        if successful == len(shapes):
            return True, f"任务3完成: {successful}/{len(shapes)} 个几何体全部成功分拣"
        elif successful > 0:
            return True, f"任务3部分完成: {successful}/{len(shapes)} 个几何体成功分拣"
        else:
            return False, "任务3失败: 所有几何体分拣均失败"

    except Exception as e:
        logger.error(f"任务3异常: {e}", exc_info=True)
        try:
            hand.release()
            arm.move_to_safe_height()
        except Exception:
            pass
        return False, f"任务3异常: {type(e).__name__}: {str(e)[:200]}"


def _estimate_pick_point(shape_info, depth_map, vision, arm):
    """从检测框区域提取稳健深度，并转换到机械臂基座坐标。"""
    from config import ARM_WORKSPACE_Y, ARM_WORKSPACE_Z

    if depth_map is None or getattr(depth_map, "size", 0) == 0:
        return None

    img_h, img_w = depth_map.shape[:2]
    cx = int(round(float(shape_info.get("cx", img_w / 2))))
    cy = int(round(float(shape_info.get("cy", img_h / 2))))

    bbox = shape_info.get("bbox")
    if bbox:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(x1 + 1, min(img_w, x2))
        y2 = max(y1 + 1, min(img_h, y2))
    else:
        half = 12
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(img_w, cx + half + 1)
        y2 = min(img_h, cy + half + 1)

    roi = depth_map[y1:y2, x1:x2]
    valid_depths = roi[(roi > 0) & (roi <= 5000)]
    if valid_depths.size == 0:
        half = 6
        px1 = max(0, cx - half)
        py1 = max(0, cy - half)
        px2 = min(img_w, cx + half + 1)
        py2 = min(img_h, cy + half + 1)
        patch = depth_map[py1:py2, px1:px2]
        valid_depths = patch[(patch > 0) & (patch <= 5000)]
        if valid_depths.size == 0:
            return None

    depth_val = float(np.median(valid_depths))
    arm_pose = arm.get_pose().get("pose")
    target_x, target_y, target_z = vision.pixel_to_arm_coord(
        cx, cy, depth_val, arm_pose=arm_pose
    )
    target_y = max(ARM_WORKSPACE_Y[0], min(ARM_WORKSPACE_Y[1], target_y))
    target_z = max(ARM_WORKSPACE_Z[0], min(ARM_WORKSPACE_Z[1], target_z))
    return float(target_x), float(target_y), float(target_z)
