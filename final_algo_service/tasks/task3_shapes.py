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
        image = vision.capture_image()

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

            # 抓取位置（基于像素坐标估算 + 占位深度）
            # TODO: 现场需用深度相机获取实际 Z
            cx = shape_info.get("cx", 320)
            cy = shape_info.get("cy", 240)

            # 占位坐标转换
            pick_x = pick_area["x_range"][0] + (cx / 640) * (
                pick_area["x_range"][1] - pick_area["x_range"][0]
            )
            pick_y = pick_area["y_range"][0] + (cy / 480) * (
                pick_area["y_range"][1] - pick_area["y_range"][0]
            )
            pick_z = pick_area["z_pick"]
            approach_z = pick_area["z_approach"]

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
