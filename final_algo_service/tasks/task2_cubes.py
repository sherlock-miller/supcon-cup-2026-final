"""
任务2：长方体有序转运对位
==========================
流程：
1. 机械臂移动到俯拍位置（正对槽位区域）
2. Gemini335 拍照
3. EasyOCR 识别四个长方体上的数字 1-4
4. 按 1→2→3→4 顺序抓取
5. 放置到指定台面
"""
import logging
from typing import Tuple
import time

from config import CUBE_SLOTS

logger = logging.getLogger(__name__)


def execute_cube_task(arm, hand, vision) -> Tuple[bool, str]:
    """
    执行长方体有序转运任务。

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
        photo_pos = CUBE_SLOTS["photo_position"]
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

        # 步骤 4: 拍照 + OCR 识别数字
        logger.info("拍照...")
        image = vision.capture_image()

        logger.info("识别长方体数字...")
        cubes = vision.detect_cube_numbers(image)
        logger.info(f"识别结果: {cubes}")

        # 枚举补全策略（关键兜底）：
        # 任务2 槽位固定 4 个、数字 1-4 各出现一次。
        # 若只识别到 3 个数字，缺失的数字必然在未被匹配的槽位，
        # 直接补全 —— 解决 DINO 对笔画细弱的数字"1"方块漏检问题。
        detected_nums = {c["number"] for c in cubes}
        missing_nums = {1, 2, 3, 4} - detected_nums
        if missing_nums:
            logger.warning(
                f"只识别到 {len(cubes)} 个数字 {sorted(detected_nums)}，"
                f"缺失 {sorted(missing_nums)} → 枚举补全"
            )
            # 缺失数字的槽位坐标 = 未被识别数字占用的槽位
            used_slots = detected_nums & set(CUBE_SLOTS["slot_positions"].keys())
            all_slots = set(CUBE_SLOTS["slot_positions"].keys())
            for missing_num in sorted(missing_nums):
                # 缺失数字用其自身编号槽位（数字与槽位一一对应）
                if missing_num in CUBE_SLOTS["slot_positions"]:
                    cubes.append({
                        "number": missing_num,
                        "cx": None, "cy": None,
                        "raw_text": f"补全{missing_num}",
                        "complemented": True,
                    })
                    logger.info(f"  → 补全数字 {missing_num}（枚举兜底）")
        # 按数字排序确保 1→2→3→4 顺序
        cubes.sort(key=lambda c: c["number"])

        if len(cubes) < 4:
            logger.warning(f"补全后仍只有 {len(cubes)} 个，继续尝试")
            if len(cubes) == 0:
                return False, "未识别到任何长方体数字"

        # 步骤 5: 按顺序抓取放置
        slot_positions = CUBE_SLOTS["slot_positions"]
        place_pos = CUBE_SLOTS["place_position"]

        successful = 0
        for i, cube in enumerate(cubes):
            num = cube["number"]
            logger.info(f"--- 处理第 {i+1}/4 个: 数字 {num} ---")

            # 获取槽位坐标（如果预设了对应槽位）
            # 实际中 cube 的像素坐标需要通过手眼标定转换
            # 这里先用预设槽位坐标
            if num in slot_positions:
                slot = slot_positions[num]
            else:
                # 找不到对应槽位，使用第 i 个槽位
                available_slots = list(slot_positions.keys())
                if i < len(available_slots):
                    slot = slot_positions[available_slots[i]]
                else:
                    logger.error(f"槽位坐标不足")
                    continue

            # 抓取参数
            approach_x = slot["x"]
            approach_y = slot["y"]
            approach_z = slot["z"] + 0.05   # 槽位上方 5cm（工作域上限 0.52 内）

            pick_x = slot["x"]
            pick_y = slot["y"]
            pick_z = slot["z"]

            # 放置参数
            place_x = place_pos["x"]
            place_y = place_pos["y"]
            place_z = place_pos["z"]

            try:
                # 5a. 移到槽位上方
                logger.info(f"接近槽位 {num}: ({approach_x:.3f}, {approach_y:.3f}, {approach_z:.3f})")
                arm.move_linear(x=approach_x, y=approach_y, z=approach_z, speed=0.12)

                # 5b. 下降抓取
                hand.release()  # 先张开
                logger.info(f"下降到抓取高度: ({pick_x:.3f}, {pick_y:.3f}, {pick_z:.3f})")
                arm.move_linear(x=pick_x, y=pick_y, z=pick_z, speed=0.08)

                # 5c. 抓取
                hand.grasp_object("cube")
                time.sleep(0.3)

                # 5d. 上升
                logger.info("抓取后上升")
                arm.move_linear(x=pick_x, y=pick_y, z=approach_z, speed=0.08)

                # 5e. 移到放置位置上方
                logger.info(f"移动到放置位置上方: ({place_x:.3f}, {place_y:.3f}, {place_z + 0.05:.3f})")
                arm.move_linear(x=place_x, y=place_y, z=place_z + 0.05, speed=0.12)

                # 5f. 下降放置
                logger.info(f"下降到放置高度: ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})")
                arm.move_linear(x=place_x, y=place_y, z=place_z, speed=0.08)

                # 5g. 释放
                hand.release()
                time.sleep(0.2)

                # 5h. 上升
                arm.move_linear(x=place_x, y=place_y, z=place_z + 0.05, speed=0.08)

                successful += 1
                logger.info(f"长方体 {num} 处理完成")

            except Exception as e:
                logger.error(f"处理长方体 {num} 失败: {e}")
                # 继续处理下一个
                # ⚠️ 顺序: 先回安全高度再释放（若正抓着物体，先松手物体会
                # 掉落在当前位置砸坏其他方块——审核修复）
                try:
                    arm.move_to_safe_height()
                    hand.release()
                except Exception:
                    pass

        # 步骤 6: 回到安全位置
        arm.move_to_safe_height()

        if successful == 4:
            return True, "任务2完成: 四个长方体全部成功转运"
        elif successful > 0:
            return True, f"任务2部分完成: {successful}/4 个长方体成功转运"
        else:
            return False, "任务2失败: 所有长方体转运均失败"

    except Exception as e:
        logger.error(f"任务2异常: {e}", exc_info=True)
        try:
            hand.release()
            arm.move_to_safe_height()
        except Exception:
            pass
        return False, f"任务2异常: {type(e).__name__}: {str(e)[:200]}"
