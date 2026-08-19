"""
任务2：长方体有序转运（示教轨迹回放方案）
==========================================
流程：
1. 检查机械臂连接 + 使能
2. 灵巧手设为位姿1
3. 回放轨迹1（初始位 → 相机识别位）
4. 拍照 + 识别从左到右的数字顺序（如 2143）
5. 按识别顺序依次执行每个数字的轨迹组：
     轨迹N-1(approach 去抓取位) → 手位姿2 → 轨迹N-2(grasp 抓取)
     → 手位姿1 → 轨迹N-3(return 回归初始位)
   每组之间回放轨迹1（回相机识别位；第一组已在识别位，跳过）
6. 完成

轨迹组（机械臂控制器 trajectories 目录）:
  goto_photo        轨迹1: 初始位 → 相机识别位
  {N}_approach      轨迹N-1: 拍照位 → 抓取位
  {N}_grasp         轨迹N-2: 抓取动作
  {N}_return        轨迹N-3: 回归初始位
"""
import logging
from typing import Tuple, List, Optional
import time

from config import (
    TASK2_PLAYBACK_SPEED, HAND_POSE_TASK2_1, HAND_POSE_TASK2_2,
    task2_traj_path,
)

logger = logging.getLogger(__name__)


def _get_cube_order(vision, image) -> Optional[List[int]]:
    """识别从左到右的数字顺序。

    返回 [2, 1, 4, 3] 形式的执行顺序（第一个=最左边）。
    识别到 3 个时枚举补全缺失数字放在末尾（位置未知, 保守处理）。
    识别 <3 个返回 None（调用方重拍）。
    """
    cubes = vision.detect_cube_numbers(image) or []
    with_pos = [c for c in cubes if c.get("cx") is not None]

    if len(with_pos) >= 4:
        # 全识别: 按像素 x 从左到右排序
        with_pos.sort(key=lambda c: c["cx"])
        order = [int(c["number"]) for c in with_pos[:4]]
        logger.info(f"识别顺序(左→右): {order}")
        return order

    if len(with_pos) == 3:
        # 缺 1 个: 缺失数字放末尾（位置未知）
        nums = {int(c["number"]) for c in with_pos}
        missing = sorted({1, 2, 3, 4} - nums)
        with_pos.sort(key=lambda c: c["cx"])
        order = [int(c["number"]) for c in with_pos] + missing
        logger.warning(
            f"只识别到 3 个数字 {sorted(nums)}，缺失 {missing} "
            f"→ 顺序近似为 {order}（缺失数字放末尾）")
        return order

    return None


def execute_cube_task(arm, hand, vision) -> Tuple[bool, str]:
    """
    执行长方体有序转运任务（竞赛软件调用一次，完成全部 4 个转运）。

    Returns:
        (ok, message)
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

        # 步骤 2: 灵巧手位姿1
        if hand is not None:
            try:
                hand.set_position(list(HAND_POSE_TASK2_1))
                logger.info(f"灵巧手位姿1: {HAND_POSE_TASK2_1}")
            except Exception as e:
                logger.warning(f"位姿1设置失败（继续执行）: {e}")

        # 步骤 3: 回放轨迹1 → 相机识别位
        goto_traj = task2_traj_path("goto_photo")
        logger.info(f"回放轨迹1（去识别位）: {goto_traj}")
        result = arm.playback(goto_traj, speed_scale=TASK2_PLAYBACK_SPEED)
        if not result.get("success", True):
            logger.warning(f"轨迹1回放异常: {result}")

        # 步骤 4: 拍照 + 识别从左到右数字顺序（失败重拍一次）
        vision.initialize()
        order = None
        for attempt in range(2):
            image = vision.capture_image()
            order = _get_cube_order(vision, image)
            if order is not None:
                break
            logger.warning(f"第 {attempt + 1} 次识别失败，重试...")
            time.sleep(0.5)

        if order is None:
            return False, "数字顺序识别失败（识别到不足 3 个数字）"
        order_str = "".join(str(d) for d in order)
        logger.info(f"执行顺序: {order_str}（左→右）")

        # 步骤 5: 按顺序执行轨迹组
        for i, digit in enumerate(order):
            logger.info(f"--- 第 {i + 1}/4 个: 数字 {digit} ---")

            # 组间回轨迹1（第一组已在识别位）
            if i > 0:
                logger.info(f"回放轨迹1（回识别位）: {goto_traj}")
                arm.playback(goto_traj, speed_scale=TASK2_PLAYBACK_SPEED)

            group = {
                "approach": task2_traj_path("approach", digit),
                "grasp": task2_traj_path("grasp", digit),
                "return": task2_traj_path("return", digit),
            }
            if not all(group.values()):
                return False, f"数字 {digit} 的轨迹组配置不完整: {group}"

            # 轨迹N-1: 去抓取位
            logger.info(f"回放轨迹{digit}-1(approach): {group['approach']}")
            arm.playback(group["approach"], speed_scale=TASK2_PLAYBACK_SPEED)

            # 手位姿2 → 轨迹N-2 抓取
            if hand is not None:
                hand.set_position(list(HAND_POSE_TASK2_2))
                logger.info(f"灵巧手位姿2（抓取）: {HAND_POSE_TASK2_2}")
            logger.info(f"回放轨迹{digit}-2(grasp): {group['grasp']}")
            arm.playback(group["grasp"], speed_scale=TASK2_PLAYBACK_SPEED)

            # 手位姿1 → 轨迹N-3 回归初始位
            if hand is not None:
                hand.set_position(list(HAND_POSE_TASK2_1))
                logger.info("灵巧手位姿1（恢复）")
            logger.info(f"回放轨迹{digit}-3(return): {group['return']}")
            arm.playback(group["return"], speed_scale=TASK2_PLAYBACK_SPEED)

        return True, f"任务2完成: 按顺序 {order_str} 转运 4 个长方体"

    except Exception as e:
        logger.error(f"任务2异常: {e}", exc_info=True)
        return False, f"任务2异常: {type(e).__name__}: {str(e)[:200]}"
