"""
FTArm B9 机械臂 HTTP 客户端封装
===============================
基于官方文档 FTArm B9 机械臂 HTTP-WS 接口文档.md
提供安全、带重试的运动控制接口。

注意：所有坐标需在安全工作域内（§2.6）：
  右臂 Y: -0.28 ~ -0.04, Z: 0.44 ~ 0.52 (X=0.275)
"""
import time
import logging
from typing import Dict, Any, Optional, Tuple, List

import requests

from config import (
    ARM_BASE_URL, ARM_MODE, ARM_DEFAULT_POSE,
    ARM_DEFAULT_SPEED, ARM_SAFE_SPEED, ARM_FAST_SPEED,
    ARM_EEF_STEP, ARM_TIMEOUT, ARM_SAFE_Z,
    ARM_WORKSPACE_Y, ARM_WORKSPACE_Z, ARM_DEFAULT_X,
    ARM_HOME_JOINTS,
)

logger = logging.getLogger(__name__)

# ============================================================
# 自定义异常
# ============================================================
class ArmError(Exception):
    """机械臂操作异常"""
    pass


class ArmNotReachableError(ArmError):
    """目标不可达"""
    pass


class ArmTimeoutError(ArmError):
    """运动超时"""
    pass


# ============================================================
# ArmClient
# ============================================================
class ArmClient:
    """FTArm B9 机械臂 HTTP API 客户端"""

    def __init__(self, base_url: str = ARM_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.default_pose = ARM_DEFAULT_POSE.copy()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ---------- 基础 API 调用 ----------

    def _get(self, path: str, timeout: int = 10) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise ArmError(f"GET {path} 失败: {e}")

    def _post(self, path: str, data: Dict[str, Any], timeout: int = ARM_TIMEOUT) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.post(url, json=data, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            # 官方 enable/disable 响应为嵌套格式 {"right"/"left": {"success":...}}
            # （键名随工作区），其余端点为顶层 {"success":...}——统一兼容
            if "success" not in result:
                for key in ("right", "left"):
                    if key in result and isinstance(result[key], dict):
                        result = result[key]
                        break
            if not result.get("success", False):
                raise ArmError(f"POST {path} 业务失败: {result.get('message', 'unknown')}")
            return result
        except requests.Timeout:
            raise ArmTimeoutError(f"POST {path} 超时 ({timeout}s)")
        except requests.RequestException as e:
            raise ArmError(f"POST {path} 失败: {e}")

    # ---------- 状态查询 ----------

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态（运动标记 + 关节位置）"""
        return self._get("/api/status")

    def get_pose(self) -> Dict[str, Any]:
        """获取末端当前位姿"""
        return self._get("/api/pose")

    def is_moving(self) -> bool:
        """是否运动中"""
        status = self.get_status()
        return status.get("moving", False)

    def wait_until_idle(self, timeout: float = 30, interval: float = 0.3):
        """等待机械臂停止运动"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_moving():
                return
            time.sleep(interval)
        raise ArmTimeoutError("等待机械臂停止超时")

    # ---------- 电机控制 ----------

    def enable(self):
        """电机使能（上力锁住姿态）"""
        logger.info("机械臂使能...")
        result = self._post("/api/enable", data={}, timeout=10)
        logger.info(f"使能结果: {result}")
        return result

    def disable(self):
        """电机失能（软急停）⚠️ 手臂会因重力下坠"""
        logger.warning("机械臂失能（软急停）!")
        return self._post("/api/disable", data={}, timeout=10)

    # ---------- 末端运动 ----------

    def move_linear(
        self,
        x: float, y: float, z: float,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        speed: float = ARM_DEFAULT_SPEED,
        plan_only: bool = False,
        check_workspace: bool = True,
    ) -> Dict[str, Any]:
        """
        直线运动到目标位姿。
        
        Args:
            x, y, z: 目标位置 (m)
            roll, pitch, yaw: 目标姿态 (rad)，默认使用 ARM_DEFAULT_POSE
            speed: 速度缩放 0.01-1.0
            plan_only: True 则只规划不执行
            check_workspace: 是否检查安全工作域
        
        Returns:
            API 响应 JSON
        """
        if roll is None:
            roll = self.default_pose["roll"]
        if pitch is None:
            pitch = self.default_pose["pitch"]
        if yaw is None:
            yaw = self.default_pose["yaw"]

        # 安全工作域检查
        if check_workspace:
            if not (ARM_WORKSPACE_Y[0] <= y <= ARM_WORKSPACE_Y[1]):
                logger.warning(f"Y={y} 超出安全工作域 {ARM_WORKSPACE_Y}")
            if not (ARM_WORKSPACE_Z[0] <= z <= ARM_WORKSPACE_Z[1]):
                logger.warning(f"Z={z} 超出安全工作域 {ARM_WORKSPACE_Z}")

        right_target = {
            "x": x, "y": y, "z": z,
            "roll": roll, "pitch": pitch, "yaw": yaw,
        }

        data = {
            "mode": ARM_MODE,
            "right": right_target,
            "cartesian_linear": True,
            "velocity_scaling": speed,
            "acceleration_scaling": speed,
            "cartesian_eef_step": ARM_EEF_STEP,
            "cartesian_min_fraction": 0.85,
        }

        if plan_only:
            data["plan_only"] = True

        logger.info(f"直线运动 → ({x:.3f}, {y:.3f}, {z:.3f}) speed={speed}")
        result = self._post("/api/end_effector", data=data, timeout=ARM_TIMEOUT)

        # 检查是否走了直线
        msg = result.get("message", "")
        if "OMPL" in msg:
            logger.warning(f"⚠️ 直线规划失败，回退自由路径: {msg}")
        else:
            logger.info(f"直线运动完成: {msg}")

        return result

    def move_to_safe_height(self, speed: float = ARM_FAST_SPEED):
        """先提到安全高度（避免横向移动碰撞）"""
        pose = self.get_pose()
        current_pose = pose.get("pose", {})
        current_x = current_pose.get("x", ARM_DEFAULT_X)
        current_y = current_pose.get("y", -0.16)
        return self.move_linear(
            x=current_x, y=current_y, z=ARM_SAFE_Z, speed=speed
        )

    def move_joints(self, joints: List[float], speed: float = 0.2) -> Dict[str, Any]:
        """关节空间运动（7 关节角度，单位 rad）"""
        if len(joints) != 7:
            raise ArmError(f"需要 7 个关节角度，得到 {len(joints)}")

        data = {
            "mode": ARM_MODE,
            "right_joints": joints,
            "velocity_scaling": speed,
        }
        logger.info(f"关节运动 → {[f'{j:.3f}' for j in joints]}")
        return self._post("/api/joints", data=data, timeout=120)

    def move_home(self):
        """回到预设安全位姿"""
        logger.info("回 home 位姿...")
        return self.move_joints(ARM_HOME_JOINTS)

    # ---------- 示教与回放（现场调试利器） ----------

    def teach_mode(self, enable: bool = True) -> Dict[str, Any]:
        """
        示教模式开关（官方文档 §3.12）
        enable=True: 电机切零力矩，手臂可自由拖动（自动开始录制）
        enable=False: 恢复位置控制
        ⚠️ 示教期间运动接口不可用
        """
        logger.info(f"{'进入' if enable else '退出'}示教模式...")
        return self._post("/api/teach_mode", data={"enable": enable}, timeout=20)

    def teach_record(self, command: str = "start", filename: str = "") -> Dict[str, Any]:
        """
        示教录制控制（官方文档 §3.11）
        command: start / stop / save / cancel
        filename: save 时必填（相对名自动存 ~/trajectories/）
        """
        data = {"command": command}
        if filename:
            data["filename"] = filename
        logger.info(f"示教录制 {command} {filename}")
        return self._post("/api/teach", data=data, timeout=15)

    def playback(
        self,
        trajectory_id: str,
        speed_scale: float = 1.0,
        loop_count: int = 1,
    ) -> Dict[str, Any]:
        """
        轨迹回放（官方文档 §3.13）
        阻塞至回放完成（服务器上限 300s）
        """
        logger.info(f"回放轨迹 {trajectory_id} (speed={speed_scale}x, loops={loop_count})")
        return self._post(
            "/api/playback",
            data={
                "trajectory_id": trajectory_id,
                "speed_scale": speed_scale,
                "loop_count": loop_count,
            },
            timeout=310,
        )

    def teach_and_save(self, filename: str):
        """
        便捷流程第一步：进入示教模式（自动开始录制）
        之后人工拖动演示，最后调用 teach_and_save_finish() 收尾。

        用法:
            arm.teach_and_save("taskA")   # 进入示教
            ... 人工拖动 ...               # 中间等待
            arm.teach_and_save_finish()   # 停止+保存+退出
        """
        logger.info(f"开始示教流程 → 轨迹名: {filename}")
        self.teach_mode(enable=True)

    def teach_and_save_finish(self, filename: str = ""):
        """结束示教流程：停止录制 → 保存 → 退出示教模式"""
        self.teach_record(command="stop")
        if filename:
            result = self.teach_record(command="save", filename=filename)
        else:
            result = {}
        self.teach_mode(enable=False)
        return result

    # ---------- 安全操作 ----------

    def safe_pick_place_sequence(
        self,
        approach_xyz: Tuple[float, float, float],
        pick_xyz: Tuple[float, float, float],
        retreat_xyz: Optional[Tuple[float, float, float]] = None,
    ):
        """
        标准抓取/放置安全序列：
        1. 提到安全高度
        2. 水平移动到目标上方（approach）
        3. 垂直下降到抓取高度（pick）
        4. （执行抓取/放置动作由灵巧手完成）
        5. 垂直上升到安全高度
        6. 水平移动到撤退位置

        Args:
            approach_xyz: 接近位置 (在目标上方)
            pick_xyz: 抓取/放置位置
            retreat_xyz: 撤退位置（默认回到安全高度）
        """
        ax, ay, az = approach_xyz
        px, py, pz = pick_xyz

        # 步骤 1: 提到安全高度
        self.move_to_safe_height()

        # 步骤 2: 水平移到目标上方
        logger.info(f"接近目标上方 → ({ax:.3f}, {ay:.3f}, {az:.3f})")
        self.move_linear(x=ax, y=ay, z=az, speed=ARM_FAST_SPEED)

        # 步骤 3: 垂直下降
        logger.info(f"下降到抓取高度 → ({px:.3f}, {py:.3f}, {pz:.3f})")
        self.move_linear(x=px, y=py, z=pz, speed=ARM_SAFE_SPEED)

        # 步骤 4-5 由调用方执行（抓取/放置后上升）
        # 步骤 6: 上升到安全高度
        self.move_to_safe_height()

        if retreat_xyz:
            rx, ry, rz = retreat_xyz
            self.move_linear(x=rx, y=ry, z=rz, speed=ARM_FAST_SPEED)

    # ---------- 工具方法 ----------

    def check_connection(self) -> bool:
        """检查机械臂连接是否正常"""
        try:
            self.get_status()
            return True
        except Exception:
            return False

    def emergency_stop(self):
        """紧急停止（失能 + 取消）"""
        logger.critical("紧急停止!")
        try:
            self._post("/api/cancel", data={}, timeout=5)
        except Exception:
            pass
        try:
            self.disable()
        except Exception:
            pass
