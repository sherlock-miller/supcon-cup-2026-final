"""
灵巧手 HTTP 客户端封装
======================
基于官方《O10 灵巧手远程控制 API 参考手册》（7.29 更新版决赛附件）。

关键规格：
- 端口: 8088（官方文档 §1.2）
- 控制接口: POST /api/set_pos，请求体 {"position": float[10]}（0-1 归一化）
- 归一化语义（数学换算确认，官方文档 curl 注释有误）:
    position=1 → 关节最大弧度 → 弯曲/握拳
    position=0 → 关节最小弧度 → 伸展/张手
- 错误码: 5 位 bitmask（堵转/过热/过流/电机异常/通讯异常）
- WebSocket: ws://<IP>:8088/ws，50ms 状态推送

现场注意: 灵巧手 HTTP 桥接服务由组委会预装（赛台主机直连）。
"""
import logging
from typing import Dict, Any, Optional, List

import requests

from config import (
    HAND_BASE_URL,
    HAND_GRASP_CLOSE,
    HAND_GRASP_OPEN,
    HAND_GRASP_GENTLE,
)

logger = logging.getLogger(__name__)


class HandError(Exception):
    """灵巧手操作异常"""
    pass


class HandClient:
    """
    灵巧手 HTTP API 客户端（O10 规格）

    接口（基于官方文档）:
      GET  /api/status   — 设备完整状态
      GET  /api/pose     — 归一化位置
      GET  /api/errors   — 错误码 bitmask
      POST /api/set_pos  — 归一化位置控制 (0-1, 10 DOF)
      POST /api/set_pvc  — 弧度/速度/电流控制
    """

    def __init__(self, base_url: str = HAND_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._connected = False

    def _get(self, path: str, timeout: int = 5) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise HandError(f"GET {path} 失败: {e}")

    def _post(self, path: str, data: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.post(url, json=data, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            if not result.get("success", True):
                raise HandError(f"POST {path} 业务失败: {result.get('message', 'unknown')}")
            return result
        except requests.RequestException as e:
            raise HandError(f"POST {path} 失败: {e}")

    # ---------- 状态查询 ----------

    def get_status(self) -> Dict[str, Any]:
        """获取灵巧手设备状态"""
        return self._get("/api/status")

    def get_pose(self) -> Dict[str, Any]:
        """获取当前归一化位置"""
        return self._get("/api/pose")

    def get_errors(self) -> Dict[str, Any]:
        """获取错误码（5位bitmask）"""
        return self._get("/api/errors")

    def is_ready(self) -> bool:
        """检查灵巧手是否就绪"""
        try:
            status = self.get_status()
            return status.get("connected", False)
        except Exception:
            return False

    def check_errors(self) -> bool:
        """检查是否有关节错误（堵转/过热等）"""
        try:
            errors = self.get_errors()
            codes = errors.get("error_codes", [])
            return all(c == 0 for c in codes)
        except Exception:
            return True  # 查询失败不阻塞流程

    # ---------- 基本操作 ----------

    def set_position(
        self,
        positions: Optional[List[float]] = None,
        value: float = HAND_GRASP_OPEN,
    ) -> Dict[str, Any]:
        """
        设置归一化位置 (0-1)，10 自由度。

        简化接口：如果未指定具体手指位置，则所有手指使用 value。
        注意语义: value=0 全张手（伸展），value=1 全握拳（弯曲）。
        """
        if positions is None:
            # 所有手指统一位置
            positions = [value] * 10

        if len(positions) != 10:
            raise HandError(f"需要 10 个位置值 (0-1)，实际收到 {len(positions)}")

        data = {"position": positions}  # 官方字段名: position（单数）
        logger.info(f"灵巧手位置控制 → {[f'{p:.2f}' for p in positions[:5]]}...")
        return self._post("/api/set_pos", data=data, timeout=10)

    def grasp(self, strength: float = HAND_GRASP_GENTLE):
        """抓取（手指弯曲）"""
        logger.info(f"灵巧手抓取 (strength={strength})")
        return self.set_position(value=strength)

    def release(self):
        """释放（手指张开）"""
        logger.info("灵巧手张开")
        return self.set_position(value=HAND_GRASP_OPEN)

    def close(self):
        """完全握拳"""
        logger.info("灵巧手完全握拳")
        return self.set_position(value=HAND_GRASP_CLOSE)

    # ---------- 高级操作 ----------

    def grasp_object(self, object_type: str = "cube"):
        """
        根据物体类型执行合适的抓取策略。

        不同物体需要不同抓取力度和姿态：
        - cube (长方体/正方体): 平行抓取，力度适中
        - cylinder (圆柱体): 包络抓取
        - toggle (拨动开关): 两指捏合，轻拨
        - button (按钮): 单指伸出，点按
        """
        strategies = {
            "cube": {"value": 0.6, "description": "平行抓取"},
            "cylinder": {"value": 0.5, "description": "包络抓取"},
            "toggle": {"value": 0.3, "description": "两指捏合"},
            "button": {"value": 0.3, "description": "单指点按"},
        }
        strategy = strategies.get(object_type, strategies["cube"])
        logger.info(f"灵巧手 {strategy['description']} ({object_type})")
        return self.set_position(value=strategy["value"])

    # ---------- 安全 ----------

    def check_connection(self) -> bool:
        """检查连接"""
        try:
            self._get("/api/status")
            return True
        except Exception:
            return False
