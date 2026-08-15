"""
灵巧手 HTTP 客户端封装
======================
基于傅利叶 DexHand SDK 的 HTTP 桥接接口。
比赛现场灵巧手通过 HTTP API 控制（非直连 SDK）。
"""
import logging
from typing import Dict, Any, Optional

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
    灵巧手 HTTP API 客户端

    接口（基于官方文档）：
      GET  /api/status   — 设备状态
      GET  /api/pose     — 归一化位置
      POST /api/set_pos  — 归一化位置控制 (0-1, 10 DOF)
      POST /api/set_pvc  — 弧度/速度/电流控制

    注意：决赛使用傅利叶 DexHand，实际 API 可能与 O10 不同。
    现场需根据实际部署的灵巧手确认接口。
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

    def is_ready(self) -> bool:
        """检查灵巧手是否就绪"""
        try:
            status = self.get_status()
            return status.get("connected", False)
        except Exception:
            return False

    # ---------- 基本操作 ----------

    def set_position(
        self,
        positions: Optional[list] = None,
        value: float = HAND_GRASP_OPEN,
    ) -> Dict[str, Any]:
        """
        设置归一化位置 (0-1)。

        简化接口：如果未指定具体手指位置，则所有手指使用 value。
        value=0 全开，value=1 全闭。

        对于 10 DOF 灵巧手，positions 应为 10 元素数组。
        """
        if positions is None:
            # 所有手指统一位置
            positions = [value] * 10

        data = {"positions": positions}
        logger.info(f"灵巧手位置控制 → {[f'{p:.2f}' for p in positions[:5]]}...")
        return self._post("/api/set_pos", data=data, timeout=10)

    def grasp(self, strength: float = HAND_GRASP_GENTLE):
        """抓取（闭合手指）"""
        logger.info(f"灵巧手抓取 (strength={strength})")
        return self.set_position(value=strength)

    def release(self):
        """释放（张开手指）"""
        logger.info("灵巧手张开")
        return self.set_position(value=HAND_GRASP_OPEN)

    def close(self):
        """完全闭合"""
        logger.info("灵巧手完全闭合")
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
