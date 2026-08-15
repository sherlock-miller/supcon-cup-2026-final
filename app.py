"""
中控杯决赛 — 汪汪队算法服务主入口
================================
FastAPI 服务，对接竞赛操作软件

端点:
  GET  /api/health         → 健康检查
  POST /api/task1/execute  → 拨按开关
  POST /api/task2/execute  → 长方体有序转运
  POST /api/task3/execute  → 几何体无序分拣
"""
import time
import logging
import sys
import traceback
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    SERVICE_NAME, SERVICE_VERSION, HOST, PORT,
    TASK_TIMEOUT_MS,
)

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("final-algo")

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title=f"{SERVICE_NAME} - 决赛",
    version=SERVICE_VERSION,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 懒加载模块
# ============================================================
_arm = None
_hand = None
_vision = None


def _get_arm():
    """获取机械臂客户端（懒加载）"""
    global _arm
    if _arm is None:
        from hardware.arm_client import ArmClient
        _arm = ArmClient()
    return _arm


def _get_hand():
    """获取灵巧手客户端（懒加载）"""
    global _hand
    if _hand is None:
        from hardware.hand_client import HandClient
        _hand = HandClient()
    return _hand


def _get_vision():
    """获取视觉模块（懒加载）"""
    global _vision
    if _vision is None:
        from vision.vision_manager import VisionManager
        _vision = VisionManager()
    return _vision


# ============================================================
# GET /api/health — 健康检查
# ============================================================
@app.get("/api/health")
def health() -> Dict[str, Any]:
    """竞赛操作软件通过此端点确认算法服务在线"""
    return {
        "success": True,
        "message": "ready",
    }


# ============================================================
# POST /api/task1/execute — 拨按开关
# ============================================================
@app.post("/api/task1/execute")
def task1_execute() -> Dict[str, Any]:
    """
    任务1：视觉定位亮灯 → 控制机械臂+灵巧手按/拨对应开关。
    竞赛软件每次随机亮一个灯，连续调用三次。
    """
    started = time.perf_counter()
    logger.info("===== 任务1: 拨按开关 开始 =====")

    try:
        from tasks.task1_switch import execute_switch_task

        ok, message = execute_switch_task(
            arm=_get_arm(),
            hand=_get_hand(),
            vision=_get_vision(),
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(f"任务1 完成: ok={ok}, message={message}, elapsed={elapsed_ms}ms")

        return {
            "success": ok,
            "message": message,
        }
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.error(f"任务1 异常: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "message": f"任务1异常: {type(e).__name__}: {str(e)[:200]}",
        }


# ============================================================
# POST /api/task2/execute — 长方体有序转运
# ============================================================
@app.post("/api/task2/execute")
def task2_execute() -> Dict[str, Any]:
    """
    任务2：识别数字 1-4 的长方体 → 按序抓取 → 放到指定台面。
    """
    started = time.perf_counter()
    logger.info("===== 任务2: 长方体有序转运 开始 =====")

    try:
        from tasks.task2_cubes import execute_cube_task

        ok, message = execute_cube_task(
            arm=_get_arm(),
            hand=_get_hand(),
            vision=_get_vision(),
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(f"任务2 完成: ok={ok}, message={message}, elapsed={elapsed_ms}ms")

        return {
            "success": ok,
            "message": message,
        }
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.error(f"任务2 异常: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "message": f"任务2异常: {type(e).__name__}: {str(e)[:200]}",
        }


# ============================================================
# POST /api/task3/execute — 几何体无序分拣
# ============================================================
@app.post("/api/task3/execute")
def task3_execute() -> Dict[str, Any]:
    """
    任务3：识别几何体形状 → 抓取 → 放入对应形状槽位。
    """
    started = time.perf_counter()
    logger.info("===== 任务3: 几何体无序分拣 开始 =====")

    try:
        from tasks.task3_shapes import execute_shape_task

        ok, message = execute_shape_task(
            arm=_get_arm(),
            hand=_get_hand(),
            vision=_get_vision(),
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(f"任务3 完成: ok={ok}, message={message}, elapsed={elapsed_ms}ms")

        return {
            "success": ok,
            "message": message,
        }
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.error(f"任务3 异常: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "message": f"任务3异常: {type(e).__name__}: {str(e)[:200]}",
        }


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动 {SERVICE_NAME} v{SERVICE_VERSION} 于 {HOST}:{PORT}")
    logger.info("接口列表:")
    logger.info("  GET  /api/health")
    logger.info("  POST /api/task1/execute")
    logger.info("  POST /api/task2/execute")
    logger.info("  POST /api/task3/execute")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
