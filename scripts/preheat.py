#!/usr/bin/env python3
"""
模型预热脚本
============
开赛前运行，将所有模型加载进内存。
预热后推理延迟 < 3 秒。

用法:
    python preheat.py
"""
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preheat")

def main():
    logger.info("=" * 50)
    logger.info("汪汪队决赛算法服务 — 模型预热")
    logger.info("=" * 50)

    t0 = time.time()

    # 1. CLIP 分类器
    logger.info("[1/4] 加载 CLIP 分类器...")
    from vision.classifier import get_shape_classifier
    clf = get_shape_classifier()
    if clf.model is None:
        logger.error("CLIP 加载失败！")
        sys.exit(1)
    logger.info(f"      CLIP 就绪 ({time.time()-t0:.1f}s)")

    # 2. Grounding DINO
    logger.info("[2/4] 加载 Grounding DINO 检测器...")
    from vision.detector import get_detector
    det = get_detector()
    if det.model is None:
        logger.warning("Grounding DINO 加载失败（可降级运行）")
    else:
        logger.info(f"      Grounding DINO 就绪 ({time.time()-t0:.1f}s)")

    # 3. EasyOCR
    logger.info("[3/4] 加载 EasyOCR...")
    from vision.ocr_engine import get_ocr_engine
    ocr = get_ocr_engine()
    if not ocr.available:
        logger.warning("EasyOCR 加载失败（可降级运行）")
    else:
        logger.info(f"      EasyOCR 就绪 ({time.time()-t0:.1f}s)")

    # 4. 相机
    logger.info("[4/4] 初始化相机...")
    try:
        from vision.camera import CameraWrapper
        cam = CameraWrapper()
        cam.initialize()
        img = cam.capture()
        logger.info(f"      相机就绪, 图像尺寸 {img.size} ({time.time()-t0:.1f}s)")
    except Exception as e:
        logger.warning(f"      相机初始化失败: {e}")

    # 5. 硬件连接检查
    logger.info("[附加] 检查机械臂和灵巧手连接...")
    try:
        from hardware.arm_client import ArmClient
        arm = ArmClient()
        if arm.check_connection():
            logger.info("      机械臂连接正常")
        else:
            logger.warning("      机械臂连接失败，请检查 IP/端口")
    except Exception as e:
        logger.warning(f"      机械臂检查异常: {e}")

    try:
        from hardware.hand_client import HandClient
        hand = HandClient()
        if hand.check_connection():
            logger.info("      灵巧手连接正常")
        else:
            logger.warning("      灵巧手连接失败，请检查 IP/端口")
    except Exception as e:
        logger.warning(f"      灵巧手检查异常: {e}")

    total = time.time() - t0
    logger.info("=" * 50)
    logger.info(f"✅ 预热完成，总耗时 {total:.1f}s")
    logger.info("   所有模型已驻留内存，推理延迟 < 3s")
    logger.info("   可以开始比赛！")
    logger.info("=" * 50)

    # 保持进程运行（模型驻留内存），直到用户 Ctrl+C
    logger.info("提示: 保持此终端开启，模型会一直驻留内存。按 Ctrl+C 退出。")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("预热进程退出")


if __name__ == "__main__":
    main()
