"""
视觉管理模块（决赛优化版）
==========================
统一管理 CLIP 分类器、EasyOCR、Grounding DINO 检测器、相机。

决赛三项任务检测策略（模型 + 传统 CV 双层降级）：
  任务1 detect_lit_light：
    HSV 颜色空间亮灯检测（红/黄/绿，V 阈值区分亮灭）→ 颜色→灯号映射
    → 布局先验验证（垂直/水平三分兜底）→ DINO 检测兜底
  任务2 detect_cube_numbers：
    方块区域检测（DINO → 传统 CV 补充合并）→ 区域裁剪
    → 单数字识别（EasyOCR ↔ 模板匹配交叉验证 + 数字1几何先验）
  任务3 detect_and_classify_shapes：
    几何体检测（DINO → 传统 CV 轮廓分割）→ CLIP 多模板分类
    + 传统 CV 几何特征分类（圆形度/顶点数/长宽比，俯拍视角）
    按置信度融合（高置信 CV > 高置信 CLIP > 低置信 CV > CLIP 兜底）

所有检测函数都有降级路径：模型不可用时不崩溃，走传统 CV 或返回空。

注意：Gemini335 深度相机需要奥比中光 SDK。
目前使用 OpenCV 作为占位接口，现场需替换为实际 SDK。
"""
import logging
import math
import socket
from typing import Dict, Any, Optional, Tuple, List

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 决赛检测阈值
# ============================================================
# CLIP 形状分类置信度阈值（低于此值回退到几何特征分类）
SHAPE_CONF_THRESHOLD = 0.40
# 几何特征分类视为"高置信"的阈值
GEOMETRY_HIGH_CONF = 0.70
# 传统 CV 物体检测：候选面积占整图比例范围
CV_MIN_AREA_RATIO = 0.004   # 过滤噪声小块
CV_MAX_AREA_RATIO = 0.30    # 过滤整图误检
# 方块矩形检测：长宽比范围（俯拍顶面近似方形）
BLOCK_ASPECT_RANGE = (0.55, 1.8)


class VisionManager:
    """视觉模块统一管理"""

    def __init__(self):
        self._classifier = None
        self._detector = None
        self._ocr = None
        self._camera = None
        self._initialized = False

    def initialize(self):
        """初始化所有视觉模型（首次调用时加载）"""
        if self._initialized:
            return

        logger.info("初始化视觉模型...")

        # 无网络时模型下载快速失败，避免挂起整个服务
        socket.setdefaulttimeout(20)

        # CLIP 分类器（通用零样本：形状 + 场景）
        try:
            from vision.classifier import get_shape_classifier
            self._classifier = get_shape_classifier()
            logger.info("CLIP 形状分类器就绪")
        except Exception as e:
            logger.error(f"CLIP 加载失败: {e}")

        # Grounding DINO 检测器（复用初赛代码）
        try:
            from vision.detector import GroundingDinoDetector
            self._detector = GroundingDinoDetector()
            logger.info("Grounding DINO 检测器就绪")
        except Exception as e:
            logger.error(f"Grounding DINO 加载失败: {e}")

        # EasyOCR
        try:
            from vision.ocr_engine import OCREngine
            self._ocr = OCREngine()
            logger.info("EasyOCR 就绪")
        except Exception as e:
            logger.error(f"EasyOCR 加载失败: {e}")

        # 相机（占位：现场需替换为 Gemini335 SDK）
        try:
            from vision.camera import CameraWrapper
            self._camera = CameraWrapper()
            logger.info("相机接口就绪")
        except Exception as e:
            logger.error(f"相机初始化失败: {e}")

        self._initialized = True
        logger.info("视觉模型初始化完成")

    # ================================================================
    # 拍照
    # ================================================================

    def capture_image(self) -> Image.Image:
        """拍照并返回 PIL Image"""
        if self._camera is None:
            raise RuntimeError("相机未初始化")
        return self._camera.capture()

    def capture_with_depth(self) -> Tuple[Image.Image, np.ndarray]:
        """拍照并返回 RGB + Depth"""
        if self._camera is None:
            raise RuntimeError("相机未初始化")
        return self._camera.capture_with_depth()

    # ================================================================
    # 传统 CV 通用工具
    # ================================================================

    @staticmethod
    def _region_dominant_color(
        arr: np.ndarray,
        bbox: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        区域内红/黄/绿主导色分析（HSV）。

        返回 {"color": "red"/"yellow"/"green", "mean_v": float, "ratio": float}
        或 None（无显著颜色）。用于 DINO 候选框的颜色验证。
        """
        from vision.detector import _get_hsv
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(arr.shape[1], x2); y2 = min(arr.shape[0], y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                return None
            region = arr[y1:y2, x1:x2]
        else:
            region = arr

        hsv_h, hsv_s, hsv_v = _get_hsv(region)
        total = hsv_h.size
        if total == 0:
            return None

        ranges = {
            "red":    [((0, 12), (168, 179))],
            "yellow": [((18, 38),)],
            "green":  [((40, 90),)],
        }
        best_color, best_ratio = None, 0.0
        for color, hranges in ranges.items():
            mask = np.zeros_like(hsv_h, dtype=bool)
            for hr in hranges:
                if len(hr) == 1:
                    h_lo, h_hi = hr[0]
                    mask |= (hsv_h >= h_lo) & (hsv_h <= h_hi)
                else:
                    (h1_lo, h1_hi), (h2_lo, h2_hi) = hr
                    mask |= ((hsv_h >= h1_lo) & (hsv_h <= h1_hi)) | \
                            ((hsv_h >= h2_lo) & (hsv_h <= h2_hi))
            # 要求饱和度高（彩色像素）
            mask &= hsv_s >= 80
            ratio = float(mask.sum()) / total
            if ratio > best_ratio:
                best_color, best_ratio = color, ratio

        if best_color is None or best_ratio < 0.05:
            return None
        return {
            "color": best_color,
            "ratio": round(best_ratio, 3),
            "mean_v": float(hsv_v.mean()),
        }

    @staticmethod
    def _detect_contours_cv(
        image: Image.Image,
        min_area_ratio: float = CV_MIN_AREA_RATIO,
        max_area_ratio: float = CV_MAX_AREA_RATIO,
    ) -> List[Dict[str, Any]]:
        """
        传统 CV 通用物体检测：Canny 边缘 → 外部轮廓 → 面积过滤 → 重叠去重。

        返回 [{"cx","cy","bbox":[x1,y1,x2,y2],"area","contour"}, ...] 按面积降序。
        cv2 缺失时返回空列表（调用方继续走其他路径）。
        """
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 不可用，跳过传统 CV 检测")
            return []

        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        img_area = gray.size
        min_area = min_area_ratio * img_area
        max_area = max_area_ratio * img_area

        # 高斯模糊 → Canny → 膨胀连接断边
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            candidates.append({
                "cx": x + w / 2.0,
                "cy": y + h / 2.0,
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "area": area,
                "contour": cnt,
            })

        # 面积降序 → 重叠去重（嵌套框只保留外框）
        candidates.sort(key=lambda c: c["area"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for cand in candidates:
            x1, y1, x2, y2 = cand["bbox"]
            overlap = False
            for k in kept:
                kx1, ky1, kx2, ky2 = k["bbox"]
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter > 0.5 * min(cand["area"], k["area"]):
                    overlap = True
                    break
            if not overlap:
                kept.append(cand)
        return kept

    @staticmethod
    def _detect_blocks_cv(image: Image.Image) -> List[Dict[str, Any]]:
        """
        传统 CV 方块检测（任务2 兜底）：找矩形顶面。

        特征：近似 4 顶点多边形 + 矩形度高 + 长宽比接近方形（俯拍顶面）。
        """
        try:
            import cv2
        except ImportError:
            return []

        # 复用通用轮廓检测（排除过小/过大区域）
        base = VisionManager._detect_contours_cv(image)
        blocks = []
        for cand in base:
            cnt = cand["contour"]
            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) != 4:
                continue  # 非四边形
            if not cv2.isContourConvex(approx):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / float(h)
            if not (BLOCK_ASPECT_RANGE[0] <= aspect <= BLOCK_ASPECT_RANGE[1]):
                continue
            # 矩形度：轮廓面积 / 外接矩形面积
            rect_ratio = cand["area"] / float(w * h) if w * h > 0 else 0.0
            if rect_ratio < 0.70:
                continue
            blocks.append({
                "cx": cand["cx"],
                "cy": cand["cy"],
                "bbox": cand["bbox"],
                "area": cand["area"],
                "aspect": round(aspect, 3),
            })
        return blocks

    # ================================================================
    # 任务1 专用：灯亮检测
    # ================================================================

    def detect_lit_light(
        self,
        image: Image.Image,
    ) -> Optional[Dict[str, Any]]:
        """
        检测开关面板上哪个灯亮了（HSV 颜色检测 + 布局先验验证）。

        策略：
        1. HSV 检测红/黄/绿亮灯候选（V 阈值区分亮灭，可靠性最高）
        2. 颜色 → 灯号映射（config.SWITCH_PANEL["lights"] 每盏灯的颜色）
        3. 布局先验兜底：颜色无法确定时按垂直/水平三分定位灯号
        4. DINO 兜底：HSV 无结果时检测"亮灯"框 + 框内颜色验证

        返回: {"light_id", "switch_type", "pixel": (x, y),
               "color", "confidence", "method"}
              或 None（未检测到亮灯）
        """
        from config import SWITCH_PANEL

        # ---- 候选收集：HSV 主路径 + DINO 兜底 ----
        lit_candidates: List[Dict[str, Any]] = []

        try:
            from vision.detector import hsv_lit_light_detect
            for cand in hsv_lit_light_detect(image):
                lit_candidates.append({**cand, "method": "hsv"})
        except Exception as e:
            logger.warning(f"HSV 亮灯检测异常: {e}")

        if not lit_candidates and self._detector:
            # DINO 兜底：检测"亮灯"框 → 框内颜色/亮度验证
            try:
                dets = self._detector.predict(image, {"scene": "light"})
                arr = np.asarray(image.convert("RGB"))
                for t in dets.get("targets", []):
                    info = self._region_dominant_color(arr, t.get("bbox"))
                    if info is None or info["mean_v"] < 200:
                        continue  # 框内无显著颜色或亮度不足 → 不是亮灯
                    lit_candidates.append({
                        "color": info["color"],
                        "cx": t["cx"],
                        "cy": t["cy"],
                        "score": round(float(t["score"]) * 0.8, 3),
                        "bbox": t.get("bbox"),
                        "area": t.get("area", 0.0),
                        "mean_v": info["mean_v"],
                        "method": "dino",
                    })
            except Exception as e:
                logger.warning(f"DINO 亮灯兜底异常: {e}")

        if not lit_candidates:
            return None

        # 按置信度排序（高亮的优先）
        lit_candidates.sort(key=lambda c: c["score"], reverse=True)
        best = lit_candidates[0]

        # ---- 颜色 → 灯号映射（config 预设每盏灯的颜色） ----
        light_config = SWITCH_PANEL["lights"]  # {light_id: {pixel_x, pixel_y, color}}
        light_id: Optional[str] = None
        for lid, cfg in light_config.items():
            if cfg.get("color") == best["color"]:
                light_id = lid
                break

        if light_id is None:
            # 颜色未匹配 → 布局先验：垂直三分优先，水平三分兜底
            light_id = self._light_id_by_layout(
                best["cx"], best["cy"], image.width, image.height
            )

        switch_type = SWITCH_PANEL["switch_type"].get(light_id, "button")
        return {
            "light_id": light_id,
            "switch_type": switch_type,
            "pixel": (float(best["cx"]), float(best["cy"])),
            "color": best["color"],
            "confidence": round(float(best.get("score", 0.0)), 3),
            "method": best.get("method", "hsv"),
        }

    @staticmethod
    def _light_id_by_layout(cx: float, cy: float, img_w: int, img_h: int) -> str:
        """
        布局先验兜底：灯垂直排列（从上到下 light_1/2/3），
        若无法按 Y 判定则按水平排列（从左到右）。
        返回 light_1/light_2/light_3。
        """
        # 垂直三分（任务情报：面板上方红黄绿3灯垂直排列）
        y_ratio = cy / max(img_h, 1)
        if y_ratio < 0.30:
            return "light_1"
        if y_ratio < 0.60:
            return "light_2"
        # 水平三分（测试图/其他布局兜底）
        x_ratio = cx / max(img_w, 1)
        if x_ratio < 0.30:
            return "light_1"
        if x_ratio < 0.60:
            return "light_2"
        return "light_3"

    # ================================================================
    # 任务2 专用：数字识别
    # ================================================================

    def detect_cube_numbers(
        self, image: Image.Image
    ) -> List[Dict[str, Any]]:
        """
        检测四个长方体上的数字（1-4）。

        策略：
        1. 方块区域检测：DINO 优先 → 传统 CV 矩形检测补充合并
           （DINO 检出数不足时 CV 补齐，IoU 去重；DINO 完全无结果时 CV 全量兜底）
        2. 每个方块顶面裁剪（向内收缩避开描边）
        3. 单数字识别：EasyOCR ↔ 模板匹配交叉验证（只认 1-4）
        4. 数字去重（每个数字唯一，保留置信度最高的方块）

        返回: [{"number": 1, "cx": 320, "cy": 240, "confidence": 0.9, ...}]
              按数字排序
        """
        from config import CUBE_SLOTS
        expected_blocks = len(CUBE_SLOTS.get("slot_positions", {})) or 4

        # ---- 步骤1：方块区域检测 ----
        blocks: List[Dict[str, Any]] = []

        if self._detector:
            try:
                dets = self._detector.predict(image, {"scene": "cube"})
                for t in dets.get("targets", []):
                    blocks.append({
                        "cx": t["cx"],
                        "cy": t["cy"],
                        "bbox": t.get("bbox"),
                        "score": t.get("score", 0.0),
                        "method": "dino",
                    })
            except Exception as e:
                logger.warning(f"DINO 方块检测异常: {e}")

        if len(blocks) < expected_blocks:
            # 传统 CV 矩形检测补充：DINO 部分漏检时补齐（IoU 去重）
            try:
                from vision.detector import _box_iou
                for b in self._detect_blocks_cv(image):
                    if not b.get("bbox"):
                        continue
                    if any(
                        blk.get("bbox")
                        and _box_iou(b["bbox"], blk["bbox"]) > 0.5
                        for blk in blocks
                    ):
                        continue
                    blocks.append({**b, "score": 0.7, "method": "cv"})
            except Exception as e:
                logger.warning(f"传统 CV 方块检测异常: {e}")

        if not blocks:
            return []

        # ---- 步骤2+3：裁剪 + 单数字识别 ----
        arr = np.asarray(image.convert("RGB"))
        img_h, img_w = arr.shape[:2]

        results: List[Dict[str, Any]] = []
        for block in blocks:
            bbox = block.get("bbox")
            if bbox:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            else:
                # 无 bbox 时以中心裁剪固定区域
                cx, cy = block["cx"], block["cy"]
                half = 70
                x1, y1 = cx - half, cy - half
                x2, y2 = cx + half, cy + half
            # 向内收缩 3px：避开顶面描边（描边会干扰数字识别）
            inset = 4
            x1, y1 = max(0, x1 + inset), max(0, y1 + inset)
            x2, y2 = min(img_w, x2 - inset), min(img_h, y2 - inset)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue

            crop = Image.fromarray(arr[int(y1):int(y2), int(x1):int(x2)])

            # 单数字识别：OCR 引擎优先（内部含模板匹配兜底）
            digit_info = None
            if self._ocr is not None:
                try:
                    digit_info = self._ocr.predict_single_digit(
                        crop, valid_digits=(1, 2, 3, 4)
                    )
                except Exception as e:
                    logger.warning(f"单数字识别异常: {e}")
            if digit_info is None:
                # 无 OCR 引擎 → 直接模板匹配兜底
                try:
                    from vision.ocr_engine import template_match_single_digit
                    digit_info = template_match_single_digit(
                        crop, valid_digits=(1, 2, 3, 4)
                    )
                except Exception as e:
                    logger.warning(f"模板匹配异常: {e}")

            if digit_info is None:
                continue

            results.append({
                "number": int(digit_info["digit"]),
                "cx": float(block["cx"]),
                "cy": float(block["cy"]),
                "confidence": round(float(digit_info.get("confidence", 0.0)), 3),
                "method": digit_info.get("method", "unknown"),
                "bbox": bbox,
            })

        # ---- 步骤4：数字去重（每个数字唯一，保留置信度最高者） ----
        deduped: Dict[int, Dict[str, Any]] = {}
        for r in results:
            num = r["number"]
            if num not in deduped or r["confidence"] > deduped[num]["confidence"]:
                deduped[num] = r

        results = sorted(deduped.values(), key=lambda x: x["number"])
        return results

    # ================================================================
    # 任务3 专用：形状分类
    # ================================================================

    def classify_shape(self, image: Image.Image) -> str:
        """
        识别几何体形状（单图分类）。

        策略：CLIP 多模板零样本分类 → 置信度不足时传统 CV 几何特征兜底
        """
        from config import SHAPE_LABELS

        label = SHAPE_LABELS[0]

        if self._classifier is not None:
            try:
                result = self._classifier.predict(image)
                label = result.get("label", SHAPE_LABELS[0])
                confidence = float(result.get("confidence", 0.0))
                if confidence < SHAPE_CONF_THRESHOLD:
                    # CLIP 置信度不足 → 几何特征兜底
                    cv_label, _ = self._classify_shape_by_geometry(image)
                    if cv_label is not None:
                        label = cv_label
            except Exception as e:
                logger.warning(f"CLIP 形状分类异常: {e}")
                cv_label, _ = self._classify_shape_by_geometry(image)
                if cv_label is not None:
                    label = cv_label
        else:
            cv_label, _ = self._classify_shape_by_geometry(image)
            if cv_label is not None:
                label = cv_label

        logger.info(f"形状分类结果: {label}")
        return label

    @staticmethod
    def _classify_shape_by_geometry(image: Image.Image) -> Tuple[Optional[str], float]:
        """
        传统 CV 几何特征形状分类（俯拍视角，几何体竖直摆放）。

        俯视图特征 → 形状映射：
          圆形度高        → 圆柱体/球体（俯视均为圆，默认圆柱体）
          3 顶点（三角形） → 三棱柱
          4 顶点 + 长宽比>1.15 → 长方体
          4 顶点 + 长宽比≈1   → 正方体
          6 顶点          → 六棱柱
          5 顶点          → 四棱锥
          其他多边形      → 多面体

        Returns:
            (shape_label, confidence) 或 (None, 0.0)
        """
        try:
            import cv2
        except ImportError:
            return None, 0.0

        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Canny 找物体外轮廓
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, 0.0

        # 取面积最大的轮廓（= 物体外轮廓）
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        img_area = float(gray.size)
        if area < 0.01 * img_area:
            return None, 0.0

        peri = cv2.arcLength(cnt, True)
        if peri < 1:
            return None, 0.0

        # 圆形度 c = 4πA/P²（圆=1，正方形≈0.785，正三角形≈0.605）
        circularity = 4.0 * math.pi * area / (peri * peri)

        # 顶点数（多边形近似）
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        k = len(approx)

        if circularity > 0.82:
            # 俯视图为圆：圆柱体/球体/圆锥同构 → 默认圆柱体（决赛常见）
            return "圆柱体", 0.72
        if k == 3:
            return "三棱柱", 0.82
        if k == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = (w / float(h)) if h > 0 else 1.0
            if aspect > 1.15:
                return "长方体", 0.80
            return "正方体", 0.75
        if k == 6:
            return "六棱柱", 0.70
        if k == 5:
            return "四棱锥", 0.60
        # 复杂多边形（含曲线混合）→ 多面体
        return "多面体", 0.50

    def detect_and_classify_shapes(
        self, image: Image.Image
    ) -> List[Dict[str, Any]]:
        """
        检测所有几何体并分类（两阶段 + 双重降级）。

        策略：
        1. 检测：DINO 检测几何体 → 传统 CV 轮廓分割兜底
        2. 分类：每个裁剪区域 CLIP 多模板分类 + 传统 CV 几何特征分类，
           按置信度融合（高置信 CV > 高置信 CLIP > 低置信 CV > CLIP 兜底）

        返回: [{"shape": "长方体", "cx": 320, "cy": 240, "confidence": 0.8,
                "method": "clip"/"geometry"}]
        """
        # ---- 步骤1：几何体检测 ----
        objects: List[Dict[str, Any]] = []

        if self._detector:
            try:
                dets = self._detector.predict(image, {"scene": "shape"})
                for t in dets.get("targets", []):
                    objects.append({
                        "cx": t["cx"],
                        "cy": t["cy"],
                        "bbox": t.get("bbox"),
                        "score": t.get("score", 0.0),
                        "method": "dino",
                    })
            except Exception as e:
                logger.warning(f"DINO 几何体检测异常: {e}")

        if not objects:
            # 传统 CV 轮廓分割兜底
            try:
                for c in self._detect_contours_cv(image):
                    objects.append({
                        "cx": c["cx"],
                        "cy": c["cy"],
                        "bbox": c["bbox"],
                        "score": 0.6,
                        "method": "cv",
                    })
            except Exception as e:
                logger.warning(f"传统 CV 几何体检测异常: {e}")

        if not objects:
            return []

        # ---- 步骤2：逐物体裁剪 + 分类 ----
        arr = np.asarray(image.convert("RGB"))
        img_h, img_w = arr.shape[:2]

        results = []
        for obj in objects:
            bbox = obj.get("bbox")
            if bbox:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            else:
                cx, cy = obj["cx"], obj["cy"]
                half = 60
                x1, y1 = cx - half, cy - half
                x2, y2 = cx + half, cy + half
            # 外扩 8px：几何体轮廓可能紧贴检测框
            pad = 8
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue

            crop = Image.fromarray(arr[int(y1):int(y2), int(x1):int(x2)])

            # ---- CLIP 分类 ----
            clip_label = None
            clip_conf = 0.0
            if self._classifier is not None:
                try:
                    clip_result = self._classifier.predict(crop)
                    clip_label = clip_result.get("label")
                    clip_conf = float(clip_result.get("confidence", 0.0))
                except Exception as e:
                    logger.warning(f"CLIP 分类异常: {e}")

            # ---- 几何特征分类 ----
            cv_label, cv_conf = self._classify_shape_by_geometry(crop)

            # ---- 置信度融合 ----
            if cv_label is not None and cv_conf >= GEOMETRY_HIGH_CONF:
                shape, conf, method = cv_label, cv_conf, "geometry"
            elif clip_label is not None and clip_conf >= SHAPE_CONF_THRESHOLD:
                shape, conf, method = clip_label, clip_conf, "clip"
            elif cv_label is not None:
                shape, conf, method = cv_label, cv_conf, "geometry"
            elif clip_label is not None:
                shape, conf, method = clip_label, clip_conf, "clip"
            else:
                shape, conf, method = "多面体", 0.0, "fallback"

            results.append({
                "shape": shape,
                "cx": float(obj["cx"]),
                "cy": float(obj["cy"]),
                "confidence": round(conf, 3),
                "method": method,
                "bbox": bbox,
            })

        return results

    # ================================================================
    # 坐标变换（像素 → 机械臂基坐标系）
    # ================================================================

    def pixel_to_arm_coord(
        self,
        pixel_x: float,
        pixel_y: float,
        depth_value: float,
        arm_pose: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float, float]:
        """
        像素坐标 + 深度 → 机械臂基坐标系 3D 坐标（完整 eye-in-hand 链路）

        变换链：
          像素 (u,v) + 深度 d
            → 相机坐标系 3D 点（相机内参 + 去畸变）
            → 机械臂末端坐标系（手眼矩阵 X = T_cam2gripper）
            → 基座坐标系（当前末端位姿 = 机械臂正运动学）

        Args:
            pixel_x, pixel_y: 像素坐标
            depth_value: 深度值 (mm)
            arm_pose: 机械臂当前末端位姿（可选；不传则读取 /api/pose）

        Returns:
            (x, y, z) 基座系坐标 (m)
        """
        # ===== 标定参数（calibrate.py 完成后由 apply_calibration.py 自动替换）=====
        fx, fy = 600.0, 600.0    # 相机内参（占位，待标定）
        cx, cy = 320.0, 240.0    # 光心（占位）
        dist_coeffs = []          # 畸变系数（占位）

        # 手眼矩阵：相机 → 末端（占位，待标定）
        R_cam2gripper = np.eye(3)
        t_cam2gripper = np.zeros(3)

        # ===== 1. 去畸变 + 相机系 3D 坐标 =====
        z_cam = depth_value / 1000.0  # mm → m
        if dist_coeffs:
            import cv2
            mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            dist = np.array(dist_coeffs)
            undistorted = cv2.undistortPoints(
                np.array([[[pixel_x, pixel_y]]], dtype=np.float32),
                mtx, dist, P=mtx,
            )[0][0]
            x_cam = float(undistorted[0]) * z_cam
            y_cam = float(undistorted[1]) * z_cam
        else:
            x_cam = (pixel_x - cx) * z_cam / fx
            y_cam = (pixel_y - cy) * z_cam / fy

        point_cam = np.array([x_cam, y_cam, z_cam])

        # ===== 2. 相机系 → 末端系（手眼矩阵） =====
        point_gripper = R_cam2gripper @ point_cam + t_cam2gripper

        # ===== 3. 末端系 → 基座系（末端当前位姿） =====
        if arm_pose is None:
            # 尝试读取机械臂位姿
            try:
                from hardware.arm_client import ArmClient
                arm = ArmClient()
                arm_pose = arm.get_pose().get("pose")
            except Exception:
                arm_pose = None

        if arm_pose:
            # 末端旋转矩阵（roll/pitch/yaw → R）
            roll, pitch, yaw = (
                arm_pose["roll"], arm_pose["pitch"], arm_pose["yaw"]
            )
            # XYZ 固定轴欧拉角 → 旋转矩阵
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ])
            Ry = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ])
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ])
            R_g2b = Rz @ Ry @ Rx
            t_g2b = np.array([
                arm_pose["x"], arm_pose["y"], arm_pose["z"],
            ])
            point_base = R_g2b @ point_gripper + t_g2b
        else:
            # 无位姿信息：假设末端在默认位（近似）
            point_base = point_gripper

        x_arm, y_arm, z_arm = point_base

        logger.debug(
            f"像素 ({pixel_x},{pixel_y}) + {depth_value}mm → 基坐标 "
            f"({x_arm:.3f}, {y_arm:.3f}, {z_arm:.3f})"
        )
        return (float(x_arm), float(y_arm), float(z_arm))
