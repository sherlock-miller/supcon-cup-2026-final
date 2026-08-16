"""
目标检测 — Grounding DINO 开放词汇检测（决赛优化版）
====================================================
决赛三项任务的专用检测词表（中英混合）+ 后处理管线：
  - NMS 去重（IoU 阈值）
  - 面积过滤（过小=噪声 / 过大=误检）
  - 置信度自适应（无结果时降阈值重试一次）
+ 传统 CV 兜底：HSV 颜色空间亮灯检测（无 ML 依赖）

注意：torch 延迟导入 — 无 ML 环境时模块仍可导入，
hsv_lit_light_detect 只依赖 numpy（cv2 可用时走 cv2 精细路径）。
"""
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from config import DETECT_LABELS, DETECT_MODEL_PATH

logger = logging.getLogger(__name__)

# ============================================================
# 默认检测阈值
# ============================================================
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
# 自适应重试的下限（低于此值不再降）
MIN_BOX_THRESHOLD = 0.12
# 面积过滤（占整图比例）：小于 min 视为噪声，大于 max 视为误检
MIN_BOX_AREA_RATIO = 0.0008
MAX_BOX_AREA_RATIO = 0.85
# NMS IoU 阈值
NMS_IOU_THRESHOLD = 0.5

# ============================================================
# 决赛场景检测词表（中英混合 — 覆盖三个任务）
# ============================================================
FINAL_DETECT_PROMPTS: Dict[str, List[str]] = {
    # 任务1：开关面板亮灯
    "light": [
        "亮着的灯", "点亮的指示灯", "发光的灯",
        "lit light", "glowing indicator light", "lighted lamp",
    ],
    # 任务1：开关/按钮（灯下方对应操作件）
    "switch": [
        "开关", "按钮", "拨杆",
        "button", "toggle switch", "switch",
    ],
    # 任务2：长方体转运块（俯拍顶面为方形，词表覆盖多种描述，
    # 不包含数字相关词——方块顶面的数字由 OCR 阶段独立识别）
    "cube": [
        "长方体", "方块", "木块", "立方体",
        "cuboid", "rectangular block", "box", "cube", "block", "square",
    ],
    # 任务3：几何体
    "shape": [
        "几何体", "长方体", "正方体", "圆柱体", "球体", "三棱柱",
        "geometric shape", "3d object", "block",
    ],
}


def get_final_prompts(scene: str) -> List[str]:
    """获取决赛场景专用词表；未知场景返回空列表"""
    return FINAL_DETECT_PROMPTS.get(scene, [])


def _box_iou(box_a: List[float], box_b: List[float]) -> float:
    """纯 numpy 计算两个 xyxy 框的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def _nms_targets(targets: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    """按 IoU 做 NMS 去重（targets 需已按 score 降序）"""
    keep: List[Dict[str, Any]] = []
    for t in targets:
        suppressed = any(
            _box_iou(t["bbox"], k["bbox"]) > iou_threshold for k in keep
        )
        if not suppressed:
            keep.append(t)
    return keep


class GroundingDinoDetector:
    """Grounding DINO 开放词汇目标检测器（决赛增强版）"""

    def __init__(self):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._init_model()

    def _init_model(self):
        try:
            from transformers import (
                GroundingDinoProcessor,
                GroundingDinoForObjectDetection,
            )
            # 使用 tiny 版本减小模型体积和推理时间
            model_name = "IDEA-Research/grounding-dino-tiny"
            logger.info(f"加载 Grounding DINO: {model_name}")
            self.processor = GroundingDinoProcessor.from_pretrained(model_name)
            self.model = GroundingDinoForObjectDetection.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info(f"Grounding DINO 就绪 (device={self.device})")
        except Exception as e:
            logger.error(f"Grounding DINO 加载失败: {e}")
            self.model = None

    def _infer_once(self, image, text_prompt: str, box_threshold: float):
        """单次前向推理，返回原始检测结果"""
        import torch
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )
        return results[0] if results else {}

    def _postprocess(
        self,
        raw: Dict[str, Any],
        class_names: List[str],
        image_w: int,
        image_h: int,
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """原始输出 → 过滤 + NMS 后的 targets 列表"""
        boxes = raw.get("boxes", [])
        labels = raw.get("labels", [])
        scores = raw.get("scores", [])

        img_area = image_w * image_h
        min_area = float(meta.get("min_area", MIN_BOX_AREA_RATIO * img_area))
        max_area = float(meta.get("max_area", MAX_BOX_AREA_RATIO * img_area))
        nms_iou = float(meta.get("nms_iou", NMS_IOU_THRESHOLD))

        targets = []
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            # 面积过滤：过小=噪声，过大=误检
            if area < min_area or area > max_area:
                continue

            # 多类别时标签可能合并（"defect valve"）→ 匹配回单个类别
            label_str = str(label)
            matched = None
            for cn in class_names:
                if cn in label_str:
                    matched = cn
                    break
            if not matched:
                matched = label_str.split()[0] if label_str else class_names[0]

            targets.append({
                "label": matched,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "score": round(float(score), 3),
                "bbox": [x1, y1, x2, y2],
                "area": area,
                "width": x2 - x1,
                "height": y2 - y1,
            })

        # 按置信度降序 → NMS 去重
        targets.sort(key=lambda t: t["score"], reverse=True)
        targets = _nms_targets(targets, nms_iou)
        return targets

    def predict(
        self,
        image,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        开放词汇目标检测（决赛增强版）

        meta 支持的键：
          class_names: List[str]        类别词表
          scene: str                    决赛场景名（light/switch/cube/shape）
          box_threshold: float          覆盖默认框阈值
          adaptive_threshold: bool      无结果时降阈值重试（默认 True）
          min_area / max_area: float    覆盖面积过滤阈值（像素）
          nms_iou: float                覆盖 NMS IoU 阈值
        """
        if self.model is None:
            return {"targets": []}

        meta = meta or {}

        # 类别词表：优先 meta 显式传入，其次决赛场景词表
        class_names = meta.get("class_names")
        if not class_names and meta.get("scene"):
            class_names = get_final_prompts(meta["scene"])
        if not class_names:
            return {"targets": []}

        # Grounding DINO 文本 prompt：用 "." 分隔类别
        text_prompt = " . ".join(class_names) + " ."

        box_thr = float(meta.get("box_threshold", BOX_THRESHOLD))

        try:
            raw = self._infer_once(image, text_prompt, box_thr)
            targets = self._postprocess(
                raw, class_names, image.width, image.height, meta
            )

            # 置信度自适应：无结果时降低阈值重试一次
            adaptive = meta.get("adaptive_threshold", True)
            if adaptive and not targets and box_thr > MIN_BOX_THRESHOLD:
                retry_thr = max(MIN_BOX_THRESHOLD, box_thr * 0.55)
                logger.info(
                    f"DINO 阈值 {box_thr} 无结果 → 降低到 {retry_thr:.2f} 重试"
                )
                raw = self._infer_once(image, text_prompt, retry_thr)
                targets = self._postprocess(
                    raw, class_names, image.width, image.height, meta
                )

            return {"targets": targets}

        except Exception as e:
            logger.error(f"Grounding DINO 推理异常: {e}")
            return {"targets": []}


# ============================================================
# 传统 CV 兜底 — HSV 颜色空间亮灯检测
# ============================================================

# 亮灯 HSV 范围（OpenCV: H 0-179, S/V 0-255）
# 亮灯特征：饱和度高 + 亮度高。V 阈值是与"灭灯"区分的核心：
# 亮灯 V 接近 255，灭灯明显更暗（测试图灭灯 V≤220，亮灯 V=255）。
# 红色 HSV 环绕 0°，需要两个区间。
_LIGHT_HSV_RANGES: Dict[str, List[Tuple[int, int, int, int]]] = {
    "red":    [(0, 12, 100, 220), (168, 179, 100, 220)],
    "yellow": [(18, 38, 100, 220)],
    "green":  [(40, 90, 90, 220)],
}

# 候选区域占整图面积比的范围（过滤噪声点/整块面板）
_LIGHT_MIN_AREA_RATIO = 0.0005
_LIGHT_MAX_AREA_RATIO = 0.30


def _rgb_to_hsv_numpy(arr_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """纯 numpy RGB(uint8 HxWx3) → HSV（cv2 不可用时的兜底）"""
    r = arr_uint8[..., 0].astype(np.float32) / 255.0
    g = arr_uint8[..., 1].astype(np.float32) / 255.0
    b = arr_uint8[..., 2].astype(np.float32) / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn

    h = np.zeros_like(mx)
    mask = d > 1e-6
    # 按最大通道分支计算色相（0-360 度）
    h_r = (60.0 * ((g - b) / np.maximum(d, 1e-6)) + 360.0) % 360.0
    h_g = 60.0 * ((b - r) / np.maximum(d, 1e-6)) + 120.0
    h_b = 60.0 * ((r - g) / np.maximum(d, 1e-6)) + 240.0
    h = np.where(mask, np.where(mx == r, h_r, np.where(mx == g, h_g, h_b)), 0.0)
    s = np.where(mx > 1e-6, d / mx * 255.0, 0.0)
    v = mx * 255.0
    # 映射到 OpenCV 范围
    return (h / 2.0), s, v  # H: 0-179


def _get_hsv(arr_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB → HSV。优先 cv2，缺省用纯 numpy"""
    try:
        import cv2
        hsv = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2HSV)
        return hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    except ImportError:
        return _rgb_to_hsv_numpy(arr_uint8)


def _mask_components_cv2(mask: np.ndarray) -> List[Dict[str, Any]]:
    """cv2 连通域分析：返回各组件 (cx, cy, area, mean_v)"""
    import cv2
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    comps = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 1:
            continue
        m = cv2.moments(cnt)
        if m["m00"] <= 0:
            continue
        comps.append({
            "cx": float(m["m10"] / m["m00"]),
            "cy": float(m["m01"] / m["m00"]),
            "area": area,
            "bbox": list(cv2.boundingRect(cnt)),
        })
    return comps


def _mask_centroid_numpy(mask: np.ndarray) -> List[Dict[str, Any]]:
    """numpy 兜底：整 mask 质心（单色亮灯通常只有一个区域）。
    bbox 与 cv2 路径保持一致，使用 (x, y, w, h) 格式。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    return [{
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "area": float(len(xs)),
        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
    }]


def hsv_lit_light_detect(
    image,
    min_area_ratio: float = _LIGHT_MIN_AREA_RATIO,
    max_area_ratio: float = _LIGHT_MAX_AREA_RATIO,
) -> List[Dict[str, Any]]:
    """
    传统 CV 兜底：HSV 颜色空间检测红/黄/绿亮灯（无 ML 依赖）

    亮灯判定：饱和度 S 高（彩色而非灰白）+ 亮度 V 高（亮灯 vs 灭灯的核心区分）。

    Args:
        image: PIL Image（RGB）
        min_area_ratio: 候选区域最小面积（占整图比例），过滤噪声
        max_area_ratio: 候选区域最大面积（占整图比例），过滤整块面板误检

    Returns:
        [{"color": "red"/"yellow"/"green", "cx": float, "cy": float,
          "score": float, "bbox": [x1,y1,x2,y2], "area": float, "mean_v": float}]
        按 score（亮度×面积）降序
    """
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    img_area = h * w
    min_area = min_area_ratio * img_area
    max_area = max_area_ratio * img_area

    hsv_h, hsv_s, hsv_v = _get_hsv(arr)

    candidates: List[Dict[str, Any]] = []
    for color, ranges in _LIGHT_HSV_RANGES.items():
        mask = np.zeros((h, w), dtype=bool)
        for (h_lo, h_hi, s_lo, v_lo) in ranges:
            mask |= (
                (hsv_h >= h_lo) & (hsv_h <= h_hi)
                & (hsv_s >= s_lo) & (hsv_v >= v_lo)
            )

        if mask.sum() < min_area:
            continue

        # 连通域分析（cv2 精细 / numpy 兜底）
        try:
            comps = _mask_components_cv2(mask)
        except ImportError:
            comps = _mask_centroid_numpy(mask)

        for comp in comps:
            area = comp["area"]
            if area < min_area or area > max_area:
                continue
            # bbox 转 xyxy（cv2 boundingRect 返回 x,y,w,h）
            bx, by, bw, bh = comp["bbox"]
            # 该组件区域的亮度均值（亮灯应接近饱和）
            xi, yi = int(bx), int(by)
            wi, hi = max(1, int(round(bw))), max(1, int(round(bh)))
            sub_mask = mask[yi:yi + hi, xi:xi + wi]
            if sub_mask.sum() > 0:
                mean_v = float(hsv_v[yi:yi + hi, xi:xi + wi][sub_mask].mean())
            else:
                mean_v = float(hsv_v[mask].mean())
            score = round(min(1.0, (mean_v / 255.0) * 0.6 + (area / img_area) * 40.0), 3)
            candidates.append({
                "color": color,
                "cx": comp["cx"],
                "cy": comp["cy"],
                "score": score,
                "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                "area": float(area),
                "mean_v": mean_v,
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


_detector: Optional[GroundingDinoDetector] = None


def get_detector() -> GroundingDinoDetector:
    global _detector
    if _detector is None:
        _detector = GroundingDinoDetector()
    return _detector
