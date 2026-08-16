"""
目标检测 — Grounding DINO 开放词汇检测
输入图片 + 类别名文本（如 "defect . valve"）→ 输出目标框

注意：torch/numpy 延迟导入（同 classifier.py）
"""
import logging
from typing import Dict, Any, Optional

from config import DETECT_LABELS, DETECT_MODEL_PATH

logger = logging.getLogger(__name__)

# 默认检测阈值
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20


class GroundingDinoDetector:
    """Grounding DINO 开放词汇目标检测器"""

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

    def predict(
        self,
        image,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """开放词汇目标检测"""
        import torch
        if self.model is None:
            return {"targets": []}

        # 从 meta 中获取类别名
        class_names = None
        if meta and isinstance(meta.get("class_names"), list):
            class_names = meta["class_names"]

        if not class_names:
            return {"targets": []}

        # Grounding DINO 文本 prompt：用 "." 分隔类别
        # 例如: class_names=['defect','valve'] → "defect . valve ."
        text_prompt = " . ".join(class_names) + " ."

        try:
            inputs = self.processor(
                images=image,
                text=text_prompt,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # 后处理：提取检测结果
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[image.size[::-1]],
            )

            targets = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.get("boxes", [])
                labels = result.get("labels", [])
                scores = result.get("scores", [])

                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box.tolist()
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    # 多类别时标签可能合并("defect valve")→匹配回单个类别
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
                        "cx": cx,
                        "cy": cy,
                        "score": round(float(score), 3),
                    })

            # 按置信度降序
            targets.sort(key=lambda t: t["score"], reverse=True)
            return {"targets": targets}

        except Exception as e:
            logger.error(f"Grounding DINO 推理异常: {e}")
            return {"targets": []}


_detector: Optional[GroundingDinoDetector] = None


def get_detector() -> GroundingDinoDetector:
    global _detector
    if _detector is None:
        _detector = GroundingDinoDetector()
    return _detector
