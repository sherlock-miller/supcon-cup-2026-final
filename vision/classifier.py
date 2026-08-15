"""
通用 CLIP 零样本分类器
======================
支持任意类别集合（场景/几何体形状），自动生成中英文 prompt 模板。

任务3 形状分类 + 备用的场景分类都走这里。
"""
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
import logging
import os

from config import CLASSIFY_LABELS, SHAPE_LABELS

logger = logging.getLogger(__name__)

# 自动生成 prompt 模板：
# 每个类别生成 中/英 各一条模板 + 通用模板
def _make_templates(label: str) -> List[str]:
    """为任意类别生成 prompt 模板"""
    return [
        f"a photo of a {label}",
        f"a clear image of {label}",
        f"{label}的图片",
    ]


class CLIPClassifier:
    """CLIP 零样本分类器（通用）"""

    def __init__(self, labels: Optional[List[str]] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = labels if labels is not None else SHAPE_LABELS
        self.model = None
        self.processor = None
        self.text_embeddings = None
        self._init_model()

    def _init_model(self):
        try:
            from transformers import CLIPProcessor, CLIPModel
            model_name = "openai/clip-vit-base-patch32"
            logger.info(f"加载 CLIP 模型: {model_name}")
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()

            # 预计算各类别文本嵌入（多模板平均）
            all_embeddings = []
            for label in self.labels:
                templates = _make_templates(label)
                text_inputs = self.processor(
                    text=templates, return_tensors="pt", padding=True
                ).to(self.device)
                with torch.no_grad():
                    text_emb = self.model.get_text_features(**text_inputs)
                    if hasattr(text_emb, 'pooler_output'):
                        text_emb = text_emb.pooler_output
                    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
                    avg_emb = text_emb.mean(dim=0, keepdim=True)
                    avg_emb = torch.nn.functional.normalize(avg_emb, dim=-1)
                    all_embeddings.append(avg_emb)
            self.text_embeddings = torch.cat(all_embeddings, dim=0)
            logger.info(
                f"CLIP 分类器就绪 (device={self.device}, classes={self.labels})"
            )
            self._warmup()
        except Exception as e:
            logger.error(f"CLIP 加载失败: {e}")
            self.model = None

    def _warmup(self):
        try:
            dummy = Image.new("RGB", (224, 224))
            _ = self.predict(dummy)
            logger.info("CLIP 预热完成")
        except Exception:
            pass

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """CLIP 零样本分类"""
        if self.model is None:
            return {"label": self.labels[0], "confidence": 0.0}

        # 图片编码
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**img_inputs)
            if hasattr(img_emb, 'pooler_output'):
                img_emb = img_emb.pooler_output
            img_emb = torch.nn.functional.normalize(img_emb, dim=-1)

        # 余弦相似度
        logits = (100.0 * img_emb @ self.text_embeddings.T).squeeze(0)
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        top_idx = int(np.argmax(probs))
        return {
            "label": self.labels[top_idx],
            "confidence": float(probs[top_idx]),
            "all_scores": {label: float(p) for label, p in zip(self.labels, probs)},
        }


# ============================================================
# 单例管理 — 按用途分实例
# ============================================================
_shape_classifier: Optional[CLIPClassifier] = None
_scene_classifier: Optional[CLIPClassifier] = None


def get_shape_classifier() -> CLIPClassifier:
    """获取形状分类器（任务3）"""
    global _shape_classifier
    if _shape_classifier is None:
        _shape_classifier = CLIPClassifier(labels=SHAPE_LABELS)
    return _shape_classifier


def get_scene_classifier() -> CLIPClassifier:
    """获取场景分类器（备用）"""
    global _scene_classifier
    if _scene_classifier is None:
        _scene_classifier = CLIPClassifier(labels=CLASSIFY_LABELS)
    return _scene_classifier


def get_classifier() -> CLIPClassifier:
    """兼容旧接口：返回形状分类器"""
    return get_shape_classifier()
