"""
通用 CLIP 零样本分类器（决赛优化版）
====================================
支持任意类别集合（几何体形状/场景），自动生成中英文 prompt 模板。

任务3 形状分类：
  - 每个形状 6 条专用模板（中英混合 + 几何特征描述）
  - 模板描述俯拍视角（"viewed from above / top view"）——决赛相机朝下
  - 多模板平均嵌入，英文模板占比高（CLIP 英文训练数据远多于中文）

注意：torch/numpy 延迟导入 — 无 ML 环境时模块仍可导入，
模型加载失败走降级路径（predict 返回默认标签）。
"""
import logging
import os
from typing import Dict, Any, Optional, List

from config import CLASSIFY_LABELS, SHAPE_LABELS

logger = logging.getLogger(__name__)

# ============================================================
# 几何体形状专用 prompt 模板集（任务3）
# ============================================================
# 每条模板描述几何特征（轮廓/顶面/截面）而非场景。
# 决赛相机俯拍 → 强调 top view / viewed from above 视角。
# 每形状 6 条：2 中文 + 4 英文（CLIP 英文表征更强，故占比更高）。
SHAPE_TEMPLATES: Dict[str, List[str]] = {
    "长方体": [
        "长方体的俯视图，长方形顶面",
        "一个长方体几何体，矩形轮廓",
        "a cuboid viewed from above with a rectangular top face",
        "a rectangular prism with an elongated rectangular outline",
        "a rectangular box, top view",
        "a cuboid geometric solid",
    ],
    "正方体": [
        "正方体的俯视图，正方形顶面",
        "一个正方体几何体，方形轮廓",
        "a cube viewed from above with a square top face",
        "a cube with equal square faces",
        "a square box, top view",
        "a cube geometric solid",
    ],
    "圆柱体": [
        "圆柱体的俯视图，圆形顶面",
        "一个圆柱体几何体，圆形轮廓",
        "a cylinder viewed from above with a circular top face",
        "a cylinder with a round cross-section",
        "a round column, top view",
        "a cylinder geometric solid",
    ],
    "球体": [
        "球体的俯视图，圆形轮廓",
        "一个球体几何体",
        "a sphere viewed from above with a circular outline",
        "a round ball with a spherical surface",
        "a sphere, top view",
        "a sphere geometric solid",
    ],
    "三棱柱": [
        "三棱柱的俯视图，三角形顶面",
        "一个三棱柱几何体，三角形轮廓",
        "a triangular prism viewed from above with a triangular top face",
        "a triangular prism with three flat sides",
        "a triangular block, top view",
        "a triangular prism geometric solid",
    ],
    "六棱柱": [
        "六棱柱的俯视图，六边形顶面",
        "一个六棱柱几何体，六边形轮廓",
        "a hexagonal prism viewed from above with a hexagonal top face",
        "a hexagonal prism with six flat sides",
        "a hexagonal block, top view",
        "a hexagonal prism geometric solid",
    ],
    "圆锥": [
        "圆锥的俯视图，圆形底面",
        "一个圆锥几何体，尖顶圆形底",
        "a cone viewed from above with a circular base",
        "a cone with a pointed tip and round base",
        "a cone, top view",
        "a cone geometric solid",
    ],
    "四棱锥": [
        "四棱锥的俯视图，方形底面",
        "一个四棱锥几何体，尖顶方形底",
        "a pyramid viewed from above with a square base",
        "a pyramid with a pointed tip and square base",
        "a square pyramid, top view",
        "a pyramid geometric solid",
    ],
    "多面体": [
        "多面体的俯视图，多边形轮廓",
        "一个多面体几何体，多个平面",
        "a polyhedron viewed from above with a polygonal outline",
        "a polyhedron with multiple flat faces",
        "a many-sided geometric solid, top view",
        "a polyhedron geometric solid",
    ],
    "椭球体": [
        "椭球体的俯视图，椭圆形轮廓",
        "一个椭球体几何体，椭圆外形",
        "an ellipsoid viewed from above with an oval outline",
        "an ellipsoid with a smooth curved surface",
        "an oval geometric solid, top view",
        "an ellipsoid geometric solid",
    ],
}


def _make_templates(label: str) -> List[str]:
    """为任意类别生成 prompt 模板：形状走专用模板集，其余走通用模板"""
    if label in SHAPE_TEMPLATES:
        return list(SHAPE_TEMPLATES[label])
    return [
        f"a photo of a {label}",
        f"a clear image of {label}",
        f"{label}的图片",
    ]


class CLIPClassifier:
    """CLIP 零样本分类器（多模板集成）"""

    def __init__(self, labels: Optional[List[str]] = None):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = labels if labels is not None else SHAPE_LABELS
        self.model = None
        self.processor = None
        self.text_embeddings = None
        self._init_model()

    def _init_model(self):
        import torch
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
            from PIL import Image
            dummy = Image.new("RGB", (224, 224))
            _ = self.predict(dummy)
            logger.info("CLIP 预热完成")
        except Exception:
            pass

    def predict(self, image) -> Dict[str, Any]:
        """CLIP 零样本分类"""
        import torch
        import numpy as np
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
