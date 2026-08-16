#!/usr/bin/env python3
"""识别竞赛手册：文本层 + 豆包视觉双通道"""
import base64
import fitz
import requests

SRC = r"E:\hermes\中控杯决赛\01-原始资料\官方文件\05-全国总决赛竞赛手册-20260813.pdf"
OUT = r"E:\hermes\中控杯决赛\02-项目构建\资料整理\文档06-全国总决赛竞赛手册.md"

# 1. 文本层
doc = fitz.open(SRC)
text_parts = []
for i, page in enumerate(doc):
    text_parts.append(f"【第{i+1}页】\n{page.get_text()}")

# 2. 豆包视觉（Key 从环境变量读取）
import os
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
ARK_MODEL = "ep-20260528213610-cl26k"
uris = []
zoom = 200 / 72.0
matrix = fitz.Matrix(zoom, zoom)
for page in doc:
    pix = page.get_pixmap(matrix=matrix, colorspace="rgb")
    b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
    uris.append(f"data:image/png;base64,{b64}")
doc.close()

content = [{"type": "image_url", "image_url": {"url": u, "detail": "high"}} for u in uris]
content.append({"type": "text", "text": "这是中控杯竞赛手册的9页内容。请完整转录：所有标题、正文、表格、时间安排、地点、联系人、注意事项。不要遗漏。"})
resp = requests.post(
    "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    headers={"Authorization": f"Bearer {ARK_API_KEY}", "Content-Type": "application/json"},
    json={"model": ARK_MODEL, "messages": [{"role": "user", "content": content}],
          "max_tokens": 6000, "temperature": 0.3, "thinking": {"type": "disabled"}},
    timeout=300,
)
vision_text = resp.json()["choices"][0]["message"]["content"]

with open(OUT, "w", encoding="utf-8") as f:
    f.write("# 文档: 2026第二届中控杯全国总决赛竞赛手册\n\n")
    f.write("> 识别: 豆包视觉 + 文本层双通道 | 9页\n\n")
    f.write("## 豆包视觉识别\n\n" + vision_text + "\n\n---\n\n## 文本层提取\n\n" + "\n\n".join(text_parts))
print("✅ 竞赛手册识别完成")
