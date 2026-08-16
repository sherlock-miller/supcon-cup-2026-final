#!/usr/bin/env python3
"""
批量识别脚本 — 用豆包视觉模型识别决赛资料
============================================
将所有截图、PDF 用豆包 Doubao-Seed-2.0-lite 视觉识别，
结果整理为 Markdown 存到 资料整理/ 文件夹。

用法:
    python batch_recognize.py
"""
import os
import sys
import base64
import json
import time
from pathlib import Path

import requests

# 豆包 API 配置（Key 从环境变量读取，禁止硬编码）
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
ARK_MODEL = "ep-20260528213610-cl26k"  # Doubao-Seed-2.0-lite
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# 目录配置
SRC_DIR = Path(r"E:\hermes\中控杯决赛\01-原始资料")
OUT_DIR = Path(r"E:\hermes\中控杯决赛\02-项目构建\资料整理")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIME_MAP = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "png": "image/png", "gif": "image/gif",
    "webp": "image/webp", "bmp": "image/bmp",
}


def file_to_base64_uri(file_path: str) -> str:
    ext = Path(file_path).suffix.lower().lstrip(".")
    mime = MIME_MAP.get(ext, "image/png")
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def pdf_to_base64_uris(file_path: str, dpi: int = 200, max_pages: int = 20) -> list:
    """PDF 渲染为图片列表"""
    import fitz
    doc = fitz.open(file_path)
    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        if len(images) >= max_pages:
            break
        pix = page.get_pixmap(matrix=matrix, colorspace="rgb")
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("ascii")
        images.append(f"data:image/png;base64,{b64}")
    doc.close()
    return images


def call_vision(image_uris: list, prompt: str) -> str:
    """调用豆包视觉 API"""
    content = []
    for uri in image_uris:
        content.append({"type": "image_url", "image_url": {"url": uri, "detail": "high"}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": ARK_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4096,
        "temperature": 0.3,
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


PROMPT_SCREENSHOT = """这是一个微信群聊的截图。请详细、完整地转录其中的内容：
1. 每条消息的发送时间、发送者昵称
2. 消息的完整文字内容（原文转录，不要概括）
3. 如果是文件分享，记录文件名和文件类型
4. 如果是图片，详细描述图片内容
5. 特别注意与「中控杯」「赛题」「机械臂」「灵巧手」「相机」「评分」「操作」相关的技术细节
按时间顺序组织输出。"""

PROMPT_PDF = """这是一个比赛文档的页面截图。请详细、完整地转录其中的内容：
1. 所有标题、正文文字（原文转录）
2. 所有表格数据（保持表格结构）
3. 所有图片/示意图的内容描述
4. 特别注意评分规则、接口规范、硬件参数等技术细节
不要遗漏任何信息，包括页脚、注释、图片中的文字。"""


def recognize_screenshots():
    """识别全部截图"""
    print("=" * 60)
    print("任务1: 识别群聊截图")
    print("=" * 60)

    screenshots = sorted(SRC_DIR.glob("ScreenShot_*.png"))
    print(f"共 {len(screenshots)} 张截图")

    all_results = []
    for i, shot in enumerate(screenshots):
        out_md = OUT_DIR / f"截图{i+1:02d}-{shot.stem}.md"
        if out_md.exists():
            print(f"[{i+1}/{len(screenshots)}] 已存在，跳过: {shot.name}")
            with open(out_md, "r", encoding="utf-8") as f:
                all_results.append(f.read())
            continue

        print(f"[{i+1}/{len(screenshots)}] 识别中: {shot.name} ...")
        try:
            uri = file_to_base64_uri(str(shot))
            result = call_vision([uri], PROMPT_SCREENSHOT)
            md_content = f"# 截图: {shot.name}\n\n> 识别模型: Doubao-Seed-2.0-lite\n\n{result}\n"
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(md_content)
            all_results.append(md_content)
            print(f"    ✅ 完成 ({len(result)} 字)")
        except Exception as e:
            print(f"    ❌ 失败: {e}")
            all_results.append(f"# 截图: {shot.name}\n\n> 识别失败: {e}\n")

        time.sleep(1)  # 限速

    # 汇总
    summary = "# 群聊截图识别汇总\n\n"
    for i, shot in enumerate(screenshots):
        summary += f"## 截图{i+1}: {shot.name}\n\n"
        summary += f"详情见: 截图{i+1:02d}-{shot.stem}.md\n\n"
    with open(OUT_DIR / "00-截图索引.md", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n截图识别完成: {len(screenshots)} 张")
    return all_results


def recognize_pdfs():
    """识别全部 PDF"""
    print("\n" + "=" * 60)
    print("任务2: 识别 PDF 文档")
    print("=" * 60)

    pdfs = sorted(SRC_DIR.glob("*.pdf"))
    print(f"共 {len(pdfs)} 个 PDF")

    for i, pdf in enumerate(pdfs):
        out_md = OUT_DIR / f"文档{i+1:02d}-{pdf.stem}.md"
        if out_md.exists():
            print(f"[{i+1}/{len(pdfs)}] 已存在，跳过: {pdf.name}")
            continue

        print(f"[{i+1}/{len(pdfs)}] 识别中: {pdf.name} ...")
        try:
            uris = pdf_to_base64_uris(str(pdf), max_pages=25)
            print(f"    渲染 {len(uris)} 页")

            # 分批识别（每批最多8页）
            all_parts = []
            for batch_start in range(0, len(uris), 8):
                batch = uris[batch_start:batch_start + 8]
                result = call_vision(batch, PROMPT_PDF)
                all_parts.append(result)
                time.sleep(1)

            md_content = f"# 文档: {pdf.name}\n\n> 识别模型: Doubao-Seed-2.0-lite | 共 {len(uris)} 页\n\n"
            md_content += "\n\n---\n\n".join(all_parts) + "\n"
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"    ✅ 完成")
        except Exception as e:
            print(f"    ❌ 失败: {e}")

    print("\nPDF 识别完成")


def extract_pdf_text_layer():
    """同时提取 PDF 文本层（对照用）"""
    print("\n" + "=" * 60)
    print("任务2b: 提取 PDF 文本层")
    print("=" * 60)
    import fitz
    pdfs = sorted(SRC_DIR.glob("*.pdf"))
    for pdf in pdfs:
        out_txt = OUT_DIR / f"文档-{pdf.stem}-文本层.md"
        if out_txt.exists():
            continue
        try:
            doc = fitz.open(str(pdf))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(f"# {pdf.name} — 文本层提取\n\n```\n{text}\n```\n")
            print(f"✅ {pdf.name} ({len(text)} 字)")
        except Exception as e:
            print(f"❌ {pdf.name}: {e}")


if __name__ == "__main__":
    print("中控杯决赛资料批量识别（豆包视觉）")
    print(f"输出目录: {OUT_DIR}")
    print()

    recognize_screenshots()
    recognize_pdfs()
    extract_pdf_text_layer()

    print("\n" + "=" * 60)
    print("全部识别完成！")
    print(f"结果保存在: {OUT_DIR}")
    print("=" * 60)
