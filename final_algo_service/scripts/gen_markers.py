#!/usr/bin/env python3
"""
生成 ArUco / ChArUco 标定标记（PDF 可直接打印）
================================================
产物（输出到 标定标记/ 目录）:
  1. charuco_calib_board_A4.pdf     — ChArUco 标定板（相机内参标定）
  2. aruco_single_markers.pdf       — 单个 ArUco 标记（贴操作台定位基准）
  3. 标记说明.md                    — 打印/使用说明
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "标定标记")
os.makedirs(OUT_DIR, exist_ok=True)

# A4 300dpi: 2480 x 3508 px
A4_W, A4_H = 2480, 3508
MM2PX = 300 / 25.4  # px per mm


def aruco_marker_px(marker_id: int, dict_id, size_mm: float,
                    border_mm: float = 10.0) -> np.ndarray:
    """生成单个 ArUco 标记（RGB 图，自带白边 quiet zone ≥10mm）
    size_mm: 黑色标记部分尺寸；border_mm: 四周白边宽度（检测必需）"""
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    size_px = int(size_mm * MM2PX)
    marker = cv2.aruco.generateImageMarker(d, marker_id, size_px)
    border_px = int(border_mm * MM2PX)
    img = np.full((size_px + 2 * border_px, size_px + 2 * border_px), 255, np.uint8)
    img[border_px:border_px + size_px, border_px:border_px + size_px] = marker
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def charuco_board_px(squares_x: int, squares_y: int,
                     square_mm: float, marker_mm: float,
                     dict_id) -> np.ndarray:
    """生成 ChArUco 标定板（RGB 图）"""
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm, marker_mm,
        cv2.aruco.getPredefinedDictionary(dict_id))
    size_px = (int(squares_x * square_mm * MM2PX),
               int(squares_y * square_mm * MM2PX))
    img = board.generateImage(size_px)
    return img


def add_caption(img: Image.Image, text: str) -> Image.Image:
    """图下方加白底说明条"""
    W = img.width
    cap_h = 90
    canvas = Image.new("RGB", (W, img.height + cap_h), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 40)
    except Exception:
        font = ImageFont.load_default()
    d.text((30, img.height + 20), text, fill="black", font=font)
    return canvas


def main():
    pages = []

    # ===== 1. ChArUco 标定板（A4 一页）=====
    print("生成 ChArUco 标定板...")
    charuco = charuco_board_px(5, 7, 30.0, 22.0, cv2.aruco.DICT_6X6_250)
    # 居中放 A4
    page = Image.new("RGB", (A4_W, A4_H), "white")
    x0 = (A4_W - charuco.shape[1]) // 2
    y0 = 150
    page.paste(Image.fromarray(charuco), (x0, y0))
    d = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 60)
        font_s = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 40)
    except Exception:
        font = font_s = ImageFont.load_default()
    d.text((200, 60), "ChArUco 标定板 — 相机内参标定", fill="black", font=font)
    d.text((200, y0 + charuco.shape[0] + 40),
           "5x7 方格 / 方格 30mm / 标记 22mm / DICT_6X6_250   【按 100% 比例打印 A4】",
           fill="black", font=font_s)
    pages.append(page)

    # ===== 2. 单个 ArUco 标记页（贴操作台）=====
    # 标记自带 10mm 白边（quiet zone），打印页间距 15mm 即可
    # 布局: 3 列 x 4 行（每格总宽 60mm = 40 标记 + 2x10 白边）
    print("生成单个 ArUco 标记...")
    markers_per_row = 3
    rows = 4
    ids = list(range(12))  # ID 0-11
    margin = 60
    gap = int(15 * MM2PX)  # 打印页上标记间隔 15mm
    m_size = 40.0
    m_px = int(m_size * MM2PX) + 2 * int(10 * MM2PX)  # 含白边总宽

    page2 = Image.new("RGB", (A4_W, A4_H), "white")
    d2 = ImageDraw.Draw(page2)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 60)
        font_s = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 36)
    except Exception:
        font = font_s = ImageFont.load_default()
    d2.text((200, 60), "ArUco 单标记 — 操作台定位基准（DICT_4X4_50）",
            fill="black", font=font)
    d2.text((200, 140), "标记尺寸 40mm，剪下时四周留 ≥10mm 白边（quiet zone），贴在操作台/物体上。ID 0-11 可区分 12 个位置。",
            fill="black", font=font_s)

    y = 220
    for row in range(rows):
        x = margin
        for col in range(markers_per_row):
            idx = row * markers_per_row + col
            m = aruco_marker_px(ids[idx], cv2.aruco.DICT_4X4_50, m_size)
            page2.paste(Image.fromarray(m), (x, y))
            d2.text((x + 20, y + m_px + 15), f"ID {ids[idx]}", fill="black", font=font_s)
            x += m_px + gap
        y += m_px + gap + 120

    pages.append(page2)

    # ===== 3. 更大尺寸标记页（远距离）=====
    # 标记自带 10mm 白边，总宽 80mm。布局: 2 列 x 3 行
    print("生成大尺寸标记页...")
    page3 = Image.new("RGB", (A4_W, A4_H), "white")
    d3 = ImageDraw.Draw(page3)
    d3.text((200, 60), "ArUco 大标记 — 远距离/低分辨率场景（DICT_6X6_250）",
            fill="black", font=font)
    d3.text((200, 140), "标记尺寸 60mm（含 10mm 白边）。用于相机离得远或需要高稳定识别的场景。",
            fill="black", font=font_s)
    big_ids = [0, 1, 2, 3, 10, 20]
    big_px = int(60 * MM2PX) + 2 * int(10 * MM2PX)  # 含白边总宽
    y = 220
    for i in range(0, 6, 2):
        x = margin
        for idx in big_ids[i:i + 2]:
            m = aruco_marker_px(idx, cv2.aruco.DICT_6X6_250, 60.0)
            page3.paste(Image.fromarray(m), (x, y))
            d3.text((x + 25, y + big_px + 15), f"ID {idx}",
                    fill="black", font=font_s)
            x += big_px + gap
        y += big_px + gap + 140

    pages.append(page3)

    # ===== 输出 PDF =====
    out_pdf = os.path.join(OUT_DIR, "标定标记全集.pdf")
    pages[0].save(out_pdf, "PDF", resolution=300, save_all=True,
                  append_images=pages[1:])
    print(f"✅ PDF 已生成: {out_pdf} ({len(pages)} 页)")
    print(f"   - 第1页: ChArUco 标定板 (5x7, 30mm方格)")
    print(f"   - 第2页: ArUco 单标记 x12 (ID 0-11, DICT_4X4_50, 40mm)")
    print(f"   - 第3页: ArUco 大标记 x6 (DICT_6X6_250, 60mm)")


if __name__ == "__main__":
    main()
