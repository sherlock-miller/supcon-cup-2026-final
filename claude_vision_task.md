# 中控杯决赛视觉管线优化任务

你是中控杯决赛（8.19 杭州）"汪汪队"队伍的算法优化工程师。请优化 vision/ 目录下的视觉管线。

## 背景
决赛三项任务（输入：RGB图+深度图；输出：目标类别+像素坐标）：
- 任务1 拨按开关：开关面板上3个灯（红/黄/绿）随机亮1个，需检测亮灯位置（像素坐标），灯下方有对应按钮/拨杆
- 任务2 长方体转运：4个长方体顶面印刷数字1-4（印刷清晰、表面哑光防反光），需识别数字并定位每个方块
- 任务3 几何体分拣：4个不同形状几何体（长方体/正方体/圆柱体/多面体等，未知具体种类，全部竖直摆放），需分类形状并定位

## 现状（初赛代码，已知问题）
1. detector.py 使用 Grounding DINO tiny，但从未在真实比赛场景验证过
2. classifier.py 用 CLIP 做形状分类，prompt 模板是场景分类遗留的（"a photo of an office"式），形状分类 prompt 未优化
3. ocr_engine.py 用 EasyOCR，后处理针对工业铭牌设计，决赛只需识别单个数字1-4
4. vision_manager.py 的 detect_lit_light 用亮度判断灯，不可靠

## 优化要求（按优先级）
1. **detector.py**：
   - 优化检测 prompt 为中文+英文混合的决赛场景词（亮灯、按钮、长方体、几何体）
   - 增加后处理：NMS去重、按面积过滤噪声、置信度自适应
   - 增加传统CV兜底：detect_lit_light 用 HSV 颜色空间检测红/黄/绿亮灯（比亮度可靠）
2. **classifier.py**：
   - 为几何体形状设计专门的 prompt 模板集（描述几何特征而非场景）
   - 多模板集成：每形状 5+ 模板（中英混合+几何描述）
3. **ocr_engine.py**：
   - 新增单数字识别模式：只认 1-4，可用模板匹配/特征规则兜底
   - 数字区域的预处理（对比度增强、二值化）
4. **vision_manager.py**：
   - detect_lit_light 重写：HSV 颜色阈值 + 候选区域验证（灯的位置布局先验：垂直排列3个）
   - detect_cube_numbers 重写：先找方块区域，再在区域内 OCR 数字，数字→方块位置映射
   - detect_and_classify_shapes 重写：DINO 检测 + CLIP 分类两阶段，加 shape 分类置信度阈值

## 约束
- 保持现有函数签名不变（tasks/ 调用这些函数）
- 保持延迟导入模式（无 ML 环境可 import）
- 中文注释
- 用 Python 3.10+ 语法
- 不要修改 config.py、tasks/、app.py
- 所有新代码要有降级路径（模型不可用时返回空/默认值，不崩溃）

## 验证
修改完成后运行：python -m pytest tests/test_suite.py -q（应全部通过）
语法检查：python -m py_compile vision/*.py

请开始工作。
