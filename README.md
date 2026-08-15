# 中控杯决赛 — 汪汪队算法服务

> 2026年第二届"中控杯"智能制造挑战赛 · 赛道一「AI赋能」赛题2「工业多模态感知与无人化智能操作」决赛
> 队伍: 汪汪队 | 决赛: 2026.8.19-20 杭州白马湖

---

## 项目概览

决赛以**单臂机械臂 + 单灵巧手 + Gemini335 深度相机**为载体，三项递进式任务：

| 任务 | 内容 | 核心能力 |
|------|------|---------|
| 任务1 | 拨按开关 | 视觉定位亮灯 → 点按/拨动开关 |
| 任务2 | 长方体有序转运 | 数字识别 1-4 → 按序抓取放置 |
| 任务3 | 几何体无序分拣 | 形状识别 → 分拣入槽 |

**API 契约**（竞赛软件 ↔ 选手算法服务）:

```
GET  /api/health          → {"success": true, "message": "ready"}
POST /api/task1/execute   → 完成开关操作
POST /api/task2/execute   → 完成长方体转运
POST /api/task3/execute   → 完成几何体分拣
```

## 架构

```
竞赛操作软件 ──HTTP──> 算法服务(本仓库) ──HTTP──> 机械臂 FTArm B9 (:8087)
                                    ├──HTTP──> 灵巧手 DexHand (:5001)
                                    └──USB───> Gemini335 相机
```

## 快速开始

### 本地开发（Mock 模式）

```bash
pip install -r requirements.txt
python tests/test_with_mock.py        # 全流程测试（无需硬件）
```

### 现场部署（Docker，推荐）

```bash
# 在家构建镜像（含全部模型）
bash build_image.sh                    # 产出 汪汪队决赛镜像.tar

# 现场工控机（U盘拷贝 tar 后）
bash run_on_site.sh                    # 自动 load + run + 自检
```

### 现场调试（非 Docker 备用方案）

```bash
python scripts/hardware_check.py       # 硬件自检
python scripts/preheat.py              # 模型预热（必须！）
python scripts/debug_tools.py          # 交互式调试控制台
python app.py                          # 启动服务
```

## 现场标定流程（关键路径）

```
1. 打印棋盘格（9x7 内角，格宽 25mm）
2. python scripts/calibrate.py --mode all
   ├── 阶段A: 相机内参（20张棋盘格照片）
   ├── 阶段B: 手眼标定（12组位姿对）
   └── 阶段C: 生成 calibration.json
3. python scripts/apply_calibration.py   # 注入标定结果
4. 用 debug_tools.py 测量各目标物坐标 → 填入 config.py
5. python scripts/preheat.py             # 预热
6. 用 debug_tools.py 选项11 模拟竞赛软件调用测试
```

## 项目结构

```
final_algo_service/
├── app.py                    # FastAPI 主服务 (4端点)
├── config.py                 # ⚠️ 所有坐标参数（现场修改）
├── Dockerfile                # Docker 镜像定义
├── docker-compose.yml        # 容器编排
├── build_image.sh            # 构建+导出镜像
├── run_on_site.sh            # 现场一键部署
├── requirements.txt
├── hardware/
│   ├── arm_client.py         # FTArm B9 机械臂 HTTP SDK
│   └── hand_client.py        # 灵巧手 HTTP SDK
├── vision/
│   ├── vision_manager.py     # 视觉总控
│   ├── camera.py             # Gemini335 相机接口
│   ├── classifier.py         # CLIP 零样本分类
│   ├── detector.py           # Grounding DINO 检测
│   └── ocr_engine.py         # EasyOCR 数字识别
├── tasks/
│   ├── task1_switch.py       # 任务1: 拨按开关
│   ├── task2_cubes.py        # 任务2: 有序转运
│   └── task3_shapes.py       # 任务3: 形状分拣
├── scripts/
│   ├── calibrate.py          # 手眼标定半自动化
│   ├── apply_calibration.py  # 标定结果注入
│   ├── preheat.py            # 模型预热
│   ├── debug_tools.py        # 现场调试控制台
│   └── hardware_check.py     # 硬件自检
└── tests/
    ├── mock_hardware.py      # Mock 硬件
    └── test_with_mock.py     # 本地全流程测试
```

## 技术方案

### 视觉
- **形状识别** (任务3): CLIP ViT-B/32 零样本分类，初赛 18/18 满分方案复用
- **数字识别** (任务2): EasyOCR + 后处理纠错，初赛 L1/L2 满分方案复用
- **目标检测** (任务1/2/3): Grounding DINO tiny 开放词汇检测

### 运动控制
- 安全优先：默认 12% 关节限速，接近目标时降至 8%
- 标准序列：安全高度 → 水平接近 → 垂直下降 → 抓取 → 垂直上升 → 水平撤退
- 安全工作域检查：Y ∈ [-0.28, -0.04], Z ∈ [0.44, 0.52]

### 灵巧手
- 按物体类型选择抓取策略：cube/cylinder/toggle/button
- 抓取力度可调（默认 0.6 归一化）

## 已知风险与应对

| 风险 | 应对 |
|------|------|
| 现场坐标未知 | 标定脚本 + 调试工具半自动化测量 |
| 模型首次加载慢 | 预热脚本 + 镜像预下载模型 |
| 工控机无外网 | 模型全部打包进 Docker 镜像 |
| 相机 SDK 缺失 | 现场安装 Orbbec SDK（U盘携带安装包） |
| 灵巧手 API 变更 | hand_client.py 接口集中，易适配 |

## 交接清单

见上级目录 `交接清单-现场必读.md`
