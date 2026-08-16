# 中控杯 2026 决赛 — 汪汪队

> 2026 第二届"中控杯"智能制造挑战赛 · 赛道一「AI赋能」赛题2「工业多模态感知与无人化智能操作」
> **决赛**: 2026.8.19-20 杭州白马湖国际会展中心 B 馆（2号馆 = 外操无人化赛区）
> 队伍: 汪汪队 | 初赛仓库: [supcon-cup-2026](https://github.com/sherlock-miller/supcon-cup-2026)（master 分支）

---

## 决赛任务

以**单臂机械臂 + 灵巧手 + Gemini335 深度相机**为载体，三项递进式任务：

| 任务 | 内容 | 判分 |
|------|------|------|
| 任务1 | 拨按开关（视觉定位亮灯→点按/拨动对应开关） | 100 |
| 任务2 | 长方体有序转运（顶面数字 1-4 按序抓取放置） | 100 |
| 任务3 | 几何体无序分拣（形状识别→对应槽位，全部竖直摆放） | 100 |

**竞赛软件只调用 4 个 HTTP 端点**，其余全部由选手算法自主完成：

```
GET  /api/health          → {"success": true, "message": "ready"}
POST /api/task1/execute   → 完成开关操作后返回
POST /api/task2/execute   → 完成长方体转运后返回
POST /api/task3/execute   → 完成几何体分拣后返回
```

## 仓库结构

```
├── final_algo_service/        ← 算法服务核心（FastAPI + 机械臂SDK + 灵巧手SDK + 视觉 + 三任务编排）
│   ├── app.py                ← 4 端点主服务
│   ├── hardware/             ← FTArm B9 机械臂(:8087) + 灵巧手(:8088) HTTP SDK
│   ├── vision/               ← CLIP/GroundingDINO/EasyOCR + eye-in-hand 坐标变换
│   ├── tasks/                ← 三任务编排（安全序列+异常恢复）
│   ├── scripts/              ← 标定半自动化/预热/调试工具/硬件自检
│   ├── tests/                ← 47 个 pytest 用例（hermes verify 全绿）
│   └── Dockerfile + 部署脚本  ← Docker 镜像 / wheels 离线双轨部署
├── 相机服务/                  ← 宿主机相机 HTTP 服务（Docker 部署时 USB 透传方案）
├── 资料整理/                  ← 官方文档+群聊情报 豆包视觉识别存档
├── 架构与开发范围说明.md       ← 队友培训材料
└── 交接清单-现场必读.md        ← 现场必读（含 8.19/8.20 日程、分工、标定流程）
```

## 快速开始

```bash
cd final_algo_service

# 本地 Mock 测试（无需硬件）
pip install -r requirements.txt pytest
python -m pytest tests/ -q        # 47 passed

# 现场部署（二选一）
bash build_image.sh && bash run_on_site.sh    # Docker 镜像（U盘携带）
bash download_wheels.sh                        # wheels 离线包（Win11 工控机推荐）
```

## 关键情报（详见 资料整理/00-关键情报汇总.md）

- 工控机 **Win11 无预装环境**，现场每组仅 1 个有线网口（自备转接头）
- 机械臂 FTArm B9（HTTP :8087）+ 灵巧手（HTTP :8088），官方 HTTP API 跨平台
- 相机 Gemini335 装机械臂末端（eye-in-hand），SDK/内外参自备自标定
- 8.19 全天调试（19:00 前确认交卷锁屏），8.20 上午评委实操演示
- 任务2 数字仅顶面；任务3 几何体全部竖直摆放
