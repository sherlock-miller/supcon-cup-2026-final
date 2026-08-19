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
│   ├── tests/                ← 63 个 pytest 用例（全绿）
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
python -m pytest tests/ -q        # 63 passed

# 现场部署（三选一，推荐①镜像分发）
bash build_image.sh && bash run_on_site.sh    # Docker 镜像（U盘携带）
bash download_wheels.sh                        # wheels 离线包（Win11 工控机推荐）
docker pull ghcr.io/sherlock-miller/supcon-cup-2026-final:latest   # 见下方「镜像分发」章节
```

## 镜像分发

> **当前状态（2026-08-19）**：镜像已在本机构建并实测通过（`wangwang-final:latest`，
> 8.5GB 含全部模型权重，tar 包 8.6GB）。GitHub Actions CI 因账户 billing 锁定暂不可用，
> **队友获取镜像方式**：① U盘/移动硬盘拷贝 tar 包 → `docker load -i`；② 自行构建：
> `docker build -t wangwang-final:latest .`（Dockerfile 已含模型下载与全部修复）。

```bash
# 方式①: 离线导入（现场推荐）
docker load -i wangwang-final-image.tar
docker run -d --name wangwang -p 5000:5000 \
  -e ARM_BASE_URL=http://<机械臂IP>:8087 \
  -e HAND_BASE_URL=http://<灵巧手IP>:8088 \
  wangwang-final:latest

# 方式②: 本地构建（需网络下载模型）
cd final_algo_service && bash build_image.sh
```

> **容器端口注意**：容器内服务监听 **5000**（`ALGO_PORT` 环境变量可改），
> 与竞赛软件 Base URL `http://127.0.0.1:5000` 一致。
> 国内拉取基础镜像失败时：`docker pull docker.m.daocloud.io/library/python:3.10-slim`

## 关键情报（详见 资料整理/00-关键情报汇总.md）

- 工控机 **Win11 无预装环境**，现场每组仅 1 个有线网口（自备转接头）
- 机械臂 FTArm B9（HTTP :8087）+ 灵巧手（HTTP :8088），官方 HTTP API 跨平台
- 相机 Gemini335 装机械臂末端（eye-in-hand），SDK/内外参自备自标定
- 8.19 全天调试（19:00 前确认交卷锁屏），8.20 上午评委实操演示
- 任务2 数字仅顶面；任务3 几何体全部竖直摆放
- **任务1 三灯为红/白/绿**（2026-08-17 官方说明书更正，此前情报红黄绿）
- 竞赛软件与算法服务**同机部署**（Base URL http://127.0.0.1:5000），
  开赛指令后 30 秒内点"开始比赛"（超时罚分），暂停/提交验证码 OK
