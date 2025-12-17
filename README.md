# 构音障碍无障碍语音助手

针对构音障碍人群的无障碍语音产品，提供实时语音转录、智能纠错和语音主持人功能。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              客户端层                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      PWA Frontend (Next.js)                      │   │
│  │     实时转录 UI │ 语音主持人 UI │ 历史记录 │ 设置中心             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ HTTP / WebSocket
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                             业务后端服务                                  │
│  ┌───────────────────────┐    ┌───────────────────────────────────┐    │
│  │     Backend           │    │          Agent SDK                │    │
│  │  (FastAPI + PG)       │    │   (Redis Stream + Worker Pool)    │    │
│  │                       │    │                                   │    │
│  │  • 用户管理           │←──→│  • 任务调度                       │    │
│  │  • 会话管理           │    │  • 负载均衡                       │    │
│  │  • 数据持久化         │    │  • 健康检查                       │    │
│  └───────────────────────┘    └──────────────┬────────────────────┘    │
└──────────────────────────────────────────────┼──────────────────────────┘
                                               │ Redis Stream
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ↓                          ↓                          ↓
          ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
          │   ASR Worker    │        │  Agent Worker   │        │   TTS Worker    │
          │   (Machine A)   │        │   (Machine B)   │        │   (Machine C)   │
          │                 │        │                 │        │                 │
          │ • SenseVoice    │        │ • LLM 对话      │        │ • 语音合成      │
          │ • Whisper       │        │ • 意图识别      │        │                 │
          │ • GER 纠错      │        │ • 工具调用      │        │                 │
          │ • GPU 推理      │        │                 │        │                 │
          └─────────────────┘        └─────────────────┘        └─────────────────┘
```

---

## 项目结构

```
dysarthria-voice-assistant/
│
├── frontend/                      # 🎨 前端 (Next.js PWA)
│   └── README.md                  # 前端开发指南
│
├── backend/                       # 🔧 后端 (FastAPI)
│   └── README.md                  # 后端开发指南
│
├── agent-sdk/                     # 📦 Agent SDK（核心中间层）
│   ├── agent_sdk/
│   │   ├── client.py              # SDK 客户端入口
│   │   ├── config.py              # 配置管理
│   │   ├── worker_base.py         # Worker 基类
│   │   ├── stream/                # Redis Stream 管理
│   │   ├── dispatcher/            # 任务调度
│   │   └── pool/                  # Worker 池管理
│   └── README.md                  # SDK 使用文档
│
├── ai-workers/                    # 🤖 AI Worker 示例
│   ├── asr_worker_example.py      # ASR Worker 示例
│   └── agent_worker_example.py    # Agent Worker 示例
│
├── shared/                        # 📋 共享数据模型
│   └── models/
│       ├── common.py              # 通用模型
│       ├── asr.py                 # ASR 模型
│       └── agent.py               # Agent 模型
│
├── docs/                          # 📚 文档
│   ├── ARCHITECTURE.md            # 架构设计文档
│   └── API_SPECIFICATION.md       # 接口规范文档
│
├── tests/                         # 🧪 测试
│   ├── test_agent_sdk.py          # SDK 单元测试
│   ├── test_ai_workers.py         # Worker 测试
│   └── test_e2e.py                # 端到端测试
│
├── .env.example                   # ⚙️ 环境变量模板
├── PRD.md                         # 产品需求文档
└── README.md                      # 本文件
```

---

## 团队分工

| 角色 | 目录 | 主要职责 |
|------|------|----------|
| **前端** | `frontend/` | UI 界面、音频采集、WebSocket、PWA |
| **后端** | `backend/` | 用户认证、会话管理、数据持久化、API |
| **AI** | `ai-workers/` | ASR 引擎、LLM 对话、GER 纠错、TTS |

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd dysarthria-voice-assistant

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装 Agent SDK
pip install -e ./agent-sdk

# 安装测试依赖
pip install -r tests/requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填写必要配置
```

### 3. 启动 Redis

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### 4. 运行测试

```bash
# 设置 PYTHONPATH
export PYTHONPATH=$(pwd)

# 运行全部测试
pytest tests/ -v

# 运行单元测试（不需要 Redis）
pytest tests/test_agent_sdk.py tests/test_ai_workers.py -v
```

---

## 环境变量配置 (.env)

完整的环境变量模板在 `.env.example` 文件中，以下是核心配置说明：

### 基础配置（所有服务都需要）

| 变量名 | 描述 | 示例值 |
|--------|------|--------|
| `REDIS_URL` | Redis 连接地址 | `redis://localhost:6379` |
| `REDIS_PASSWORD` | Redis 密码 | 留空或填写密码 |
| `DATABASE_URL` | PostgreSQL 连接地址 | `postgresql://user:pass@localhost:5432/db` |

### ASR Worker 配置

| 变量名 | 描述 | 示例值 |
|--------|------|--------|
| `ASR_WORKER_TYPE` | ASR 引擎类型 | `sensevoice` 或 `whisper` |
| `SENSEVOICE_MODEL_PATH` | SenseVoice 模型路径 | `/models/sensevoice` |
| `SENSEVOICE_DEVICE` | 推理设备 | `cuda:0` |
| `GER_ENABLED` | 是否启用 GER 纠错 | `true` |
| `GER_MODEL_PATH` | GER 模型路径 | `/models/ger` |

### Agent Worker 配置

| 变量名 | 描述 | 示例值 |
|--------|------|--------|
| `AGENT_WORKER_TYPE` | Agent 类型 | `qwen` / `openai` / `anthropic` |
| `QWEN_MODEL_PATH` | Qwen 模型路径 | `/models/qwen` |
| `OPENAI_API_KEY` | OpenAI API Key | `sk-xxx` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | `sk-ant-xxx` |

### TTS 配置

| 变量名 | 描述 | 示例值 |
|--------|------|--------|
| `TTS_TYPE` | TTS 引擎类型 | `edge_tts` |
| `EDGE_TTS_VOICE` | 语音角色 | `zh-CN-XiaoxiaoNeural` |

---

## 各团队开发指南

### 🎨 前端开发

```bash
cd frontend
cat README.md  # 查看前端开发指南
```

**技术栈**: Next.js 14 + React 18 + Tailwind CSS + Zustand

**关键文档**:
- `frontend/README.md` - 前端开发详细指南
- `docs/API_SPECIFICATION.md` - API 和 WebSocket 规范

### 🔧 后端开发

```bash
cd backend
cat README.md  # 查看后端开发指南
```

**技术栈**: FastAPI + SQLAlchemy + PostgreSQL + Redis

**快速示例**：

```python
from agent_sdk import AgentSDK, SDKConfig

# 通过 SDK 调用 ASR 服务
async def transcribe_audio(audio_data: bytes, user_id: str):
    config = SDKConfig.from_env()
    async with AgentSDK(config) as sdk:
        result = await sdk.transcribe_file("audio.wav", user_id=user_id)
        return result.text
```

**关键文档**:
- `backend/README.md` - 后端开发详细指南
- `docs/API_SPECIFICATION.md` - 完整 API 规范
- `agent-sdk/README.md` - SDK 使用方法

### 🤖 AI Worker 开发

```bash
# 查看示例代码
cat ai-workers/asr_worker_example.py
cat ai-workers/agent_worker_example.py
```

**实现 ASR Worker**：

```python
from agent_sdk import SDKConfig, ASRWorkerBase

class MyASRWorker(ASRWorkerBase):
    async def setup(self):
        """加载模型"""
        from funasr import AutoModel
        self.model = AutoModel(model="iic/SenseVoiceSmall", device="cuda")

    async def transcribe(self, audio_data: bytes, options: dict) -> str:
        """执行转录（替换此实现）"""
        result = self.model.generate(input=audio_data)
        return result[0]["text"]

# 启动
if __name__ == "__main__":
    import asyncio
    config = SDKConfig.from_env()
    worker = MyASRWorker(config)
    asyncio.run(worker.start())
```

**实现 Agent Worker**：

```python
from agent_sdk import SDKConfig, AgentWorkerBase

class MyAgentWorker(AgentWorkerBase):
    async def setup(self):
        """初始化 LLM"""
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic()

    async def generate_response(self, input_text: str, context: dict, options: dict) -> dict:
        """生成响应（替换此实现）"""
        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": input_text}],
            max_tokens=500
        )
        return {"text": response.content[0].text, "intent": "general", "tools": [], "confidence": 0.9}
```

**关键文档**:
- `ai-workers/` - Worker 示例代码
- `agent-sdk/agent_sdk/worker_base.py` - Worker 基类定义
- `shared/models/` - 数据模型定义

---

## 多机器部署

### 部署架构

```
┌─────────────────┐
│   Machine A     │
│  (API Server)   │
│                 │
│  • Backend      │
│  • Redis        │
│  • PostgreSQL   │
└────────┬────────┘
         │ Redis Stream
    ┌────┴────────────────────┐
    │                         │
┌───┴───┐               ┌─────┴─────┐
│Machine│               │  Machine  │
│   B   │               │     C     │
│(GPU)  │               │  (GPU)    │
│       │               │           │
│• ASR  │               │• Agent    │
│Worker │               │  Worker   │
└───────┘               └───────────┘
```

### Machine A（主服务器）

```bash
# 启动 Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 启动 PostgreSQL
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=your_password postgres:16

# 启动后端服务
cd backend
export REDIS_URL=redis://localhost:6379
export DATABASE_URL=postgresql://postgres:your_password@localhost:5432/dva
python -m app.main
```

### Machine B（ASR Worker - GPU）

```bash
pip install -e ./agent-sdk

export REDIS_URL=redis://machine_a_ip:6379
export ASR_WORKER_TYPE=sensevoice
export SENSEVOICE_MODEL_PATH=/models/sensevoice

python ai-workers/asr_worker_example.py
```

### Machine C（Agent Worker）

```bash
pip install -e ./agent-sdk

export REDIS_URL=redis://machine_a_ip:6379
export ANTHROPIC_API_KEY=your_api_key

python ai-workers/agent_worker_example.py
```

---

## 文档索引

| 文档 | 描述 |
|------|------|
| [前端开发指南](./frontend/README.md) | 前端技术栈和开发流程 |
| [后端开发指南](./backend/README.md) | 后端 API 和数据库设计 |
| [Agent SDK 文档](./agent-sdk/README.md) | SDK 使用方法 |
| [架构设计文档](./docs/ARCHITECTURE.md) | 系统架构详解 |
| [API 规范文档](./docs/API_SPECIFICATION.md) | HTTP/WebSocket API |
| [环境变量模板](./.env.example) | 所有配置项说明 |
| [产品需求文档](./PRD.md) | 产品功能定义 |

---

## 测试

```bash
# 全部测试（需要 Redis）
pytest tests/ -v

# 单元测试
pytest tests/test_agent_sdk.py -v
pytest tests/test_ai_workers.py -v

# 端到端测试
pytest tests/test_e2e.py -v
```

---

## FAQ

**Q: ai-workers 目录下的示例代码需要删除吗？**

A: 不需要。这些是**示例代码**，展示如何实现 Worker。AI 同事应该参考这些示例，替换其中的 `transcribe()` 和 `generate_response()` 方法实现。

**Q: 前端和后端代码在哪里？**

A:
- `frontend/` - 前端项目目录（待实现）
- `backend/` - 后端项目目录（待实现）

各目录下有详细的 README.md 说明开发规范和技术栈。

**Q: 如何验证 Worker 已连接？**

A: 检查 Redis 中的心跳：
```bash
redis-cli KEYS "heartbeat:*"
```

---

## License

MIT License