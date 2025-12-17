# 后端开发指南 (Backend)

## 技术栈

- **框架**: FastAPI
- **数据库**: PostgreSQL + SQLAlchemy
- **缓存/队列**: Redis
- **文件存储**: MinIO
- **认证**: JWT

## 目录结构

```
backend/
├── app/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置管理
│   ├── dependencies.py         # 依赖注入
│   ├── routers/
│   │   ├── auth.py             # 认证路由
│   │   ├── users.py            # 用户路由
│   │   ├── sessions.py         # 会话路由
│   │   ├── transcripts.py      # 转录路由
│   │   ├── vocabulary.py       # 词汇路由
│   │   └── training.py         # 训练任务路由
│   ├── models/
│   │   ├── user.py             # 用户模型
│   │   ├── session.py          # 会话模型
│   │   ├── transcript.py       # 转录模型
│   │   └── training_task.py    # 训练任务模型
│   ├── schemas/
│   │   ├── user.py             # Pydantic Schema
│   │   └── ...
│   ├── services/
│   │   ├── user_service.py     # 用户业务逻辑
│   │   ├── session_service.py  # 会话业务逻辑
│   │   ├── asr_service.py      # ASR 服务调用
│   │   └── agent_service.py    # Agent 服务调用
│   └── utils/
│       ├── security.py         # JWT 相关
│       └── storage.py          # MinIO 客户端
├── alembic/                    # 数据库迁移
├── tests/
└── requirements.txt
```

## 快速开始

### 1. 环境准备

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 到 `.env` 并填写配置：

```bash
cp ../.env.example .env
```

### 3. 启动服务

```bash
# 确保 Redis 和 PostgreSQL 已启动
docker run -d -p 6379:6379 redis:7-alpine
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16

# 运行数据库迁移
alembic upgrade head

# 启动开发服务器
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 核心功能

### 1. 用户认证

- JWT Token 认证
- 支持邮箱/手机号登录
- 第三方 OAuth（可选）

### 2. 会话管理

- 创建/查询/删除会话
- 会话历史记录
- 会话上下文维护

### 3. 转录服务

通过 Agent SDK 调用 ASR Worker：

```python
from agent_sdk import AgentSDK

async def transcribe(audio_data: bytes):
    sdk = AgentSDK.from_env()
    result = await sdk.transcribe(audio_data)
    return result.text
```

### 4. Agent 对话

```python
async def chat(user_input: str, context: dict):
    sdk = AgentSDK.from_env()
    result = await sdk.chat_text(user_input, context=context)
    return result.output.text
```

## 数据库设计

### 用户表 (users)
| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| email | String | 邮箱 |
| phone | String | 手机号 |
| password_hash | String | 密码哈希 |
| created_at | DateTime | 创建时间 |

### 会话表 (sessions)
| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| user_id | UUID | 用户 ID |
| title | String | 会话标题 |
| created_at | DateTime | 创建时间 |

### 转录表 (transcripts)
| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| session_id | UUID | 会话 ID |
| text | Text | 转录文本 |
| audio_url | String | 音频文件 URL |
| created_at | DateTime | 创建时间 |

## API 端点

参考 `docs/API_SPECIFICATION.md` 获取完整 API 文档。

### 认证
- `POST /api/v1/auth/register` - 用户注册
- `POST /api/v1/auth/login` - 用户登录
- `POST /api/v1/auth/refresh` - 刷新 Token

### 用户
- `GET /api/v1/users/me` - 获取当前用户
- `PUT /api/v1/users/me` - 更新用户信息

### 会话
- `GET /api/v1/sessions` - 获取会话列表
- `POST /api/v1/sessions` - 创建会话
- `GET /api/v1/sessions/{id}` - 获取会话详情

### AI 服务
- `POST /api/v1/asr/transcribe` - 语音转文字
- `POST /api/v1/agent/chat` - 对话交互

## 与 AI Workers 交互

后端**不直接**调用 AI 模型，而是通过 Agent SDK 与 AI Workers 通信：

```
Backend -> Agent SDK -> Redis Stream -> AI Worker
                                 └─> 结果返回
```

这种设计允许：
- AI Workers 独立部署在 GPU 机器
- 自动负载均衡
- 故障隔离

## 相关文档

- [Agent SDK 文档](../agent-sdk/README.md)
- [API 规范](../docs/API_SPECIFICATION.md)
- [架构设计](../docs/ARCHITECTURE.md)
- [主项目 README](../README.md)
