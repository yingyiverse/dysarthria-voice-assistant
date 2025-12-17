# 系统架构设计文档

## 1. 总体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    客户端层                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                           PWA Frontend (Next.js)                             │   │
│  │   实时转录 UI │ 语音主持人 UI │ 历史记录 │ 设置中心 │ 个性化训练              │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │ HTTP / WebSocket
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   API 网关层                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                      API Gateway (Traefik / Kong)                            │   │
│  │            认证鉴权 │ 限流熔断 │ 路由转发 │ 负载均衡 │ 日志监控              │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ↓                       ↓                       ↓
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│    业务后端服务      │   │    Agent SDK        │   │   前端 BFF 服务     │
│   (biz-service)     │   │   (agent-sdk)       │   │  (可选，合并到后端)  │
│                     │   │                     │   │                     │
│  • 用户管理         │   │  • 任务调度         │   │  • SSE 推送         │
│  • 会话管理         │   │  • Redis Stream     │   │  • WebSocket 管理   │
│  • 数据持久化       │   │  • 进程池管理       │   │  • 前端聚合接口     │
│  • 配置管理         │   │  • HTTP 客户端      │   │                     │
│  • 文件存储         │   │  • 健康检查         │   │                     │
│                     │   │  • 指标收集         │   │                     │
│  FastAPI + PG       │   │  Python SDK         │   │  FastAPI            │
└──────────┬──────────┘   └──────────┬──────────┘   └─────────────────────┘
           │                         │
           │                         │ 调用
           │              ┌──────────┴──────────┐
           │              │                     │
           │              ↓                     ↓
           │   ┌─────────────────┐   ┌─────────────────┐
           │   │   ASR Worker    │   │  Agent Worker   │
           │   │  (ai-asr)       │   │  (ai-agent)     │
           │   │                 │   │                 │
           │   │ • SenseVoice    │   │ • LLM 对话      │
           │   │ • Whisper       │   │ • TTS 合成      │
           │   │ • GER 纠错      │   │ • 意图识别      │
           │   │ • VAD 检测      │   │ • 任务执行      │
           │   │                 │   │ • 工具调用      │
           │   │ GPU Worker      │   │ GPU/CPU Worker  │
           │   └─────────────────┘   └─────────────────┘
           │
           ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    数据层                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  PostgreSQL  │  │    Redis     │  │    MinIO     │  │      模型存储           │  │
│  │              │  │              │  │              │  │                        │  │
│  │ • 用户表     │  │ • 任务队列   │  │ • 音频文件   │  │ • SenseVoice 模型      │  │
│  │ • 会话表     │  │ • Stream     │  │ • 训练数据   │  │ • LoRA 权重            │  │
│  │ • 转录表     │  │ • 缓存       │  │ • 导出文件   │  │ • GER 模型             │  │
│  │ • 词汇表     │  │ • 会话状态   │  │              │  │ • TTS 模型             │  │
│  │ • 训练任务   │  │              │  │              │  │                        │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 各层职责详解

### 2.1 前端层 (PWA Frontend)

**技术栈**: Next.js 14 + React 18 + Tailwind CSS + Zustand

**职责**:
- UI 渲染和交互
- 音频采集 (Web Audio API)
- 本地 VAD 预处理
- WebSocket 连接管理
- 离线缓存 (Service Worker)
- PWA 安装体验

**不负责**:
- 业务逻辑处理
- 数据持久化
- AI 推理

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── page.tsx            # 首页/实时转录
│   │   ├── history/            # 历史记录
│   │   ├── assistant/          # 语音主持人
│   │   ├── training/           # 个性化训练
│   │   └── settings/           # 设置
│   ├── components/
│   │   ├── audio/              # 音频相关组件
│   │   │   ├── AudioRecorder.tsx
│   │   │   ├── VoiceVisualizer.tsx
│   │   │   └── StreamingPlayer.tsx
│   │   ├── transcript/         # 转录相关
│   │   └── ui/                 # 通用 UI 组件
│   ├── lib/
│   │   ├── api/                # API 客户端
│   │   ├── audio/              # 音频处理工具
│   │   └── websocket/          # WebSocket 客户端
│   ├── hooks/                  # React Hooks
│   └── stores/                 # Zustand 状态管理
├── public/
│   ├── manifest.json           # PWA 配置
│   └── sw.js                   # Service Worker
└── package.json
```

---

### 2.2 业务后端服务 (biz-service)

**技术栈**: FastAPI + SQLAlchemy + PostgreSQL + Redis

**职责**:
- 用户认证和授权 (JWT)
- 用户信息管理
- 会话 CRUD 操作
- 转录数据存储
- 用户词汇表管理
- 训练任务管理
- 文件上传/下载
- 配置管理

**与 Redis 的交互**:
- 缓存用户信息
- 缓存热点数据
- 存储会话状态
- **不直接**处理 AI 任务队列（交给 Agent SDK）

**与数据库的交互**:
- 所有持久化数据的 CRUD
- 事务管理
- 数据迁移

```
services/biz-service/
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
│   │   └── ...
│   └── utils/
│       ├── security.py         # JWT 相关
│       └── storage.py          # MinIO 客户端
├── alembic/                    # 数据库迁移
├── tests/
└── requirements.txt
```

---

### 2.3 Agent SDK (核心中间层)

**这是架构的关键层，负责解耦业务服务和 AI Worker**

**技术栈**: Python + Redis Streams + asyncio + aiohttp

**核心职责**:

| 模块 | 职责 | 详细说明 |
|------|------|----------|
| **TaskDispatcher** | 任务调度 | 将请求分发到正确的 Worker |
| **StreamManager** | Redis Stream 管理 | 生产/消费消息，确保可靠投递 |
| **WorkerPool** | 进程/线程池管理 | Worker 生命周期、健康检查、自动重启 |
| **HTTPClient** | HTTP 请求处理 | 封装对 AI Worker 的 HTTP 调用 |
| **ResultCollector** | 结果收集 | 聚合 Worker 返回的结果 |
| **MetricsCollector** | 指标收集 | 延迟、吞吐量、错误率等 |

**设计原则**:
1. **业务服务不直接调用 AI Worker** - 通过 SDK 中转
2. **AI Worker 不直接访问数据库** - 通过 SDK 中转
3. **SDK 负责所有基础设施** - 队列、池化、重试、监控

```
agent-sdk/
├── agent_sdk/
│   ├── __init__.py
│   ├── client.py               # SDK 客户端入口
│   ├── config.py               # SDK 配置
│   │
│   ├── dispatcher/
│   │   ├── __init__.py
│   │   ├── task_dispatcher.py  # 任务调度器
│   │   └── router.py           # 任务路由规则
│   │
│   ├── stream/
│   │   ├── __init__.py
│   │   ├── producer.py         # Redis Stream 生产者
│   │   ├── consumer.py         # Redis Stream 消费者
│   │   └── stream_manager.py   # Stream 管理器
│   │
│   ├── pool/
│   │   ├── __init__.py
│   │   ├── process_pool.py     # 进程池管理
│   │   ├── thread_pool.py      # 线程池管理
│   │   └── worker_manager.py   # Worker 生命周期管理
│   │
│   ├── http/
│   │   ├── __init__.py
│   │   ├── async_client.py     # 异步 HTTP 客户端
│   │   └── retry.py            # 重试策略
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── task.py             # 任务数据模型
│   │   ├── result.py           # 结果数据模型
│   │   └── worker.py           # Worker 状态模型
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # 指标收集
│       ├── health.py           # 健康检查
│       └── logger.py           # 日志工具
│
├── tests/
├── examples/
│   ├── simple_asr.py           # 简单 ASR 调用示例
│   └── streaming_asr.py        # 流式 ASR 示例
├── setup.py
└── requirements.txt
```

---

### 2.4 AI ASR Worker (ai-asr)

**技术栈**: Python + PyTorch + FunASR/Whisper

**职责** (纯 AI 功能):
- 语音识别 (ASR)
- 语音活动检测 (VAD)
- 语义纠错 (GER)
- 模型推理加速
- 流式处理

**不负责**:
- 用户认证
- 数据库操作
- 队列管理（由 SDK 处理）
- HTTP 服务（由 SDK 调用）

```
services/ai-asr/
├── app/
│   ├── main.py                 # Worker 入口
│   ├── config.py               # AI 配置
│   │
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── sensevoice.py       # SenseVoice 引擎
│   │   ├── whisper.py          # Whisper 引擎
│   │   └── base.py             # 引擎基类
│   │
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── vad.py              # VAD 处理器
│   │   ├── ger.py              # GER 纠错器
│   │   └── streaming.py        # 流式处理器
│   │
│   ├── models/                 # 模型权重存放
│   │   ├── sensevoice/
│   │   ├── lora_adapters/
│   │   └── ger/
│   │
│   └── handlers/
│       ├── __init__.py
│       ├── transcribe.py       # 转录处理
│       └── stream.py           # 流式处理
│
├── Dockerfile
└── requirements.txt
```

---

### 2.5 AI Agent Worker (ai-agent)

**技术栈**: Python + Pipecat/LangChain + LLM APIs

**职责** (纯 AI 功能):
- 对话管理
- 意图识别
- LLM 推理调用
- TTS 语音合成
- 工具调用 (Function Calling)
- 多轮对话上下文

**不负责**:
- 用户认证
- 数据库操作
- WebSocket 管理（由 SDK 处理）

```
services/ai-agent/
├── app/
│   ├── main.py                 # Worker 入口
│   ├── config.py
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── voice_pipeline.py   # 语音对话 Pipeline
│   │   └── text_pipeline.py    # 文本对话 Pipeline
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── router.py           # LLM 路由
│   │   ├── claude.py           # Claude 适配器
│   │   ├── qwen.py             # Qwen 适配器
│   │   └── ollama.py           # Ollama 适配器
│   │
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── cartesia.py         # Cartesia TTS
│   │   └── edge_tts.py         # Edge TTS (备选)
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── weather.py          # 天气工具
│   │   ├── reminder.py         # 提醒工具
│   │   └── search.py           # 搜索工具
│   │
│   └── prompts/
│       ├── __init__.py
│       ├── system.py           # 系统 Prompt
│       └── dysarthria.py       # 构音障碍专用 Prompt
│
├── Dockerfile
└── requirements.txt
```

---

## 3. Agent SDK 详细设计

### 3.1 核心类设计

```python
# agent_sdk/client.py

class AgentSDK:
    """Agent SDK 主入口"""

    def __init__(self, config: SDKConfig):
        self.config = config
        self.stream_manager = StreamManager(config.redis)
        self.worker_pool = WorkerPool(config.workers)
        self.task_dispatcher = TaskDispatcher(self.stream_manager, self.worker_pool)
        self.http_client = AsyncHTTPClient(config.http)
        self.metrics = MetricsCollector()

    async def submit_asr_task(
        self,
        audio_data: bytes,
        user_id: str,
        options: ASROptions
    ) -> TaskResult:
        """提交 ASR 任务"""
        task = ASRTask(
            task_id=generate_task_id(),
            user_id=user_id,
            audio_data=audio_data,
            options=options
        )
        return await self.task_dispatcher.dispatch(task)

    async def create_streaming_session(
        self,
        user_id: str,
        on_result: Callable
    ) -> StreamingSession:
        """创建流式转录会话"""
        session = StreamingSession(
            session_id=generate_session_id(),
            user_id=user_id,
            stream_manager=self.stream_manager,
            on_result=on_result
        )
        await session.start()
        return session

    async def create_agent_session(
        self,
        user_id: str,
        agent_type: str = "voice_assistant"
    ) -> AgentSession:
        """创建 Agent 会话"""
        session = AgentSession(
            session_id=generate_session_id(),
            user_id=user_id,
            agent_type=agent_type,
            dispatcher=self.task_dispatcher
        )
        return session
```

### 3.2 任务调度器设计

```python
# agent_sdk/dispatcher/task_dispatcher.py

class TaskDispatcher:
    """任务调度器"""

    def __init__(self, stream_manager: StreamManager, worker_pool: WorkerPool):
        self.stream_manager = stream_manager
        self.worker_pool = worker_pool
        self.routing_rules = RoutingRules()

    async def dispatch(self, task: BaseTask) -> TaskResult:
        """调度任务到对应的 Worker"""

        # 1. 确定目标 Worker 类型
        worker_type = self.routing_rules.get_worker_type(task)

        # 2. 获取可用的 Worker
        worker = await self.worker_pool.get_available_worker(worker_type)

        # 3. 发送任务到 Redis Stream
        stream_key = f"tasks:{worker_type}"
        message_id = await self.stream_manager.publish(stream_key, task.to_dict())

        # 4. 等待结果
        result = await self.wait_for_result(task.task_id, timeout=task.timeout)

        return result

    async def dispatch_streaming(
        self,
        task: StreamingTask,
        result_callback: Callable
    ) -> None:
        """调度流式任务"""

        worker_type = "asr_streaming"
        stream_key = f"tasks:{worker_type}"
        result_stream = f"results:{task.task_id}"

        # 发布任务
        await self.stream_manager.publish(stream_key, task.to_dict())

        # 订阅结果流
        async for result in self.stream_manager.subscribe(result_stream):
            await result_callback(result)
            if result.get("is_final"):
                break
```

### 3.3 Redis Stream 管理器

```python
# agent_sdk/stream/stream_manager.py

class StreamManager:
    """Redis Stream 管理器"""

    def __init__(self, redis_config: RedisConfig):
        self.redis = aioredis.from_url(redis_config.url)
        self.consumer_group = redis_config.consumer_group

    async def publish(self, stream: str, message: dict) -> str:
        """发布消息到 Stream"""
        message_id = await self.redis.xadd(
            stream,
            message,
            maxlen=10000  # 保留最近 10000 条
        )
        return message_id

    async def subscribe(self, stream: str) -> AsyncIterator[dict]:
        """订阅 Stream"""
        last_id = "0"
        while True:
            messages = await self.redis.xread(
                {stream: last_id},
                count=1,
                block=5000  # 5秒超时
            )
            for _, message_list in messages:
                for message_id, message in message_list:
                    last_id = message_id
                    yield message

    async def consume(
        self,
        stream: str,
        consumer_name: str,
        handler: Callable
    ) -> None:
        """消费 Stream 消息（Worker 使用）"""

        # 确保消费者组存在
        try:
            await self.redis.xgroup_create(stream, self.consumer_group, mkstream=True)
        except Exception:
            pass  # 组已存在

        while True:
            messages = await self.redis.xreadgroup(
                self.consumer_group,
                consumer_name,
                {stream: ">"},
                count=1,
                block=5000
            )

            for _, message_list in messages:
                for message_id, message in message_list:
                    try:
                        await handler(message)
                        await self.redis.xack(stream, self.consumer_group, message_id)
                    except Exception as e:
                        # 处理失败，不 ACK，消息会被重新投递
                        logger.error(f"Message processing failed: {e}")
```

### 3.4 Worker 池管理器

```python
# agent_sdk/pool/worker_manager.py

class WorkerPool:
    """Worker 池管理器"""

    def __init__(self, config: WorkerPoolConfig):
        self.config = config
        self.workers: Dict[str, List[WorkerInfo]] = {}
        self.health_checker = HealthChecker()

    async def register_worker(self, worker: WorkerInfo) -> None:
        """注册 Worker"""
        worker_type = worker.type
        if worker_type not in self.workers:
            self.workers[worker_type] = []
        self.workers[worker_type].append(worker)

        # 启动健康检查
        asyncio.create_task(self._health_check_loop(worker))

    async def get_available_worker(self, worker_type: str) -> WorkerInfo:
        """获取可用的 Worker（负载均衡）"""
        available = [
            w for w in self.workers.get(worker_type, [])
            if w.status == WorkerStatus.HEALTHY
        ]

        if not available:
            raise NoAvailableWorkerError(f"No available {worker_type} worker")

        # 简单的轮询负载均衡
        worker = min(available, key=lambda w: w.current_load)
        worker.current_load += 1
        return worker

    async def _health_check_loop(self, worker: WorkerInfo) -> None:
        """健康检查循环"""
        while True:
            try:
                is_healthy = await self.health_checker.check(worker)
                worker.status = WorkerStatus.HEALTHY if is_healthy else WorkerStatus.UNHEALTHY

                if not is_healthy:
                    logger.warning(f"Worker {worker.id} is unhealthy")
                    # 触发自动重启（如果配置了）
                    if self.config.auto_restart:
                        await self._restart_worker(worker)
            except Exception as e:
                logger.error(f"Health check failed: {e}")

            await asyncio.sleep(self.config.health_check_interval)
```

---

## 4. 数据流设计

### 4.1 实时转录数据流

```
┌──────────┐    音频块     ┌──────────┐   任务消息    ┌──────────────┐
│  前端    │ ──────────→  │ biz-api  │ ──────────→  │  Agent SDK   │
│  PWA     │  WebSocket   │          │    HTTP      │              │
└──────────┘              └──────────┘              └──────┬───────┘
                                                          │
                              ┌────────────────────────────┘
                              │  Redis Stream
                              ↓
                    ┌──────────────────┐
                    │   ai-asr Worker  │
                    │                  │
                    │  VAD → ASR → GER │
                    └────────┬─────────┘
                             │
                             │  结果
                             ↓
┌──────────┐   WebSocket   ┌──────────┐   HTTP/Stream  ┌──────────────┐
│  前端    │ ←───────────  │ biz-api  │ ←───────────  │  Agent SDK   │
│  PWA     │              │          │               │              │
└──────────┘              └──────────┘               └──────────────┘
```

### 4.2 语音 Agent 数据流

```
┌──────────┐                ┌──────────┐                ┌──────────────┐
│  前端    │    音频块      │ biz-api  │   创建会话    │  Agent SDK   │
│  PWA     │ ──────────→   │          │ ──────────→   │              │
└──────────┘   WebSocket    └──────────┘    HTTP       └──────┬───────┘
                                                              │
                   ┌──────────────────────────────────────────┘
                   │
     ┌─────────────┼─────────────┐
     │             │             │
     ↓             ↓             ↓
┌─────────┐  ┌──────────┐  ┌──────────┐
│ ai-asr  │  │ ai-agent │  │ ai-tts   │
│         │  │          │  │          │
│ 语音→   │→ │ LLM      │→ │ 文字→   │
│ 文字    │  │ 处理     │  │ 语音    │
└─────────┘  └──────────┘  └──────────┘
     │             │             │
     └─────────────┼─────────────┘
                   │
                   ↓
┌──────────┐   音频流    ┌──────────┐      结果      ┌──────────────┐
│  前端    │ ←────────   │ biz-api  │ ←───────────  │  Agent SDK   │
│  PWA     │            │          │               │              │
└──────────┘            └──────────┘               └──────────────┘
```

---

## 5. 接口数据格式

### 5.1 任务消息格式 (Redis Stream)

```typescript
// ASR 任务消息
interface ASRTaskMessage {
  task_id: string;           // 任务唯一 ID
  task_type: "asr" | "asr_streaming";
  user_id: string;           // 用户 ID
  session_id?: string;       // 会话 ID（流式场景）

  // 音频数据
  audio: {
    data: string;            // Base64 编码的音频
    format: "wav" | "pcm";   // 音频格式
    sample_rate: number;     // 采样率，默认 16000
    channels: number;        // 通道数，默认 1
  };

  // ASR 选项
  options: {
    engine: "sensevoice" | "whisper";  // ASR 引擎
    language: string;        // 语言，默认 "zh"
    enable_ger: boolean;     // 是否启用 GER 纠错
    use_personal_model: boolean;  // 是否使用个人模型
    model_id?: string;       // 个人模型 ID
  };

  // 元数据
  metadata: {
    created_at: string;      // ISO 时间戳
    priority: number;        // 优先级 1-10
    timeout_ms: number;      // 超时时间
  };
}

// ASR 结果消息
interface ASRResultMessage {
  task_id: string;
  session_id?: string;

  result: {
    text: string;            // 识别文本
    original_text?: string;  // 纠错前的原始文本
    confidence: number;      // 置信度 0-1
    is_final: boolean;       // 是否最终结果

    // 词级别信息（可选）
    words?: Array<{
      word: string;
      start_time: number;    // 开始时间（秒）
      end_time: number;      // 结束时间（秒）
      confidence: number;
    }>;
  };

  // 性能指标
  metrics: {
    inference_time_ms: number;  // 推理耗时
    audio_duration_ms: number;  // 音频时长
    rtf: number;                // 实时率
  };

  metadata: {
    completed_at: string;
    worker_id: string;
  };
}
```

### 5.2 Agent 会话消息格式

```typescript
// Agent 输入消息
interface AgentInputMessage {
  session_id: string;
  message_id: string;
  user_id: string;

  input: {
    type: "audio" | "text";

    // 音频输入
    audio?: {
      data: string;          // Base64
      format: string;
    };

    // 文本输入
    text?: string;
  };

  context: {
    conversation_history: Array<{
      role: "user" | "assistant";
      content: string;
    }>;
    user_vocabulary: string[];  // 用户词汇表
  };

  options: {
    enable_tts: boolean;       // 是否返回语音
    tts_voice: string;         // TTS 声音
    max_response_length: number;
  };
}

// Agent 输出消息
interface AgentOutputMessage {
  session_id: string;
  message_id: string;

  output: {
    text: string;              // 文本回复

    // 语音输出（可选）
    audio?: {
      data: string;            // Base64 音频
      format: string;
      duration_ms: number;
    };

    // 工具调用结果（可选）
    tool_calls?: Array<{
      tool_name: string;
      arguments: Record<string, any>;
      result: any;
    }>;
  };

  // 意图识别
  intent: {
    name: string;              // 意图名称
    confidence: number;
    slots: Record<string, string>;
  };

  metadata: {
    model_used: string;
    response_time_ms: number;
  };
}
```

### 5.3 HTTP API 接口

```yaml
# biz-service API

# 认证
POST /api/v1/auth/login
POST /api/v1/auth/register
POST /api/v1/auth/refresh

# 用户
GET  /api/v1/users/me
PATCH /api/v1/users/me
GET  /api/v1/users/me/settings
PATCH /api/v1/users/me/settings

# 会话
GET  /api/v1/sessions
POST /api/v1/sessions
GET  /api/v1/sessions/{session_id}
DELETE /api/v1/sessions/{session_id}

# 转录
POST /api/v1/transcribe/batch          # 批量转录
GET  /api/v1/transcribe/tasks/{task_id}  # 查询任务状态

# 词汇表
GET  /api/v1/vocabulary
POST /api/v1/vocabulary
DELETE /api/v1/vocabulary/{word_id}

# 训练
POST /api/v1/training/tasks            # 创建训练任务
GET  /api/v1/training/tasks/{task_id}  # 查询训练状态
GET  /api/v1/training/models           # 获取用户模型列表

# WebSocket 端点
WS   /ws/transcribe                    # 实时转录
WS   /ws/agent                         # 语音 Agent
```

---

## 6. 部署架构

### 6.1 开发环境

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
    environment:
      - API_URL=http://localhost:8000

  biz-service:
    build: ./services/biz-service
    ports:
      - "8000:8000"
    volumes:
      - ./services/biz-service:/app
    depends_on:
      - postgres
      - redis
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/db
      - REDIS_URL=redis://redis:6379

  ai-asr:
    build: ./services/ai-asr
    ports:
      - "8001:8001"
    volumes:
      - ./services/ai-asr:/app
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  ai-agent:
    build: ./services/ai-agent
    ports:
      - "8002:8002"
    volumes:
      - ./services/ai-agent:/app
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=db
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  minio_data:
```

### 6.2 生产环境 (Kubernetes)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Kubernetes Cluster                           │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         Ingress (Traefik)                         │  │
│  └────────────────────────────────┬─────────────────────────────────┘  │
│                                   │                                     │
│     ┌─────────────────────────────┼─────────────────────────────┐      │
│     │                             │                             │      │
│     ↓                             ↓                             ↓      │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐  │
│  │  frontend   │           │ biz-service │           │   ai-asr    │  │
│  │  Deployment │           │  Deployment │           │  Deployment │  │
│  │  (3 pods)   │           │  (3 pods)   │           │  (2 pods)   │  │
│  └─────────────┘           └─────────────┘           └─────────────┘  │
│                                   │                         │          │
│                                   ↓                         ↓          │
│                         ┌─────────────────────────────────────┐       │
│                         │           Stateful Services          │       │
│                         │  ┌─────────┐ ┌─────┐ ┌─────────┐   │       │
│                         │  │Postgres │ │Redis│ │  MinIO  │   │       │
│                         │  └─────────┘ └─────┘ └─────────┘   │       │
│                         └─────────────────────────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 各服务责任边界总结

| 服务 | 负责 | 不负责 |
|------|------|--------|
| **Frontend** | UI渲染、音频采集、本地VAD、用户交互 | 业务逻辑、数据持久化、AI推理 |
| **biz-service** | 用户管理、会话管理、数据CRUD、文件存储 | AI推理、任务调度、Worker管理 |
| **Agent SDK** | 任务调度、Redis Stream、进程池、HTTP封装、健康检查 | 业务逻辑、AI推理、用户认证 |
| **ai-asr** | ASR推理、VAD、GER纠错、流式处理 | 用户认证、数据库、队列管理 |
| **ai-agent** | LLM对话、TTS合成、意图识别、工具调用 | 用户认证、数据库、队列管理 |

---

## 8. 你的职责范围 (AI/Agent 部分)

根据你说的职责主要是 AI/Agent 部分，你需要专注的代码目录：

```
dysarthria-voice-assistant/
├── services/
│   ├── ai-asr/              ← 你的主要职责
│   │   ├── engines/         # ASR 引擎实现
│   │   ├── processors/      # VAD、GER 处理器
│   │   └── handlers/        # 请求处理
│   │
│   └── ai-agent/            ← 你的主要职责
│       ├── pipelines/       # 对话 Pipeline
│       ├── llm/             # LLM 适配器
│       ├── tts/             # TTS 适配器
│       ├── tools/           # 工具实现
│       └── prompts/         # Prompt 模板
│
└── agent-sdk/               ← 共同职责（与后端协作）
    └── models/              # 数据模型定义
```

**你不需要关心的部分**:
- 前端 UI 开发
- 用户认证/授权
- 数据库设计和 CRUD
- 文件存储
- WebSocket 连接管理
- 部署和运维

**你需要定义的接口**:
- ASRTask / ASRResult 数据格式
- AgentInput / AgentOutput 数据格式
- Worker 健康检查接口
- 模型加载/卸载接口
