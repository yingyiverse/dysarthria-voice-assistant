# 接口规范文档

## 1. 概述

本文档定义了系统各服务之间的接口规范，包括：
- HTTP REST API
- WebSocket API
- Redis Stream 消息格式
- 内部 gRPC 接口（可选）

---

## 2. 数据模型定义

### 2.1 通用模型

```python
# shared/models/common.py

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(int, Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10

class BaseTask(BaseModel):
    """任务基类"""
    task_id: str = Field(..., description="任务唯一标识")
    task_type: str = Field(..., description="任务类型")
    user_id: str = Field(..., description="用户 ID")
    session_id: Optional[str] = Field(None, description="会话 ID")
    priority: TaskPriority = Field(TaskPriority.NORMAL, description="任务优先级")
    timeout_ms: int = Field(30000, description="超时时间（毫秒）")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)

class BaseResult(BaseModel):
    """结果基类"""
    task_id: str
    status: TaskStatus
    error: Optional[str] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None
    worker_id: Optional[str] = None
```

### 2.2 ASR 相关模型

```python
# shared/models/asr.py

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ASREngine(str, Enum):
    SENSEVOICE = "sensevoice"
    WHISPER = "whisper"
    SENSEVOICE_LORA = "sensevoice_lora"  # 个性化模型

class AudioFormat(str, Enum):
    WAV = "wav"
    PCM = "pcm"
    MP3 = "mp3"
    WEBM = "webm"

class AudioData(BaseModel):
    """音频数据"""
    data: str = Field(..., description="Base64 编码的音频数据")
    format: AudioFormat = Field(AudioFormat.WAV)
    sample_rate: int = Field(16000, description="采样率")
    channels: int = Field(1, description="通道数")
    duration_ms: Optional[int] = Field(None, description="音频时长（毫秒）")

class ASROptions(BaseModel):
    """ASR 选项"""
    engine: ASREngine = Field(ASREngine.SENSEVOICE)
    language: str = Field("zh", description="语言代码")
    enable_ger: bool = Field(True, description="是否启用 GER 纠错")
    enable_punctuation: bool = Field(True, description="是否添加标点")
    enable_word_timestamps: bool = Field(False, description="是否返回词级时间戳")

    # 个性化选项
    use_personal_model: bool = Field(False, description="是否使用个人模型")
    personal_model_id: Optional[str] = Field(None, description="个人模型 ID")
    user_vocabulary: List[str] = Field(default_factory=list, description="用户词汇表")

    # GER 选项
    ger_confidence_threshold: float = Field(0.85, description="触发 GER 的置信度阈值")

class WordInfo(BaseModel):
    """词级信息"""
    word: str
    start_time: float = Field(..., description="开始时间（秒）")
    end_time: float = Field(..., description="结束时间（秒）")
    confidence: float = Field(..., ge=0, le=1)

class ASRTask(BaseTask):
    """ASR 任务"""
    task_type: str = "asr"
    audio: AudioData
    options: ASROptions = Field(default_factory=ASROptions)

class ASRStreamingTask(BaseTask):
    """流式 ASR 任务"""
    task_type: str = "asr_streaming"
    options: ASROptions = Field(default_factory=ASROptions)
    # 流式任务不包含 audio，音频通过 Stream 发送

class ASRResult(BaseResult):
    """ASR 结果"""
    text: str = Field("", description="识别文本")
    original_text: Optional[str] = Field(None, description="GER 纠错前的原始文本")
    confidence: float = Field(0.0, ge=0, le=1, description="整体置信度")
    is_final: bool = Field(True, description="是否最终结果")

    words: Optional[List[WordInfo]] = Field(None, description="词级信息")

    # 性能指标
    audio_duration_ms: Optional[int] = None
    inference_time_ms: Optional[int] = None
    rtf: Optional[float] = Field(None, description="实时率 (inference_time / audio_duration)")

class ASRStreamingChunk(BaseModel):
    """流式 ASR 结果块"""
    session_id: str
    chunk_id: int
    text: str
    is_partial: bool = Field(True, description="是否部分结果")
    is_final: bool = Field(False, description="是否最终结果")
    confidence: float = Field(0.0)
    timestamp_ms: int = Field(..., description="时间戳（相对于会话开始）")
```

### 2.3 Agent 相关模型

```python
# shared/models/agent.py

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union
from enum import Enum
from datetime import datetime

class AgentType(str, Enum):
    VOICE_ASSISTANT = "voice_assistant"  # 语音助手
    TASK_EXECUTOR = "task_executor"      # 任务执行
    REHABILITATION = "rehabilitation"    # 康复训练

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationMessage(BaseModel):
    """对话消息"""
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None
    audio_url: Optional[str] = None  # 关联的音频

class AgentInputType(str, Enum):
    AUDIO = "audio"
    TEXT = "text"

class AgentInput(BaseModel):
    """Agent 输入"""
    type: AgentInputType

    # 音频输入
    audio: Optional[AudioData] = None

    # 文本输入（可能来自 ASR 或直接输入）
    text: Optional[str] = None
    asr_confidence: Optional[float] = None

class AgentContext(BaseModel):
    """Agent 上下文"""
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list,
        description="对话历史"
    )
    user_vocabulary: List[str] = Field(
        default_factory=list,
        description="用户常用词汇"
    )
    user_profile: Optional[dict] = Field(
        None,
        description="用户画像（偏好等）"
    )

class AgentOptions(BaseModel):
    """Agent 选项"""
    enable_tts: bool = Field(True, description="是否返回语音")
    tts_voice: str = Field("zh_female_calm", description="TTS 声音")
    tts_speed: float = Field(1.0, description="TTS 语速")

    max_response_tokens: int = Field(500, description="最大回复 token 数")
    temperature: float = Field(0.7, description="LLM temperature")

    enable_tools: bool = Field(True, description="是否启用工具调用")
    allowed_tools: Optional[List[str]] = Field(None, description="允许的工具列表")

class AgentTask(BaseTask):
    """Agent 任务"""
    task_type: str = "agent"
    agent_type: AgentType = Field(AgentType.VOICE_ASSISTANT)
    input: AgentInput
    context: AgentContext = Field(default_factory=AgentContext)
    options: AgentOptions = Field(default_factory=AgentOptions)

class ToolCall(BaseModel):
    """工具调用"""
    tool_name: str
    arguments: dict
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None

class IntentInfo(BaseModel):
    """意图识别结果"""
    intent_name: str
    confidence: float
    slots: dict = Field(default_factory=dict)

class AgentOutput(BaseModel):
    """Agent 输出"""
    text: str = Field(..., description="文本回复")

    # 语音输出
    audio: Optional[AudioData] = None

    # 意图识别
    intent: Optional[IntentInfo] = None

    # 工具调用
    tool_calls: List[ToolCall] = Field(default_factory=list)

    # 建议的后续问题
    suggestions: List[str] = Field(default_factory=list)

class AgentResult(BaseResult):
    """Agent 结果"""
    output: AgentOutput

    # 性能指标
    asr_time_ms: Optional[int] = None
    llm_time_ms: Optional[int] = None
    tts_time_ms: Optional[int] = None
    total_time_ms: Optional[int] = None

class AgentStreamingChunk(BaseModel):
    """流式 Agent 输出块"""
    session_id: str
    chunk_type: str = Field(..., description="text | audio | tool_call | done")

    # 文本块
    text_delta: Optional[str] = None

    # 音频块
    audio_chunk: Optional[str] = None  # Base64

    # 工具调用
    tool_call: Optional[ToolCall] = None

    # 结束标记
    is_done: bool = False
    final_result: Optional[AgentResult] = None
```

### 2.4 训练相关模型

```python
# shared/models/training.py

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime

class TrainingStatus(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"

class TrainingSample(BaseModel):
    """训练样本"""
    audio_url: str
    text: str
    duration_ms: int
    quality_score: Optional[float] = None

class TrainingConfig(BaseModel):
    """训练配置"""
    base_model: str = Field("sensevoice", description="基础模型")
    lora_rank: int = Field(16)
    lora_alpha: int = Field(32)
    learning_rate: float = Field(5e-5)
    epochs: int = Field(10)
    batch_size: int = Field(4)

class TrainingTask(BaseModel):
    """训练任务"""
    task_id: str
    user_id: str
    status: TrainingStatus = TrainingStatus.PENDING

    samples: List[TrainingSample] = Field(default_factory=list)
    config: TrainingConfig = Field(default_factory=TrainingConfig)

    # 进度
    progress: float = Field(0.0, ge=0, le=1)
    current_epoch: int = Field(0)
    current_step: int = Field(0)

    # 结果
    model_id: Optional[str] = None
    metrics: Optional[dict] = None  # 训练指标

    # 时间
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    error: Optional[str] = None
```

---

## 3. HTTP REST API

### 3.1 业务服务 API (biz-service)

#### 3.1.1 认证接口

```yaml
# POST /api/v1/auth/register
# 用户注册
Request:
  {
    "email": "user@example.com",
    "password": "password123",
    "name": "张三"
  }
Response:
  {
    "user_id": "usr_xxx",
    "email": "user@example.com",
    "name": "张三",
    "access_token": "eyJhbGci...",
    "refresh_token": "eyJhbGci...",
    "expires_in": 3600
  }

# POST /api/v1/auth/login
# 用户登录
Request:
  {
    "email": "user@example.com",
    "password": "password123"
  }
Response:
  {
    "access_token": "eyJhbGci...",
    "refresh_token": "eyJhbGci...",
    "expires_in": 3600,
    "user": {
      "user_id": "usr_xxx",
      "email": "user@example.com",
      "name": "张三"
    }
  }

# POST /api/v1/auth/refresh
# 刷新 Token
Request:
  {
    "refresh_token": "eyJhbGci..."
  }
Response:
  {
    "access_token": "eyJhbGci...",
    "expires_in": 3600
  }
```

#### 3.1.2 用户接口

```yaml
# GET /api/v1/users/me
# 获取当前用户信息
Response:
  {
    "user_id": "usr_xxx",
    "email": "user@example.com",
    "name": "张三",
    "avatar_url": null,
    "created_at": "2024-01-01T00:00:00Z",
    "settings": {
      "default_engine": "sensevoice",
      "enable_ger": true,
      "tts_voice": "zh_female_calm"
    }
  }

# PATCH /api/v1/users/me/settings
# 更新用户设置
Request:
  {
    "default_engine": "whisper",
    "enable_ger": false
  }
Response:
  {
    "message": "Settings updated successfully"
  }
```

#### 3.1.3 会话接口

```yaml
# GET /api/v1/sessions
# 获取会话列表
Query Parameters:
  - page: int = 1
  - page_size: int = 20
  - sort_by: str = "created_at"
  - order: str = "desc"
Response:
  {
    "items": [
      {
        "session_id": "ses_xxx",
        "title": "2024年1月1日的对话",
        "created_at": "2024-01-01T10:00:00Z",
        "updated_at": "2024-01-01T10:30:00Z",
        "transcript_count": 15,
        "total_duration_ms": 180000
      }
    ],
    "total": 100,
    "page": 1,
    "page_size": 20
  }

# POST /api/v1/sessions
# 创建新会话
Request:
  {
    "title": "新对话"  // 可选，默认自动生成
  }
Response:
  {
    "session_id": "ses_xxx",
    "title": "新对话",
    "created_at": "2024-01-01T10:00:00Z"
  }

# GET /api/v1/sessions/{session_id}
# 获取会话详情
Response:
  {
    "session_id": "ses_xxx",
    "title": "2024年1月1日的对话",
    "created_at": "2024-01-01T10:00:00Z",
    "transcripts": [
      {
        "transcript_id": "tr_xxx",
        "text": "今天天气很好",
        "original_text": "今天天器很好",
        "confidence": 0.92,
        "start_time_ms": 0,
        "end_time_ms": 2000,
        "created_at": "2024-01-01T10:00:00Z"
      }
    ]
  }

# DELETE /api/v1/sessions/{session_id}
# 删除会话
Response:
  {
    "message": "Session deleted successfully"
  }
```

#### 3.1.4 转录接口

```yaml
# POST /api/v1/transcribe/batch
# 批量转录（上传音频文件）
Request (multipart/form-data):
  - file: audio file
  - options: JSON string (ASROptions)
Response:
  {
    "task_id": "task_xxx",
    "status": "pending",
    "estimated_time_ms": 5000
  }

# GET /api/v1/transcribe/tasks/{task_id}
# 查询转录任务状态
Response:
  {
    "task_id": "task_xxx",
    "status": "completed",  // pending | processing | completed | failed
    "result": {
      "text": "识别的文本内容",
      "confidence": 0.95,
      "words": [...]
    },
    "created_at": "2024-01-01T10:00:00Z",
    "completed_at": "2024-01-01T10:00:05Z"
  }
```

#### 3.1.5 词汇表接口

```yaml
# GET /api/v1/vocabulary
# 获取用户词汇表
Response:
  {
    "items": [
      {
        "word_id": "voc_xxx",
        "word": "构音障碍",
        "frequency": 10,
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "total": 50
  }

# POST /api/v1/vocabulary
# 添加词汇
Request:
  {
    "words": ["构音障碍", "语言康复"]
  }
Response:
  {
    "added_count": 2,
    "items": [...]
  }

# DELETE /api/v1/vocabulary/{word_id}
# 删除词汇
Response:
  {
    "message": "Word deleted successfully"
  }

# POST /api/v1/vocabulary/import
# 批量导入词汇
Request:
  {
    "source": "file",  // file | text
    "data": "词汇1\n词汇2\n词汇3"
  }
Response:
  {
    "imported_count": 3,
    "skipped_count": 0
  }
```

#### 3.1.6 训练接口

```yaml
# POST /api/v1/training/tasks
# 创建训练任务
Request:
  {
    "config": {
      "base_model": "sensevoice",
      "epochs": 10
    }
  }
Response:
  {
    "task_id": "train_xxx",
    "status": "pending",
    "upload_url": "https://storage.example.com/upload/xxx"  // 用于上传训练数据
  }

# POST /api/v1/training/tasks/{task_id}/samples
# 上传训练样本
Request (multipart/form-data):
  - file: audio file
  - text: 对应的文本
Response:
  {
    "sample_id": "sample_xxx",
    "uploaded_count": 1,
    "total_count": 10
  }

# POST /api/v1/training/tasks/{task_id}/start
# 开始训练
Response:
  {
    "task_id": "train_xxx",
    "status": "training",
    "estimated_time_minutes": 30
  }

# GET /api/v1/training/tasks/{task_id}
# 查询训练状态
Response:
  {
    "task_id": "train_xxx",
    "status": "training",
    "progress": 0.45,
    "current_epoch": 5,
    "total_epochs": 10,
    "metrics": {
      "train_loss": 0.234,
      "val_loss": 0.256,
      "wer": 0.15
    }
  }

# GET /api/v1/training/models
# 获取用户的个人模型列表
Response:
  {
    "items": [
      {
        "model_id": "model_xxx",
        "name": "我的个人模型",
        "created_at": "2024-01-01T00:00:00Z",
        "metrics": {
          "wer": 0.15,
          "sample_count": 100
        },
        "is_active": true
      }
    ]
  }
```

---

## 4. WebSocket API

### 4.1 实时转录 WebSocket

```yaml
# 端点: /ws/transcribe
# 协议: WebSocket

# 1. 连接建立
# 客户端发送连接请求，Header 中包含 Authorization

# 2. 开始会话
# 客户端 → 服务端
{
  "type": "start_session",
  "session_id": "ses_xxx",  // 可选，不传则创建新会话
  "options": {
    "engine": "sensevoice",
    "enable_ger": true,
    "language": "zh"
  }
}

# 服务端 → 客户端
{
  "type": "session_started",
  "session_id": "ses_xxx"
}

# 3. 发送音频数据
# 客户端 → 服务端 (Binary WebSocket Frame)
# 直接发送 PCM 音频数据（16kHz, 16bit, mono）

# 4. 接收转录结果
# 服务端 → 客户端
{
  "type": "transcript",
  "data": {
    "text": "今天天气",
    "is_partial": true,
    "confidence": 0.85,
    "timestamp_ms": 1500
  }
}

# 5. 接收最终结果
{
  "type": "transcript",
  "data": {
    "text": "今天天气很好",
    "is_partial": false,
    "is_final": true,
    "original_text": "今天天器很好",  // GER 纠错前
    "confidence": 0.92,
    "timestamp_ms": 3000
  }
}

# 6. 结束会话
# 客户端 → 服务端
{
  "type": "end_session"
}

# 服务端 → 客户端
{
  "type": "session_ended",
  "session_id": "ses_xxx",
  "summary": {
    "total_duration_ms": 30000,
    "transcript_count": 10
  }
}

# 7. 错误处理
{
  "type": "error",
  "error": {
    "code": "ASR_ERROR",
    "message": "Failed to process audio"
  }
}
```

### 4.2 语音 Agent WebSocket

```yaml
# 端点: /ws/agent
# 协议: WebSocket

# 1. 开始对话会话
# 客户端 → 服务端
{
  "type": "start_session",
  "agent_type": "voice_assistant",
  "options": {
    "enable_tts": true,
    "tts_voice": "zh_female_calm"
  }
}

# 服务端 → 客户端
{
  "type": "session_started",
  "session_id": "agent_ses_xxx",
  "greeting": {
    "text": "您好！有什么可以帮您的吗？",
    "audio": "base64..."  // TTS 音频
  }
}

# 2. 发送语音输入
# 客户端 → 服务端 (Binary Frame: 音频数据)

# 或发送文本输入
{
  "type": "user_input",
  "input_type": "text",
  "text": "今天北京天气怎么样"
}

# 3. 接收 Agent 响应（流式）
# 服务端 → 客户端

# ASR 结果
{
  "type": "asr_result",
  "text": "今天北京天气怎么样",
  "confidence": 0.9
}

# 思考中
{
  "type": "thinking",
  "message": "正在查询天气..."
}

# 文本响应（流式）
{
  "type": "response_text",
  "delta": "今天北京",
  "is_final": false
}

{
  "type": "response_text",
  "delta": "天气晴朗",
  "is_final": false
}

{
  "type": "response_text",
  "delta": "，最高温度8度。",
  "is_final": true,
  "full_text": "今天北京天气晴朗，最高温度8度。"
}

# 语音响应
{
  "type": "response_audio",
  "audio": "base64...",  // 完整音频或分块
  "is_chunk": false
}

# 工具调用
{
  "type": "tool_call",
  "tool_name": "get_weather",
  "arguments": {"city": "北京"},
  "result": {"weather": "晴", "temp_high": 8, "temp_low": -2}
}

# 4. 中断 Agent
# 客户端 → 服务端
{
  "type": "interrupt"
}

# 服务端 → 客户端
{
  "type": "interrupted",
  "message": "好的，请说"
}

# 5. 结束会话
# 客户端 → 服务端
{
  "type": "end_session"
}
```

---

## 5. Redis Stream 消息格式

### 5.1 Stream 命名规范

```
# 任务队列
tasks:asr              # ASR 任务
tasks:asr_streaming    # 流式 ASR 任务
tasks:agent            # Agent 任务
tasks:training         # 训练任务

# 结果队列
results:{task_id}      # 任务结果（短期存储）

# 流式会话
stream:{session_id}:audio    # 音频流（输入）
stream:{session_id}:result   # 结果流（输出）
```

### 5.2 消息格式

```python
# ASR 任务消息
{
    "task_id": "task_xxx",
    "task_type": "asr",
    "user_id": "usr_xxx",
    "audio_data": "base64...",  # 或存储 URL
    "audio_format": "pcm",
    "sample_rate": "16000",
    "options": "{\"engine\": \"sensevoice\", ...}",  # JSON 字符串
    "created_at": "1704067200000",  # Unix timestamp ms
    "priority": "5",
    "timeout_ms": "30000"
}

# ASR 结果消息
{
    "task_id": "task_xxx",
    "status": "completed",
    "text": "识别结果",
    "original_text": "原始结果",
    "confidence": "0.92",
    "is_final": "true",
    "inference_time_ms": "150",
    "worker_id": "worker_asr_1",
    "completed_at": "1704067205000"
}

# Agent 任务消息
{
    "task_id": "task_xxx",
    "task_type": "agent",
    "user_id": "usr_xxx",
    "session_id": "agent_ses_xxx",
    "input_type": "audio",  # audio | text
    "input_data": "base64...",  # 音频或文本
    "context": "{...}",  # JSON: 对话历史、用户词汇等
    "options": "{...}",  # JSON: TTS、工具等选项
    "created_at": "1704067200000"
}
```

---

## 6. Worker 接口规范

### 6.1 Worker 注册

```python
# Worker 启动时向 Agent SDK 注册

# POST /internal/workers/register
Request:
{
    "worker_id": "worker_asr_1",
    "worker_type": "asr",
    "capabilities": ["sensevoice", "whisper", "ger"],
    "max_concurrent": 4,
    "gpu_info": {
        "name": "NVIDIA T4",
        "memory_mb": 16384
    },
    "health_endpoint": "http://worker-asr:8001/health"
}

Response:
{
    "registered": true,
    "assigned_streams": ["tasks:asr"]
}
```

### 6.2 健康检查接口

```python
# 每个 Worker 必须实现

# GET /health
Response:
{
    "status": "healthy",  # healthy | unhealthy | degraded
    "worker_id": "worker_asr_1",
    "uptime_seconds": 3600,
    "current_load": 2,
    "max_load": 4,
    "gpu_memory_used_mb": 4096,
    "gpu_memory_total_mb": 16384,
    "last_task_at": "2024-01-01T10:00:00Z",
    "error_rate_1m": 0.01
}
```

### 6.3 ASR Worker 内部接口

```python
# POST /internal/transcribe
# 单次转录（由 Agent SDK 调用）
Request:
{
    "task_id": "task_xxx",
    "audio": {
        "data": "base64...",
        "format": "pcm",
        "sample_rate": 16000
    },
    "options": {
        "engine": "sensevoice",
        "enable_ger": true,
        "user_vocabulary": ["构音障碍", "语言康复"]
    }
}

Response:
{
    "task_id": "task_xxx",
    "text": "识别结果",
    "original_text": "原始结果",
    "confidence": 0.92,
    "words": [...],
    "inference_time_ms": 150
}
```

### 6.4 Agent Worker 内部接口

```python
# POST /internal/chat
# 单轮对话（由 Agent SDK 调用）
Request:
{
    "task_id": "task_xxx",
    "session_id": "agent_ses_xxx",
    "input": {
        "type": "text",
        "text": "今天天气怎么样"
    },
    "context": {
        "conversation_history": [...],
        "user_vocabulary": [...]
    },
    "options": {
        "enable_tts": true,
        "tts_voice": "zh_female_calm",
        "enable_tools": true
    }
}

Response:
{
    "task_id": "task_xxx",
    "output": {
        "text": "今天北京天气晴朗，最高温度8度。",
        "audio": "base64...",
        "intent": {
            "name": "query_weather",
            "confidence": 0.95,
            "slots": {"city": "北京"}
        },
        "tool_calls": [
            {
                "tool_name": "get_weather",
                "arguments": {"city": "北京"},
                "result": {"weather": "晴", "temp": 8}
            }
        ]
    },
    "timing": {
        "llm_time_ms": 500,
        "tts_time_ms": 200,
        "total_time_ms": 700
    }
}
```

---

## 7. 错误码规范

```python
# 通用错误码
ERROR_CODES = {
    # 认证错误 (1xxx)
    1001: "INVALID_TOKEN",
    1002: "TOKEN_EXPIRED",
    1003: "UNAUTHORIZED",

    # 请求错误 (2xxx)
    2001: "INVALID_REQUEST",
    2002: "MISSING_PARAMETER",
    2003: "INVALID_PARAMETER",
    2004: "RESOURCE_NOT_FOUND",

    # 业务错误 (3xxx)
    3001: "SESSION_NOT_FOUND",
    3002: "VOCABULARY_LIMIT_EXCEEDED",
    3003: "TRAINING_IN_PROGRESS",
    3004: "MODEL_NOT_FOUND",

    # AI 服务错误 (4xxx)
    4001: "ASR_FAILED",
    4002: "ASR_TIMEOUT",
    4003: "GER_FAILED",
    4004: "AGENT_FAILED",
    4005: "TTS_FAILED",
    4006: "MODEL_LOADING",
    4007: "GPU_OOM",

    # 系统错误 (5xxx)
    5001: "INTERNAL_ERROR",
    5002: "SERVICE_UNAVAILABLE",
    5003: "RATE_LIMITED",
    5004: "STORAGE_ERROR",
}

# 错误响应格式
{
    "error": {
        "code": 4001,
        "name": "ASR_FAILED",
        "message": "Failed to transcribe audio",
        "details": {
            "reason": "Invalid audio format"
        }
    }
}
```
