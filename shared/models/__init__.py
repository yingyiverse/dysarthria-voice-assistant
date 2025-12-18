"""
共享数据模型包
"""

from .common import (
    TaskStatus,
    TaskPriority,
    BaseTask,
    BaseResult,
    WorkerStatus,
    WorkerInfo,
    HealthCheckResponse,
    generate_id,
)

from .asr import (
    ASREngine,
    AudioFormat,
    AudioData,
    ASROptions,
    WordInfo,
    ASRTask,
    ASRStreamingTask,
    ASRResult,
    ASRStreamingChunk,
    VADResult,
    GERResult,
)

from .agent import (
    AgentType,
    MessageRole,
    ConversationMessage,
    AgentInputType,
    AgentInput,
    AgentContext,
    TTSVoice,
    AgentOptions,
    AgentTask,
    ToolCall,
    IntentInfo,
    AgentOutput,
    AgentResult,
    AgentStreamingChunkType,
    AgentStreamingChunk,
    ToolParameter,
    ToolDefinition,
    PredefinedIntent,
)

__all__ = [
    # Common
    "TaskStatus",
    "TaskPriority",
    "BaseTask",
    "BaseResult",
    "WorkerStatus",
    "WorkerInfo",
    "HealthCheckResponse",
    "generate_id",

    # ASR
    "ASREngine",
    "AudioFormat",
    "AudioData",
    "ASROptions",
    "WordInfo",
    "ASRTask",
    "ASRStreamingTask",
    "ASRResult",
    "ASRStreamingChunk",
    "VADResult",
    "GERResult",

    # Agent
    "AgentType",
    "MessageRole",
    "ConversationMessage",
    "AgentInputType",
    "AgentInput",
    "AgentContext",
    "TTSVoice",
    "AgentOptions",
    "AgentTask",
    "ToolCall",
    "IntentInfo",
    "AgentOutput",
    "AgentResult",
    "AgentStreamingChunkType",
    "AgentStreamingChunk",
    "ToolParameter",
    "ToolDefinition",
    "PredefinedIntent",
]
