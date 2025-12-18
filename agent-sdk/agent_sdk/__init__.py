"""
Agent SDK - 主入口包

Agent SDK 提供统一的接口来调用 AI 服务，包括：
- ASR 语音识别
- Agent 对话
- 流式处理

主要组件：
- AgentSDK: 主客户端类
- StreamManager: Redis Stream 管理
- TaskDispatcher: 任务调度
- WorkerPool: Worker 池管理
- AsyncHTTPClient: HTTP 客户端
"""

from .client import AgentSDK, StreamingSession, AgentSession
from .config import (
    SDKConfig,
    RedisConfig,
    WorkerPoolConfig,
    HTTPClientConfig,
    MetricsConfig
)
from .stream.stream_manager import StreamManager
from .dispatcher.task_dispatcher import TaskDispatcher
from .pool.worker_manager import WorkerPool, WorkerHeartbeatManager, LoadBalanceStrategy
from .http.async_client import AsyncHTTPClient, HTTPResponse, ExternalAPIClient
from .worker_base import BaseWorker, ASRWorkerBase, AgentWorkerBase

__version__ = "0.1.0"

__all__ = [
    # 主客户端
    "AgentSDK",
    "StreamingSession",
    "AgentSession",

    # 配置
    "SDKConfig",
    "RedisConfig",
    "WorkerPoolConfig",
    "HTTPClientConfig",
    "MetricsConfig",

    # Stream 管理
    "StreamManager",

    # 任务调度
    "TaskDispatcher",

    # Worker 池
    "WorkerPool",
    "WorkerHeartbeatManager",
    "LoadBalanceStrategy",

    # HTTP 客户端
    "AsyncHTTPClient",
    "HTTPResponse",
    "ExternalAPIClient",

    # Worker 基类
    "BaseWorker",
    "ASRWorkerBase",
    "AgentWorkerBase",
]
