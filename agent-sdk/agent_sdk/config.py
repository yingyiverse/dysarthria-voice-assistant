"""
Agent SDK - 配置管理
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


class RedisConfig(BaseModel):
    """Redis 配置"""
    url: str = Field("redis://localhost:6379", description="Redis URL")
    db: int = Field(0, description="数据库编号")
    password: Optional[str] = None

    # Stream 配置
    consumer_group: str = Field("agent-sdk-group", description="消费者组名称")
    max_stream_length: int = Field(10000, description="Stream 最大长度")
    block_timeout_ms: int = Field(5000, description="阻塞等待超时")

    # 连接池
    max_connections: int = Field(20, description="最大连接数")


class WorkerPoolConfig(BaseModel):
    """Worker 池配置"""
    health_check_interval: int = Field(30, description="健康检查间隔（秒）")
    health_check_timeout: int = Field(5, description="健康检查超时（秒）")
    auto_restart: bool = Field(True, description="是否自动重启不健康的 Worker")
    max_restart_attempts: int = Field(3, description="最大重启尝试次数")

    # 负载均衡
    load_balance_strategy: str = Field("least_loaded", description="负载均衡策略")


class HTTPClientConfig(BaseModel):
    """HTTP 客户端配置"""
    timeout: int = Field(30, description="请求超时（秒）")
    max_retries: int = Field(3, description="最大重试次数")
    retry_delay: float = Field(1.0, description="重试延迟（秒）")
    retry_backoff: float = Field(2.0, description="重试延迟指数退避")

    # 连接池
    max_connections: int = Field(100, description="最大连接数")
    keepalive_timeout: int = Field(30, description="Keep-alive 超时")


class MetricsConfig(BaseModel):
    """监控指标配置"""
    enabled: bool = Field(True, description="是否启用指标收集")
    export_interval: int = Field(10, description="指标导出间隔（秒）")
    prometheus_port: Optional[int] = Field(9090, description="Prometheus 端口")


class SDKConfig(BaseModel):
    """Agent SDK 总配置"""
    service_name: str = Field("agent-sdk", description="服务名称")

    redis: RedisConfig = Field(default_factory=RedisConfig)
    workers: WorkerPoolConfig = Field(default_factory=WorkerPoolConfig)
    http: HTTPClientConfig = Field(default_factory=HTTPClientConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    # 任务配置
    default_timeout_ms: int = Field(30000, description="默认任务超时")
    max_queue_size: int = Field(1000, description="最大队列大小")

    # Stream 名称配置
    stream_prefix: str = Field("dva", description="Stream 名称前缀")

    @property
    def asr_task_stream(self) -> str:
        return f"{self.stream_prefix}:tasks:asr"

    @property
    def asr_streaming_task_stream(self) -> str:
        return f"{self.stream_prefix}:tasks:asr_streaming"

    @property
    def agent_task_stream(self) -> str:
        return f"{self.stream_prefix}:tasks:agent"

    @property
    def training_task_stream(self) -> str:
        return f"{self.stream_prefix}:tasks:training"

    @classmethod
    def from_env(cls) -> "SDKConfig":
        """从环境变量加载配置"""
        import os

        return cls(
            service_name=os.getenv("SDK_SERVICE_NAME", "agent-sdk"),
            redis=RedisConfig(
                url=os.getenv("REDIS_URL", "redis://localhost:6379"),
                password=os.getenv("REDIS_PASSWORD"),
            ),
        )
