"""
Redis Stream 配置
"""

from pydantic import BaseModel, Field
from typing import Optional


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
