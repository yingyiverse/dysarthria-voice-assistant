"""
共享数据模型 - 通用类型
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum
import uuid


def generate_id(prefix: str = "") -> str:
    """生成唯一 ID"""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{uid}" if prefix else uid


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10


class BaseTask(BaseModel):
    """任务基类

    所有任务类型都继承自此基类
    """
    task_id: str = Field(default_factory=lambda: generate_id("task"))
    task_type: str = Field(..., description="任务类型")
    user_id: str = Field(..., description="用户 ID")
    session_id: Optional[str] = Field(None, description="会话 ID")
    priority: TaskPriority = Field(TaskPriority.NORMAL, description="任务优先级")
    timeout_ms: int = Field(30000, description="超时时间（毫秒）")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict, description="扩展元数据")

    class Config:
        use_enum_values = True


class BaseResult(BaseModel):
    """结果基类

    所有结果类型都继承自此基类
    """
    task_id: str
    status: TaskStatus
    error: Optional[str] = None
    error_code: Optional[int] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None
    worker_id: Optional[str] = None

    class Config:
        use_enum_values = True


class WorkerStatus(str, Enum):
    """Worker 状态"""
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"  # 正在排空，不接受新任务
    STOPPED = "stopped"


class WorkerInfo(BaseModel):
    """Worker 信息"""
    worker_id: str
    worker_type: str  # asr, agent, training
    status: WorkerStatus = WorkerStatus.STARTING

    # 能力
    capabilities: list[str] = Field(default_factory=list)
    max_concurrent: int = Field(4, description="最大并发数")

    # 负载
    current_load: int = Field(0, description="当前负载")

    # GPU 信息
    gpu_name: Optional[str] = None
    gpu_memory_total_mb: Optional[int] = None
    gpu_memory_used_mb: Optional[int] = None

    # 健康检查
    health_endpoint: Optional[str] = None
    last_health_check: Optional[datetime] = None

    # 统计
    total_tasks: int = Field(0)
    failed_tasks: int = Field(0)
    uptime_seconds: int = Field(0)

    class Config:
        use_enum_values = True


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str  # healthy, unhealthy, degraded
    worker_id: str
    uptime_seconds: int
    current_load: int
    max_load: int
    gpu_memory_used_mb: Optional[int] = None
    gpu_memory_total_mb: Optional[int] = None
    last_task_at: Optional[datetime] = None
    error_rate_1m: float = Field(0.0, description="最近1分钟错误率")
    details: dict = Field(default_factory=dict)
