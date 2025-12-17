"""
Agent SDK - 任务调度器
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from ..stream.stream_manager import StreamManager
from ..config import SDKConfig

logger = logging.getLogger(__name__)


class TaskDispatcher:
    """任务调度器

    负责：
    - 将任务路由到正确的 Worker
    - 管理任务生命周期
    - 等待和收集结果
    """

    def __init__(
        self,
        stream_manager: StreamManager,
        worker_pool: "WorkerPool",
        config: SDKConfig
    ):
        self.stream_manager = stream_manager
        self.worker_pool = worker_pool
        self.config = config

        # 任务状态跟踪
        self._pending_tasks: Dict[str, asyncio.Future] = {}

    async def dispatch_asr(self, task: "ASRTask") -> "ASRResult":
        """调度 ASR 任务

        Args:
            task: ASR 任务

        Returns:
            ASR 结果
        """
        from shared.models import ASRResult, TaskStatus

        # 1. 发布任务到 Stream
        stream = self.config.asr_task_stream
        await self.stream_manager.publish(stream, task.model_dump())

        logger.info(f"Dispatched ASR task {task.task_id}")

        # 2. 等待结果
        result_data = await self.stream_manager.get_result(
            f"results:{task.task_id}",
            timeout_ms=task.timeout_ms
        )

        if result_data is None:
            return ASRResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error="Timeout waiting for ASR result",
                text=""
            )

        return ASRResult(**result_data)

    async def dispatch_agent(self, task: "AgentTask") -> "AgentResult":
        """调度 Agent 任务

        Args:
            task: Agent 任务

        Returns:
            Agent 结果
        """
        from shared.models import AgentResult, TaskStatus

        # 1. 发布任务到 Stream
        stream = self.config.agent_task_stream
        await self.stream_manager.publish(stream, task.model_dump())

        logger.info(f"Dispatched Agent task {task.task_id}")

        # 2. 等待结果
        result_data = await self.stream_manager.get_result(
            f"results:{task.task_id}",
            timeout_ms=task.timeout_ms
        )

        if result_data is None:
            return AgentResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error="Timeout waiting for Agent result"
            )

        return AgentResult(**result_data)

    async def start_streaming_session(self, task: "ASRStreamingTask"):
        """启动流式会话

        Args:
            task: 流式 ASR 任务
        """
        stream = self.config.asr_streaming_task_stream
        await self.stream_manager.publish(stream, task.model_dump())

        logger.info(f"Started streaming session {task.session_id}")

    async def dispatch_streaming(
        self,
        task: "ASRStreamingTask",
        result_callback
    ):
        """调度流式任务并处理结果

        Args:
            task: 流式任务
            result_callback: 结果回调
        """
        # 启动会话
        await self.start_streaming_session(task)

        # 订阅结果流
        result_stream = f"stream:{task.session_id}:result"

        async for message in self.stream_manager.subscribe(result_stream):
            await result_callback(message)

            if message.get("is_final"):
                break
