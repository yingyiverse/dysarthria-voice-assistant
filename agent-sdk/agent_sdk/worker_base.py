"""
Agent SDK - Worker 基类

提供给 AI Worker 实现者使用的基类，简化 Worker 开发。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from datetime import datetime

from .stream.stream_manager import StreamManager
from .pool.worker_manager import WorkerHeartbeatManager
from .config import SDKConfig

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from shared.models import WorkerInfo, WorkerStatus, generate_id

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """Worker 基类

    AI Worker 实现者继承此类，只需实现 process() 方法。

    Usage:
        class MyASRWorker(BaseWorker):
            worker_type = "asr"

            async def process(self, task: dict) -> dict:
                # 处理 ASR 任务
                audio_data = task["audio"]
                result = await self.do_asr(audio_data)
                return {"text": result, "status": "success"}

        worker = MyASRWorker(config)
        await worker.start()
    """

    worker_type: str = "base"  # 子类覆盖

    def __init__(
        self,
        config: SDKConfig,
        worker_id: Optional[str] = None,
        max_concurrent: int = 1
    ):
        self.config = config
        self.worker_id = worker_id or generate_id(f"worker-{self.worker_type}")
        self.max_concurrent = max_concurrent

        # 核心组件
        self.stream_manager = StreamManager(config.redis)

        # Worker 信息
        self.worker_info = WorkerInfo(
            worker_id=self.worker_id,
            worker_type=self.worker_type,
            status=WorkerStatus.HEALTHY,
            max_concurrent=max_concurrent,
            current_load=0
        )

        # 心跳管理
        self._heartbeat_manager: Optional[WorkerHeartbeatManager] = None

        # 状态
        self._running = False
        self._current_tasks = 0
        self._semaphore: Optional[asyncio.Semaphore] = None

    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理任务（子类实现）

        Args:
            task: 任务数据

        Returns:
            处理结果
        """
        pass

    async def setup(self):
        """初始化（子类可覆盖）

        在 Worker 启动前调用，用于加载模型等。
        """
        pass

    async def teardown(self):
        """清理（子类可覆盖）

        在 Worker 停止时调用。
        """
        pass

    def get_task_stream(self) -> str:
        """获取任务 Stream 名称

        子类可覆盖以自定义 Stream 名称。
        """
        stream_map = {
            "asr": self.config.asr_task_stream,
            "asr_streaming": self.config.asr_streaming_task_stream,
            "agent": self.config.agent_task_stream,
            "training": self.config.training_task_stream,
        }
        return stream_map.get(self.worker_type, f"{self.config.stream_prefix}:tasks:{self.worker_type}")

    async def start(self):
        """启动 Worker"""
        if self._running:
            return

        logger.info(f"Starting worker {self.worker_id} (type={self.worker_type})")

        # 初始化
        await self.setup()

        # 连接 Redis
        await self.stream_manager.connect()

        # 初始化并发控制
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # 启动心跳
        self._heartbeat_manager = WorkerHeartbeatManager(
            worker_info=self.worker_info,
            heartbeat_interval=10,
            on_heartbeat=self._on_heartbeat
        )
        await self._heartbeat_manager.start()

        # 标记运行状态
        self._running = True

        # 开始消费任务
        await self._consume_tasks()

    async def stop(self):
        """停止 Worker"""
        if not self._running:
            return

        logger.info(f"Stopping worker {self.worker_id}")

        self._running = False

        # 停止心跳
        if self._heartbeat_manager:
            await self._heartbeat_manager.stop()

        # 清理
        await self.teardown()

        # 断开 Redis
        await self.stream_manager.disconnect()

    async def _consume_tasks(self):
        """消费任务循环"""
        stream = self.get_task_stream()
        consumer_name = self.worker_id

        await self.stream_manager.consume(
            stream=stream,
            consumer_name=consumer_name,
            handler=self._handle_task
        )

    async def _handle_task(self, task: Dict[str, Any]):
        """处理单个任务"""
        task_id = task.get("task_id", "unknown")

        async with self._semaphore:
            self._current_tasks += 1
            self.worker_info.current_load = self._current_tasks

            logger.info(f"Processing task {task_id}")

            try:
                # 调用子类实现
                result = await self.process(task)

                # 发布结果
                await self.stream_manager.publish_result(task_id, result)

                logger.info(f"Task {task_id} completed successfully")

            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")

                # 发布错误结果
                error_result = {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e)
                }
                await self.stream_manager.publish_result(task_id, error_result)

            finally:
                self._current_tasks -= 1
                self.worker_info.current_load = self._current_tasks

    async def _on_heartbeat(self, worker_info: WorkerInfo):
        """心跳回调"""
        # 发布心跳到 Redis
        heartbeat_key = f"heartbeat:{self.worker_id}"
        await self.stream_manager.redis.setex(
            heartbeat_key,
            30,  # 30秒过期
            worker_info.model_dump_json()
        )


class ASRWorkerBase(BaseWorker):
    """ASR Worker 基类"""
    worker_type = "asr"

    @abstractmethod
    async def transcribe(self, audio_data: bytes, options: Dict[str, Any]) -> str:
        """转录音频（子类实现）

        Args:
            audio_data: 音频数据
            options: 转录选项

        Returns:
            转录文本
        """
        pass

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理 ASR 任务"""
        import base64
        from shared.models import TaskStatus

        # 解码音频
        audio_b64 = task.get("audio", {}).get("data", "")
        audio_data = base64.b64decode(audio_b64)

        options = task.get("options", {})

        # 调用转录
        text = await self.transcribe(audio_data, options)

        return {
            "task_id": task.get("task_id"),
            "status": TaskStatus.COMPLETED.value,
            "text": text,
            "confidence": 0.95  # 子类可以返回实际置信度
        }


class AgentWorkerBase(BaseWorker):
    """Agent Worker 基类"""
    worker_type = "agent"

    @abstractmethod
    async def generate_response(
        self,
        input_text: str,
        context: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成响应（子类实现）

        Args:
            input_text: 用户输入
            context: 对话上下文
            options: Agent 选项

        Returns:
            响应数据 {"text": str, "intent": str, "tools": list, ...}
        """
        pass

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理 Agent 任务"""
        from shared.models import TaskStatus

        input_data = task.get("input", {})
        input_text = input_data.get("text", "")

        # 如果是音频输入，这里应该先 ASR（可以调用 ASR 服务）
        if input_data.get("type") == "audio":
            # TODO: 调用 ASR 服务
            input_text = "[audio transcription]"

        context = task.get("context", {})
        options = task.get("options", {})

        # 调用生成
        response = await self.generate_response(input_text, context, options)

        return {
            "task_id": task.get("task_id"),
            "status": TaskStatus.COMPLETED.value,
            "output": {
                "text": response.get("text", ""),
                "type": "text"
            },
            "intent": response.get("intent"),
            "tool_calls": response.get("tools", [])
        }
