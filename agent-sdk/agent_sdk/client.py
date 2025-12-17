"""
Agent SDK - 客户端入口
"""

import asyncio
import logging
from typing import Callable, Optional, AsyncIterator
from datetime import datetime

from .config import SDKConfig
from .stream.stream_manager import StreamManager
from .dispatcher.task_dispatcher import TaskDispatcher
from .pool.worker_manager import WorkerPool

# 导入共享模型
import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from shared.models import (
    ASRTask, ASRResult, ASRStreamingTask, ASRStreamingChunk,
    AgentTask, AgentResult, AgentStreamingChunk,
    AudioData, ASROptions, AgentOptions, AgentContext, AgentInput,
    WorkerInfo, generate_id
)

logger = logging.getLogger(__name__)


class AgentSDK:
    """Agent SDK 主入口

    提供统一的接口来调用 AI 服务：
    - ASR 语音识别
    - Agent 对话
    - 流式处理

    Usage:
        sdk = AgentSDK(config)
        await sdk.connect()

        # 单次 ASR
        result = await sdk.transcribe(audio_data, user_id="user_123")

        # 流式 ASR
        session = await sdk.create_streaming_session(user_id="user_123")
        async for chunk in session.stream(audio_chunks):
            print(chunk.text)

        # Agent 对话
        response = await sdk.chat(input_text, user_id="user_123")
    """

    def __init__(self, config: Optional[SDKConfig] = None):
        self.config = config or SDKConfig.from_env()

        # 核心组件
        self.stream_manager = StreamManager(self.config.redis)
        self.worker_pool = WorkerPool(self.config.workers)
        self.task_dispatcher = TaskDispatcher(
            stream_manager=self.stream_manager,
            worker_pool=self.worker_pool,
            config=self.config
        )

        self._connected = False

    async def connect(self):
        """连接到后端服务"""
        if self._connected:
            return

        await self.stream_manager.connect()
        self._connected = True
        logger.info("Agent SDK connected")

    async def disconnect(self):
        """断开连接"""
        if not self._connected:
            return

        await self.stream_manager.disconnect()
        self._connected = False
        logger.info("Agent SDK disconnected")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    # ==================== ASR 接口 ====================

    async def transcribe(
        self,
        audio: AudioData,
        user_id: str,
        options: Optional[ASROptions] = None,
        timeout_ms: Optional[int] = None
    ) -> ASRResult:
        """单次语音识别

        Args:
            audio: 音频数据
            user_id: 用户 ID
            options: ASR 选项
            timeout_ms: 超时时间

        Returns:
            ASR 结果
        """
        task = ASRTask(
            user_id=user_id,
            audio=audio,
            options=options or ASROptions(),
            timeout_ms=timeout_ms or self.config.default_timeout_ms
        )

        result = await self.task_dispatcher.dispatch_asr(task)
        return result

    async def transcribe_file(
        self,
        file_path: str,
        user_id: str,
        options: Optional[ASROptions] = None
    ) -> ASRResult:
        """转录音频文件

        Args:
            file_path: 文件路径
            user_id: 用户 ID
            options: ASR 选项

        Returns:
            ASR 结果
        """
        import base64

        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        # 推断格式
        ext = file_path.rsplit(".", 1)[-1].lower()
        format_map = {"wav": "wav", "mp3": "mp3", "webm": "webm", "ogg": "ogg"}
        audio_format = format_map.get(ext, "wav")

        audio = AudioData(
            data=base64.b64encode(audio_bytes).decode(),
            format=audio_format
        )

        return await self.transcribe(audio, user_id, options)

    async def create_streaming_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        options: Optional[ASROptions] = None,
        on_result: Optional[Callable[[ASRStreamingChunk], None]] = None
    ) -> "StreamingSession":
        """创建流式转录会话

        Args:
            user_id: 用户 ID
            session_id: 会话 ID（可选）
            options: ASR 选项
            on_result: 结果回调函数

        Returns:
            流式会话对象
        """
        session = StreamingSession(
            sdk=self,
            user_id=user_id,
            session_id=session_id or generate_id("stream"),
            options=options or ASROptions(),
            on_result=on_result
        )
        return session

    # ==================== Agent 接口 ====================

    async def chat(
        self,
        input_data: AgentInput,
        user_id: str,
        context: Optional[AgentContext] = None,
        options: Optional[AgentOptions] = None,
        timeout_ms: Optional[int] = None
    ) -> AgentResult:
        """Agent 对话（单次）

        Args:
            input_data: 用户输入（文本或音频）
            user_id: 用户 ID
            context: 对话上下文
            options: Agent 选项
            timeout_ms: 超时时间

        Returns:
            Agent 结果
        """
        task = AgentTask(
            user_id=user_id,
            input=input_data,
            context=context or AgentContext(),
            options=options or AgentOptions(),
            timeout_ms=timeout_ms or self.config.default_timeout_ms
        )

        result = await self.task_dispatcher.dispatch_agent(task)
        return result

    async def chat_text(
        self,
        text: str,
        user_id: str,
        context: Optional[AgentContext] = None,
        options: Optional[AgentOptions] = None
    ) -> AgentResult:
        """文本对话

        Args:
            text: 用户输入文本
            user_id: 用户 ID
            context: 对话上下文
            options: Agent 选项

        Returns:
            Agent 结果
        """
        from shared.models import AgentInputType

        input_data = AgentInput(
            type=AgentInputType.TEXT,
            text=text
        )
        return await self.chat(input_data, user_id, context, options)

    async def chat_audio(
        self,
        audio: AudioData,
        user_id: str,
        context: Optional[AgentContext] = None,
        options: Optional[AgentOptions] = None
    ) -> AgentResult:
        """语音对话

        Args:
            audio: 音频数据
            user_id: 用户 ID
            context: 对话上下文
            options: Agent 选项

        Returns:
            Agent 结果
        """
        from shared.models import AgentInputType

        input_data = AgentInput(
            type=AgentInputType.AUDIO,
            audio=audio
        )
        return await self.chat(input_data, user_id, context, options)

    async def create_agent_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        options: Optional[AgentOptions] = None
    ) -> "AgentSession":
        """创建 Agent 会话（多轮对话）

        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            options: Agent 选项

        Returns:
            Agent 会话对象
        """
        session = AgentSession(
            sdk=self,
            user_id=user_id,
            session_id=session_id or generate_id("agent"),
            options=options or AgentOptions()
        )
        return session

    # ==================== Worker 管理 ====================

    async def register_worker(self, worker: WorkerInfo):
        """注册 Worker"""
        await self.worker_pool.register_worker(worker)

    async def get_workers(self, worker_type: Optional[str] = None):
        """获取 Worker 列表"""
        return self.worker_pool.get_workers(worker_type)


class StreamingSession:
    """流式转录会话"""

    def __init__(
        self,
        sdk: AgentSDK,
        user_id: str,
        session_id: str,
        options: ASROptions,
        on_result: Optional[Callable] = None
    ):
        self.sdk = sdk
        self.user_id = user_id
        self.session_id = session_id
        self.options = options
        self.on_result = on_result

        self._started = False
        self._chunk_id = 0

    async def start(self):
        """启动会话"""
        if self._started:
            return

        # 创建流式任务
        task = ASRStreamingTask(
            user_id=self.user_id,
            session_id=self.session_id,
            options=self.options
        )

        await self.sdk.task_dispatcher.start_streaming_session(task)
        self._started = True

    async def send_audio(self, audio_chunk: bytes):
        """发送音频块"""
        if not self._started:
            await self.start()

        self._chunk_id += 1

        await self.sdk.stream_manager.publish(
            f"stream:{self.session_id}:audio",
            {
                "chunk_id": self._chunk_id,
                "data": audio_chunk.hex(),  # 二进制转 hex
                "timestamp_ms": int(datetime.utcnow().timestamp() * 1000)
            }
        )

    async def get_results(self) -> AsyncIterator[ASRStreamingChunk]:
        """获取结果流"""
        result_stream = f"stream:{self.session_id}:result"

        async for message in self.sdk.stream_manager.subscribe(result_stream):
            chunk = ASRStreamingChunk(**message)

            if self.on_result:
                self.on_result(chunk)

            yield chunk

            if chunk.is_final:
                break

    async def stop(self):
        """停止会话"""
        if not self._started:
            return

        await self.sdk.stream_manager.publish(
            f"stream:{self.session_id}:audio",
            {"type": "end", "chunk_id": self._chunk_id + 1}
        )
        self._started = False


class AgentSession:
    """Agent 多轮对话会话"""

    def __init__(
        self,
        sdk: AgentSDK,
        user_id: str,
        session_id: str,
        options: AgentOptions
    ):
        self.sdk = sdk
        self.user_id = user_id
        self.session_id = session_id
        self.options = options
        self.context = AgentContext()

    async def send_text(self, text: str) -> AgentResult:
        """发送文本消息"""
        result = await self.sdk.chat_text(
            text=text,
            user_id=self.user_id,
            context=self.context,
            options=self.options
        )

        # 更新上下文
        if result.output:
            from shared.models import ConversationMessage, MessageRole
            self.context.add_message(ConversationMessage(
                role=MessageRole.USER,
                content=text
            ))
            self.context.add_message(ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=result.output.text
            ))

        return result

    async def send_audio(self, audio: AudioData) -> AgentResult:
        """发送语音消息"""
        result = await self.sdk.chat_audio(
            audio=audio,
            user_id=self.user_id,
            context=self.context,
            options=self.options
        )

        # 更新上下文
        if result.output:
            from shared.models import ConversationMessage, MessageRole
            self.context.add_message(ConversationMessage(
                role=MessageRole.USER,
                content=result.asr_text or "[audio]"
            ))
            self.context.add_message(ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=result.output.text
            ))

        return result

    def clear_context(self):
        """清空上下文"""
        self.context = AgentContext()
