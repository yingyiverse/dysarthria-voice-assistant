"""
Agent SDK 单元测试

测试 Agent SDK 核心组件的功能正确性。
运行方式: pytest tests/test_agent_sdk.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_sdk import (
    SDKConfig,
    RedisConfig,
    AgentSDK,
    StreamManager,
    TaskDispatcher,
    WorkerPool,
    WorkerPoolConfig,
)
from shared.models import (
    ASRTask,
    ASRResult,
    ASROptions,
    AudioData,
    AudioFormat,
    TaskStatus,
    WorkerInfo,
    WorkerStatus,
    AgentTask,
    AgentInput,
    AgentInputType,
    AgentContext,
    AgentOptions,
)


class TestSDKConfig:
    """SDK 配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = SDKConfig()

        assert config.service_name == "agent-sdk"
        assert config.default_timeout_ms == 30000
        assert config.stream_prefix == "dva"
        assert config.redis.url == "redis://localhost:6379"

    def test_stream_names(self):
        """测试 Stream 名称生成"""
        config = SDKConfig(stream_prefix="test")

        assert config.asr_task_stream == "test:tasks:asr"
        assert config.agent_task_stream == "test:tasks:agent"
        assert config.asr_streaming_task_stream == "test:tasks:asr_streaming"

    def test_from_env(self):
        """测试从环境变量加载配置"""
        with patch.dict(os.environ, {
            "REDIS_URL": "redis://custom:6380",
            "SDK_SERVICE_NAME": "custom-sdk"
        }):
            config = SDKConfig.from_env()

            assert config.redis.url == "redis://custom:6380"
            assert config.service_name == "custom-sdk"


class TestWorkerPool:
    """Worker 池测试"""

    @pytest.fixture
    def pool(self):
        """创建测试用的 Worker 池"""
        config = WorkerPoolConfig()
        return WorkerPool(config)

    @pytest.fixture
    def sample_worker(self):
        """创建测试用的 Worker 信息"""
        return WorkerInfo(
            worker_id="test-worker-1",
            worker_type="asr",
            status=WorkerStatus.HEALTHY,
            max_concurrent=4,
            current_load=0,
            health_endpoint="http://localhost:8001/health"
        )

    @pytest.mark.asyncio
    async def test_register_worker(self, pool, sample_worker):
        """测试 Worker 注册"""
        await pool.register_worker(sample_worker)

        workers = pool.get_workers("asr")
        assert len(workers) == 1
        assert workers[0].worker_id == "test-worker-1"

    @pytest.mark.asyncio
    async def test_unregister_worker(self, pool, sample_worker):
        """测试 Worker 注销"""
        await pool.register_worker(sample_worker)
        await pool.unregister_worker(sample_worker.worker_id)

        workers = pool.get_workers("asr")
        assert len(workers) == 0

    @pytest.mark.asyncio
    async def test_select_worker_least_loaded(self, pool):
        """测试最少负载选择策略"""
        worker1 = WorkerInfo(
            worker_id="worker-1",
            worker_type="asr",
            status=WorkerStatus.HEALTHY,
            current_load=5
        )
        worker2 = WorkerInfo(
            worker_id="worker-2",
            worker_type="asr",
            status=WorkerStatus.HEALTHY,
            current_load=2
        )

        await pool.register_worker(worker1)
        await pool.register_worker(worker2)

        selected = pool.select_worker("asr")
        assert selected.worker_id == "worker-2"  # 负载更低

    @pytest.mark.asyncio
    async def test_select_worker_excludes_unhealthy(self, pool):
        """测试选择时排除不健康的 Worker"""
        healthy = WorkerInfo(
            worker_id="healthy-worker",
            worker_type="asr",
            status=WorkerStatus.HEALTHY
        )
        unhealthy = WorkerInfo(
            worker_id="unhealthy-worker",
            worker_type="asr",
            status=WorkerStatus.UNHEALTHY
        )

        await pool.register_worker(healthy)
        await pool.register_worker(unhealthy)

        selected = pool.select_worker("asr")
        assert selected.worker_id == "healthy-worker"

    def test_get_pool_stats(self, pool):
        """测试获取池统计信息"""
        stats = pool.get_pool_stats()

        assert "total_workers" in stats
        assert "by_type" in stats
        assert "by_status" in stats


class TestSharedModels:
    """共享数据模型测试"""

    def test_audio_data_validation(self):
        """测试音频数据验证"""
        # 有效的音频数据
        audio = AudioData(
            data="base64encodeddata",
            format=AudioFormat.WAV,
            sample_rate=16000
        )
        assert audio.format == AudioFormat.WAV

        # 使用 URL
        audio_url = AudioData(
            url="http://example.com/audio.wav",
            format=AudioFormat.WAV
        )
        assert audio_url.url is not None

    def test_asr_task_creation(self):
        """测试 ASR 任务创建"""
        audio = AudioData(data="test", format=AudioFormat.PCM)
        options = ASROptions(
            engine="sensevoice",
            enable_ger=True,
            language="zh"
        )

        task = ASRTask(
            user_id="user_123",
            audio=audio,
            options=options
        )

        assert task.task_type == "asr"
        assert task.user_id == "user_123"
        assert task.task_id.startswith("task_")
        assert task.options.enable_ger is True

    def test_asr_result_creation(self):
        """测试 ASR 结果创建"""
        result = ASRResult(
            task_id="task_123",
            status=TaskStatus.COMPLETED,
            text="识别的文本",
            confidence=0.95,
            inference_time_ms=150
        )

        assert result.status == TaskStatus.COMPLETED
        assert result.text == "识别的文本"
        assert result.confidence == 0.95

    def test_agent_input_validation(self):
        """测试 Agent 输入验证"""
        # 文本输入
        text_input = AgentInput(
            type=AgentInputType.TEXT,
            text="你好"
        )
        assert text_input.type == AgentInputType.TEXT

        # 音频输入
        audio = AudioData(data="test", format=AudioFormat.PCM)
        audio_input = AgentInput(
            type=AgentInputType.AUDIO,
            audio=audio
        )
        assert audio_input.type == AgentInputType.AUDIO

    def test_agent_context_add_message(self):
        """测试 Agent 上下文消息管理"""
        from shared.models import ConversationMessage, MessageRole

        context = AgentContext(max_history_turns=5)

        # 添加消息
        for i in range(12):
            msg = ConversationMessage(
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=f"消息 {i}"
            )
            context.add_message(msg)

        # 应该只保留最近的 10 条（5 轮 * 2）
        assert len(context.conversation_history) == 10


class TestStreamManager:
    """Stream 管理器测试"""

    @pytest.fixture
    def config(self):
        return RedisConfig(
            url="redis://localhost:6379",
            consumer_group="test-group"
        )

    @pytest.mark.asyncio
    async def test_message_serialization(self, config):
        """测试消息序列化"""
        manager = StreamManager(config)

        # 测试反序列化
        raw_message = {
            "task_id": "task_123",
            "options": '{"engine": "sensevoice"}',
            "audio": '{"data": "base64data"}'
        }

        deserialized = manager._deserialize_message(raw_message)

        assert deserialized["task_id"] == "task_123"
        assert isinstance(deserialized["options"], dict)
        assert deserialized["options"]["engine"] == "sensevoice"


class TestIntegration:
    """集成测试（需要 Redis）"""

    @pytest.fixture
    def config(self):
        return SDKConfig(
            redis=RedisConfig(url="redis://localhost:6379")
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.environ.get("SKIP_INTEGRATION_TESTS") == "1",
        reason="跳过集成测试"
    )
    async def test_sdk_connect_disconnect(self, config):
        """测试 SDK 连接和断开"""
        sdk = AgentSDK(config)

        await sdk.connect()
        assert sdk._connected is True

        await sdk.disconnect()
        assert sdk._connected is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.environ.get("SKIP_INTEGRATION_TESTS") == "1",
        reason="跳过集成测试"
    )
    async def test_stream_publish_subscribe(self, config):
        """测试 Stream 发布和订阅"""
        manager = StreamManager(config.redis)
        await manager.connect()

        test_stream = "test:integration:stream"
        test_message = {"task_id": "test_123", "data": "hello"}

        # 发布消息
        msg_id = await manager.publish(test_stream, test_message)
        assert msg_id is not None

        # 清理
        await manager.delete_stream(test_stream)
        await manager.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
