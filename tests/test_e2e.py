"""
端到端集成测试脚本

测试完整的任务流程：从客户端提交任务到 Worker 处理并返回结果。
需要运行 Redis 服务。

运行方式:
    # 启动 Redis
    docker run -d -p 6379:6379 redis:7-alpine

    # 运行测试
    pytest tests/test_e2e.py -v

    # 跳过集成测试
    SKIP_INTEGRATION_TESTS=1 pytest tests/test_e2e.py -v
"""

import asyncio
import pytest
import os
import sys
import base64
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_sdk import (
    AgentSDK,
    SDKConfig,
    RedisConfig,
    ASRWorkerBase,
    AgentWorkerBase,
)
from shared.models import (
    AudioData,
    AudioFormat,
    ASROptions,
    AgentInput,
    AgentInputType,
    TaskStatus,
)

SKIP_INTEGRATION = os.environ.get("SKIP_INTEGRATION_TESTS") == "1"


class TestASRWorker(ASRWorkerBase):
    """测试用 ASR Worker"""

    async def setup(self):
        print("ASR Worker 初始化完成")

    async def transcribe(self, audio_data: bytes, options: dict) -> str:
        # 简单的回显测试
        return f"转录结果: 收到 {len(audio_data)} 字节音频"


class TestAgentWorker(AgentWorkerBase):
    """测试用 Agent Worker"""

    async def setup(self):
        print("Agent Worker 初始化完成")

    async def generate_response(
        self,
        input_text: str,
        context: dict,
        options: dict
    ) -> dict:
        return {
            "text": f"Agent 回复: {input_text}",
            "intent": "echo",
            "tools": [],
            "confidence": 0.95
        }


@pytest.fixture
def sdk_config():
    """SDK 配置"""
    return SDKConfig(
        redis=RedisConfig(url="redis://localhost:6379"),
        stream_prefix="test",
        default_timeout_ms=10000
    )


@pytest.mark.skipif(SKIP_INTEGRATION, reason="跳过集成测试")
class TestEndToEnd:
    """端到端集成测试"""

    @pytest.mark.asyncio
    async def test_sdk_client_lifecycle(self, sdk_config):
        """测试 SDK 客户端生命周期"""
        async with AgentSDK(sdk_config) as sdk:
            assert sdk._connected is True

        assert sdk._connected is False

    @pytest.mark.asyncio
    async def test_stream_manager_operations(self, sdk_config):
        """测试 Stream 管理器基本操作"""
        sdk = AgentSDK(sdk_config)
        await sdk.connect()

        try:
            # 发布消息
            test_stream = "test:e2e:stream"
            msg_id = await sdk.stream_manager.publish(
                test_stream,
                {"task_id": "test_123", "data": "hello"}
            )
            assert msg_id is not None

            # 获取 Stream 长度
            length = await sdk.stream_manager.get_stream_length(test_stream)
            assert length >= 1

            # 清理
            await sdk.stream_manager.delete_stream(test_stream)

        finally:
            await sdk.disconnect()

    @pytest.mark.asyncio
    async def test_worker_registration(self, sdk_config):
        """测试 Worker 注册"""
        from shared.models import WorkerInfo, WorkerStatus

        sdk = AgentSDK(sdk_config)
        await sdk.connect()

        try:
            worker_info = WorkerInfo(
                worker_id="test-asr-worker-1",
                worker_type="asr",
                status=WorkerStatus.HEALTHY,
                max_concurrent=4
            )

            await sdk.register_worker(worker_info)

            workers = await sdk.get_workers("asr")
            assert len(workers) == 1
            assert workers[0].worker_id == "test-asr-worker-1"

        finally:
            await sdk.disconnect()


@pytest.mark.skipif(SKIP_INTEGRATION, reason="跳过集成测试")
class TestFullWorkflow:
    """完整工作流测试"""

    @pytest.mark.asyncio
    async def test_asr_full_workflow(self, sdk_config):
        """测试完整的 ASR 工作流

        1. 启动 Worker
        2. SDK 提交任务
        3. Worker 处理任务
        4. SDK 接收结果
        """
        # 创建 Worker
        worker = TestASRWorker(sdk_config, worker_id="e2e-asr-worker")

        # 启动 Worker 在后台
        worker_task = None

        async def start_worker():
            # Mock 消费循环以便测试控制
            await worker.setup()
            await worker.stream_manager.connect()

        await start_worker()

        try:
            # 创建 SDK 客户端
            async with AgentSDK(sdk_config) as sdk:
                # 发布一个任务到 Stream
                audio_data = b"test audio data for e2e"
                audio_b64 = base64.b64encode(audio_data).decode()

                task_data = {
                    "task_id": "e2e_task_001",
                    "task_type": "asr",
                    "user_id": "test_user",
                    "audio": {
                        "data": audio_b64,
                        "format": "pcm",
                        "sample_rate": 16000
                    },
                    "options": {
                        "engine": "sensevoice",
                        "language": "zh"
                    }
                }

                # 发布任务
                msg_id = await sdk.stream_manager.publish(
                    sdk_config.asr_task_stream,
                    task_data
                )
                assert msg_id is not None

                # 模拟 Worker 处理
                result = await worker.process(task_data)
                assert "转录结果" in result["text"]

                # 发布结果
                await sdk.stream_manager.publish_result(
                    task_data["task_id"],
                    result
                )

                # 验证结果可以被获取
                result_data = await sdk.stream_manager.get_result(
                    f"results:{task_data['task_id']}",
                    timeout_ms=5000
                )
                assert result_data is not None

        finally:
            await worker.stream_manager.disconnect()


class TestMockWorkflow:
    """使用 Mock 的工作流测试（不需要 Redis）"""

    @pytest.fixture
    def mock_sdk(self):
        """创建 Mock SDK"""
        config = SDKConfig()
        sdk = AgentSDK(config)

        # Mock stream manager
        sdk.stream_manager.connect = AsyncMock()
        sdk.stream_manager.disconnect = AsyncMock()
        sdk.stream_manager.publish = AsyncMock(return_value="msg_123")
        sdk.stream_manager.get_result = AsyncMock(return_value={
            "task_id": "task_123",
            "status": "completed",
            "text": "模拟转录结果",
            "confidence": 0.95
        })

        return sdk

    @pytest.mark.asyncio
    async def test_transcribe_workflow(self, mock_sdk):
        """测试转录工作流（Mock）"""
        await mock_sdk.connect()

        audio = AudioData(
            data=base64.b64encode(b"audio").decode(),
            format=AudioFormat.PCM
        )
        options = ASROptions(enable_ger=True)

        result = await mock_sdk.transcribe(audio, user_id="user_123", options=options)

        assert result.text == "模拟转录结果"
        assert result.confidence == 0.95

        await mock_sdk.disconnect()

    @pytest.mark.asyncio
    async def test_chat_workflow(self, mock_sdk):
        """测试对话工作流（Mock）"""
        mock_sdk.stream_manager.get_result = AsyncMock(return_value={
            "task_id": "task_456",
            "status": "completed",
            "output": {
                "text": "Agent 回复"
            }
        })

        await mock_sdk.connect()

        result = await mock_sdk.chat_text(
            text="你好",
            user_id="user_123"
        )

        assert result.output.text == "Agent 回复"

        await mock_sdk.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
