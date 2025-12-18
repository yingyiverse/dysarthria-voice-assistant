"""
AI Worker 测试脚本

测试 ASR Worker 和 Agent Worker 的核心功能。
运行方式: pytest tests/test_ai_workers.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_sdk import SDKConfig, ASRWorkerBase, AgentWorkerBase
from agent_sdk.worker_base import BaseWorker
from shared.models import (
    TaskStatus,
    WorkerInfo,
    WorkerStatus,
)


class MockASRWorker(ASRWorkerBase):
    """测试用的 ASR Worker 实现"""

    def __init__(self, config, response_text="测试转录结果"):
        super().__init__(config, max_concurrent=2)
        self.response_text = response_text
        self.setup_called = False
        self.teardown_called = False
        self.transcribe_calls = []

    async def setup(self):
        self.setup_called = True

    async def teardown(self):
        self.teardown_called = True

    async def transcribe(self, audio_data: bytes, options: dict) -> str:
        self.transcribe_calls.append({
            "audio_size": len(audio_data),
            "options": options
        })
        return self.response_text


class MockAgentWorker(AgentWorkerBase):
    """测试用的 Agent Worker 实现"""

    def __init__(self, config, response_text="测试回复"):
        super().__init__(config, max_concurrent=2)
        self.response_text = response_text
        self.setup_called = False
        self.generate_calls = []

    async def setup(self):
        self.setup_called = True

    async def teardown(self):
        pass

    async def generate_response(
        self,
        input_text: str,
        context: dict,
        options: dict
    ) -> dict:
        self.generate_calls.append({
            "input": input_text,
            "context": context,
            "options": options
        })
        return {
            "text": self.response_text,
            "intent": "test_intent",
            "tools": [],
            "confidence": 0.9
        }


class TestASRWorkerBase:
    """ASR Worker 基类测试"""

    @pytest.fixture
    def config(self):
        return SDKConfig()

    @pytest.fixture
    def worker(self, config):
        return MockASRWorker(config)

    def test_worker_type(self, worker):
        """测试 Worker 类型"""
        assert worker.worker_type == "asr"

    def test_worker_id_generation(self, worker):
        """测试 Worker ID 生成"""
        assert worker.worker_id.startswith("worker-asr_")

    def test_worker_info(self, worker):
        """测试 Worker 信息"""
        info = worker.worker_info

        assert info.worker_type == "asr"
        assert info.status == WorkerStatus.HEALTHY
        assert info.max_concurrent == 2

    @pytest.mark.asyncio
    async def test_process_asr_task(self, worker):
        """测试处理 ASR 任务"""
        audio_bytes = b"fake audio data"
        audio_b64 = base64.b64encode(audio_bytes).decode()

        task = {
            "task_id": "task_123",
            "audio": {"data": audio_b64},
            "options": {"language": "zh", "enable_ger": True}
        }

        result = await worker.process(task)

        assert result["task_id"] == "task_123"
        assert result["text"] == "测试转录结果"
        assert len(worker.transcribe_calls) == 1

    @pytest.mark.asyncio
    async def test_transcribe_receives_options(self, worker):
        """测试转录函数接收正确的选项"""
        audio_b64 = base64.b64encode(b"audio").decode()

        task = {
            "task_id": "task_123",
            "audio": {"data": audio_b64},
            "options": {"language": "en", "enable_punctuation": False}
        }

        await worker.process(task)

        assert worker.transcribe_calls[0]["options"]["language"] == "en"
        assert worker.transcribe_calls[0]["options"]["enable_punctuation"] is False


class TestAgentWorkerBase:
    """Agent Worker 基类测试"""

    @pytest.fixture
    def config(self):
        return SDKConfig()

    @pytest.fixture
    def worker(self, config):
        return MockAgentWorker(config)

    def test_worker_type(self, worker):
        """测试 Worker 类型"""
        assert worker.worker_type == "agent"

    @pytest.mark.asyncio
    async def test_process_text_input(self, worker):
        """测试处理文本输入"""
        task = {
            "task_id": "task_456",
            "input": {
                "type": "text",
                "text": "你好"
            },
            "context": {},
            "options": {"temperature": 0.7}
        }

        result = await worker.process(task)

        assert result["task_id"] == "task_456"
        assert result["output"]["text"] == "测试回复"
        assert len(worker.generate_calls) == 1

    @pytest.mark.asyncio
    async def test_generate_receives_context(self, worker):
        """测试生成函数接收正确的上下文"""
        task = {
            "task_id": "task_456",
            "input": {"type": "text", "text": "查天气"},
            "context": {
                "conversation_history": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "您好！"}
                ],
                "user_profile": {"name": "测试用户"}
            },
            "options": {}
        }

        await worker.process(task)

        call = worker.generate_calls[0]
        assert call["input"] == "查天气"
        assert "conversation_history" in call["context"]
        assert len(call["context"]["conversation_history"]) == 2


class TestWorkerLifecycle:
    """Worker 生命周期测试"""

    @pytest.fixture
    def config(self):
        return SDKConfig()

    @pytest.mark.asyncio
    async def test_setup_called_on_start(self, config):
        """测试启动时调用 setup"""
        worker = MockASRWorker(config)

        # Mock Redis 连接
        with patch.object(worker.stream_manager, 'connect', new_callable=AsyncMock):
            with patch.object(worker, '_consume_tasks', new_callable=AsyncMock):
                await worker.start()

        assert worker.setup_called is True
        assert worker._running is True

    @pytest.mark.asyncio
    async def test_teardown_called_on_stop(self, config):
        """测试停止时调用 teardown"""
        worker = MockASRWorker(config)
        worker._running = True

        with patch.object(worker.stream_manager, 'disconnect', new_callable=AsyncMock):
            with patch.object(worker._heartbeat_manager or MagicMock(), 'stop', new_callable=AsyncMock):
                await worker.stop()

        assert worker.teardown_called is True
        assert worker._running is False


class TestConcurrencyControl:
    """并发控制测试"""

    @pytest.fixture
    def config(self):
        return SDKConfig()

    @pytest.mark.asyncio
    async def test_max_concurrent_respected(self, config):
        """测试最大并发数限制"""
        worker = MockASRWorker(config, response_text="result")
        worker._semaphore = asyncio.Semaphore(2)

        # 追踪并发任务数
        max_concurrent = 0
        current = 0

        original_process = worker.process

        async def tracked_process(task):
            nonlocal max_concurrent, current
            current += 1
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.1)  # 模拟处理时间
            current -= 1
            return await original_process(task)

        worker.process = tracked_process

        # 创建多个任务
        audio_b64 = base64.b64encode(b"audio").decode()
        tasks = [
            {"task_id": f"task_{i}", "audio": {"data": audio_b64}, "options": {}}
            for i in range(5)
        ]

        # 并发执行
        async def run_task(task):
            async with worker._semaphore:
                return await worker.process(task)

        await asyncio.gather(*[run_task(t) for t in tasks])

        # 最大并发应该不超过 2
        assert max_concurrent <= 2


class TestErrorHandling:
    """错误处理测试"""

    @pytest.fixture
    def config(self):
        return SDKConfig()

    @pytest.mark.asyncio
    async def test_transcribe_error_handling(self, config):
        """测试转录错误处理"""

        class FailingASRWorker(ASRWorkerBase):
            async def setup(self):
                pass

            async def transcribe(self, audio_data, options):
                raise ValueError("模型加载失败")

        worker = FailingASRWorker(config)

        audio_b64 = base64.b64encode(b"audio").decode()
        task = {
            "task_id": "task_fail",
            "audio": {"data": audio_b64},
            "options": {}
        }

        with pytest.raises(ValueError, match="模型加载失败"):
            await worker.process(task)

    @pytest.mark.asyncio
    async def test_generate_error_handling(self, config):
        """测试生成错误处理"""

        class FailingAgentWorker(AgentWorkerBase):
            async def setup(self):
                pass

            async def generate_response(self, input_text, context, options):
                raise ConnectionError("LLM 服务不可用")

        worker = FailingAgentWorker(config)

        task = {
            "task_id": "task_fail",
            "input": {"type": "text", "text": "hello"},
            "context": {},
            "options": {}
        }

        with pytest.raises(ConnectionError, match="LLM 服务不可用"):
            await worker.process(task)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
