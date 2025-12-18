# Agent SDK

统一的 AI 服务调用接口，为 Dysarthria Voice Assistant 提供基础设施支持。

## 功能

- **Redis Stream 管理**: 消息发布、订阅、消费者组管理
- **任务调度**: 将任务路由到正确的 Worker
- **Worker 池管理**: 健康检查、负载均衡
- **HTTP 客户端**: 外部 API 调用、重试机制

## 安装

```bash
pip install -e .
```

## 快速开始

### 作为 SDK 客户端使用

```python
from agent_sdk import AgentSDK, SDKConfig

async def main():
    config = SDKConfig.from_env()

    async with AgentSDK(config) as sdk:
        # ASR 转录
        result = await sdk.transcribe_file("audio.wav", user_id="user_123")
        print(f"转录结果: {result.text}")

        # Agent 对话
        response = await sdk.chat_text("你好", user_id="user_123")
        print(f"Agent 回复: {response.output.text}")
```

### 作为 Worker 实现

```python
from agent_sdk import SDKConfig
from agent_sdk.worker_base import ASRWorkerBase

class MyASRWorker(ASRWorkerBase):
    async def setup(self):
        # 加载模型
        self.model = await load_my_asr_model()

    async def transcribe(self, audio_data: bytes, options: dict) -> str:
        # 执行转录
        return self.model.transcribe(audio_data)

async def main():
    config = SDKConfig.from_env()
    worker = MyASRWorker(config)
    await worker.start()
```

## 架构

```
agent-sdk/
├── agent_sdk/
│   ├── __init__.py          # 包入口
│   ├── client.py             # SDK 客户端
│   ├── config.py             # 配置管理
│   ├── worker_base.py        # Worker 基类
│   ├── stream/               # Redis Stream 模块
│   │   └── stream_manager.py
│   ├── dispatcher/           # 任务调度模块
│   │   └── task_dispatcher.py
│   ├── pool/                 # Worker 池模块
│   │   └── worker_manager.py
│   └── http/                 # HTTP 客户端模块
│       └── async_client.py
```

## 配置

通过环境变量配置:

```bash
export REDIS_URL=redis://localhost:6379
export REDIS_PASSWORD=your_password
export SDK_SERVICE_NAME=agent-sdk
```

或通过代码配置:

```python
from agent_sdk import SDKConfig, RedisConfig

config = SDKConfig(
    redis=RedisConfig(
        url="redis://localhost:6379",
        password="your_password"
    ),
    default_timeout_ms=30000
)
```

## API 参考

详见 [API 文档](../docs/API_SPECIFICATION.md)
