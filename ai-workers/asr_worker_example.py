"""
ASR Worker 示例实现

这是一个示例，展示如何基于 Agent SDK 实现自己的 ASR Worker。
用户需要根据实际的 ASR 模型（如 SenseVoice + LoRA）替换具体实现。
"""

import asyncio
import logging
from typing import Dict, Any

# 从 agent-sdk 导入基类
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from agent_sdk import SDKConfig, ASRWorkerBase

logger = logging.getLogger(__name__)


class SenseVoiceASRWorker(ASRWorkerBase):
    """基于 SenseVoice 的 ASR Worker 示例

    实现者需要：
    1. 在 setup() 中加载模型
    2. 在 transcribe() 中实现转录逻辑
    """

    def __init__(self, config: SDKConfig, model_path: str = None):
        super().__init__(config, max_concurrent=2)
        self.model_path = model_path
        self.model = None

    async def setup(self):
        """加载 ASR 模型

        示例：加载 SenseVoice + LoRA 微调模型
        """
        logger.info("Loading SenseVoice model...")

        # TODO: 替换为实际的模型加载代码
        # from funasr import AutoModel
        # self.model = AutoModel(
        #     model=self.model_path or "iic/SenseVoiceSmall",
        #     vad_model="fsmn-vad",
        #     ...
        # )

        # 模拟模型加载
        await asyncio.sleep(1)
        logger.info("SenseVoice model loaded")

    async def teardown(self):
        """清理资源"""
        logger.info("Unloading model...")
        self.model = None

    async def transcribe(self, audio_data: bytes, options: Dict[str, Any]) -> str:
        """执行语音转录

        Args:
            audio_data: 音频字节数据
            options: 转录选项，包含：
                - language: 语言代码
                - enable_punctuation: 是否启用标点
                - enable_itn: 是否启用逆文本正则化
                - hotwords: 热词列表

        Returns:
            转录文本
        """
        logger.info(f"Transcribing audio ({len(audio_data)} bytes)")

        # TODO: 替换为实际的转录代码
        # result = self.model.generate(
        #     input=audio_data,
        #     language=options.get("language", "auto"),
        #     use_itn=options.get("enable_itn", True),
        #     ...
        # )
        # return result[0]["text"]

        # 模拟转录（实际实现时删除）
        await asyncio.sleep(0.5)
        return "这是一个模拟的转录结果"


class WhisperASRWorker(ASRWorkerBase):
    """基于 Whisper 的 ASR Worker 示例"""

    def __init__(self, config: SDKConfig, model_size: str = "base"):
        super().__init__(config, max_concurrent=1)
        self.model_size = model_size
        self.model = None

    async def setup(self):
        """加载 Whisper 模型"""
        logger.info(f"Loading Whisper model ({self.model_size})...")

        # TODO: 替换为实际的模型加载代码
        # import whisper
        # self.model = whisper.load_model(self.model_size)

        await asyncio.sleep(2)
        logger.info("Whisper model loaded")

    async def transcribe(self, audio_data: bytes, options: Dict[str, Any]) -> str:
        """执行 Whisper 转录"""
        logger.info(f"Transcribing with Whisper...")

        # TODO: 替换为实际的转录代码
        # import tempfile
        # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        #     f.write(audio_data)
        #     result = self.model.transcribe(f.name, language=options.get("language"))
        # return result["text"]

        await asyncio.sleep(1)
        return "Whisper 模拟转录结果"


async def main():
    """Worker 启动入口"""
    import os

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 加载配置
    config = SDKConfig.from_env()

    # 选择 Worker 类型
    worker_type = os.getenv("ASR_WORKER_TYPE", "sensevoice")

    if worker_type == "sensevoice":
        worker = SenseVoiceASRWorker(
            config,
            model_path=os.getenv("SENSEVOICE_MODEL_PATH")
        )
    else:
        worker = WhisperASRWorker(
            config,
            model_size=os.getenv("WHISPER_MODEL_SIZE", "base")
        )

    # 启动 Worker
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
