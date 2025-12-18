"""
Agent Worker 示例实现

这是一个示例，展示如何基于 Agent SDK 实现自己的 Agent Worker。
用户需要根据实际的 LLM 和 Agent 框架替换具体实现。
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from agent_sdk import SDKConfig, AgentWorkerBase

logger = logging.getLogger(__name__)


class QwenAgentWorker(AgentWorkerBase):
    """基于 Qwen 的 Agent Worker 示例

    实现者需要：
    1. 在 setup() 中初始化 LLM 和工具
    2. 在 generate_response() 中实现对话逻辑
    """

    def __init__(
        self,
        config: SDKConfig,
        model_name: str = "qwen2.5-7b-instruct",
        tools: Optional[List[Dict]] = None
    ):
        super().__init__(config, max_concurrent=2)
        self.model_name = model_name
        self.tools = tools or []
        self.llm = None

    async def setup(self):
        """初始化 LLM 和工具"""
        logger.info(f"Initializing Qwen Agent ({self.model_name})...")

        # TODO: 替换为实际的 LLM 初始化代码
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.llm = AutoModelForCausalLM.from_pretrained(self.model_name)

        # 或使用 API
        # from openai import AsyncOpenAI
        # self.client = AsyncOpenAI(base_url="http://localhost:8000/v1")

        await asyncio.sleep(1)
        logger.info("Qwen Agent initialized")

    async def teardown(self):
        """清理资源"""
        self.llm = None

    async def generate_response(
        self,
        input_text: str,
        context: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成 Agent 响应

        Args:
            input_text: 用户输入文本
            context: 对话上下文，包含：
                - conversation_history: 历史消息列表
                - user_profile: 用户画像
                - session_data: 会话数据
            options: Agent 选项，包含：
                - enable_tools: 是否启用工具
                - temperature: 生成温度
                - max_tokens: 最大 token 数

        Returns:
            响应数据：
            {
                "text": str,           # 回复文本
                "intent": str,         # 识别的意图
                "tools": list,         # 工具调用
                "confidence": float    # 置信度
            }
        """
        logger.info(f"Generating response for: {input_text[:50]}...")

        # 构建消息
        messages = self._build_messages(input_text, context)

        # TODO: 替换为实际的 LLM 调用
        # response = await self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,
        #     tools=self.tools if options.get("enable_tools") else None,
        #     temperature=options.get("temperature", 0.7),
        #     max_tokens=options.get("max_tokens", 512)
        # )
        # assistant_message = response.choices[0].message
        # ...

        # 模拟响应
        await asyncio.sleep(0.5)

        # 简单的意图识别示例
        intent = self._detect_intent(input_text)

        return {
            "text": f"我理解您说的是: {input_text}。这是一个模拟的回复。",
            "intent": intent,
            "tools": [],
            "confidence": 0.9
        }

    def _build_messages(
        self,
        input_text: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """构建 LLM 消息列表"""
        messages = []

        # 系统提示
        system_prompt = self._get_system_prompt(context)
        messages.append({"role": "system", "content": system_prompt})

        # 历史消息
        history = context.get("conversation_history", [])
        for msg in history[-10:]:  # 最近 10 条
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        # 当前输入
        messages.append({"role": "user", "content": input_text})

        return messages

    def _get_system_prompt(self, context: Dict[str, Any]) -> str:
        """获取系统提示"""
        user_profile = context.get("user_profile", {})
        user_name = user_profile.get("name", "用户")

        return f"""你是一个专为构音障碍用户设计的语音助手。
用户名称: {user_name}

你的职责：
1. 理解用户的语音输入（可能存在发音不清晰的情况）
2. 提供友好、耐心的回复
3. 必要时请求用户确认或重复
4. 执行用户请求的任务

请用简洁、清晰的语言回复。"""

    def _detect_intent(self, text: str) -> str:
        """简单的意图识别"""
        text_lower = text.lower()

        if any(w in text_lower for w in ["天气", "weather"]):
            return "query_weather"
        elif any(w in text_lower for w in ["提醒", "remind", "闹钟"]):
            return "set_reminder"
        elif any(w in text_lower for w in ["打电话", "call", "拨打"]):
            return "make_call"
        elif any(w in text_lower for w in ["播放", "音乐", "play"]):
            return "play_music"
        else:
            return "general_chat"


class GERAgentWorker(AgentWorkerBase):
    """GER (Generative Error Correction) Agent Worker

    专门用于纠正 ASR 输出的错误，提高构音障碍用户的识别准确率。
    """

    worker_type = "ger"  # 覆盖类型

    def __init__(self, config: SDKConfig, model_name: str = "qwen2.5-7b-instruct"):
        super().__init__(config, max_concurrent=4)
        self.model_name = model_name

    async def setup(self):
        """初始化 GER 模型"""
        logger.info("Initializing GER model...")
        await asyncio.sleep(0.5)
        logger.info("GER model initialized")

    async def generate_response(
        self,
        input_text: str,
        context: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行生成式错误纠正

        Args:
            input_text: ASR 原始输出
            context: 包含 N-best 假设列表
            options: GER 选项

        Returns:
            纠正后的文本
        """
        logger.info(f"GER correcting: {input_text}")

        # 获取 N-best 假设
        nbest = context.get("nbest_hypotheses", [input_text])

        # TODO: 实现实际的 GER 逻辑
        # 参考 dysarthria_asr 仓库中的 GER 实现
        # prompt = self._build_ger_prompt(nbest, context.get("user_profile"))
        # corrected = await self._call_llm(prompt)

        # 模拟纠正
        await asyncio.sleep(0.3)
        corrected_text = input_text  # 实际应返回纠正后的文本

        return {
            "text": corrected_text,
            "intent": "correction",
            "tools": [],
            "confidence": 0.85,
            "original": input_text,
            "was_corrected": corrected_text != input_text
        }


async def main():
    """Worker 启动入口"""
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config = SDKConfig.from_env()

    # 选择 Worker 类型
    worker_type = os.getenv("AGENT_WORKER_TYPE", "qwen")

    if worker_type == "qwen":
        worker = QwenAgentWorker(
            config,
            model_name=os.getenv("QWEN_MODEL_NAME", "qwen2.5-7b-instruct")
        )
    else:
        worker = GERAgentWorker(
            config,
            model_name=os.getenv("GER_MODEL_NAME", "qwen2.5-7b-instruct")
        )

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
