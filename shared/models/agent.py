"""
共享数据模型 - Agent 相关
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union
from enum import Enum
from datetime import datetime
from .common import BaseTask, BaseResult, generate_id
from .asr import AudioData


class AgentType(str, Enum):
    """Agent 类型"""
    VOICE_ASSISTANT = "voice_assistant"      # 通用语音助手
    TASK_EXECUTOR = "task_executor"          # 任务执行 Agent
    REHABILITATION = "rehabilitation"        # 康复训练 Agent
    COMPANION = "companion"                  # 陪伴聊天 Agent


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationMessage(BaseModel):
    """对话消息"""
    message_id: str = Field(default_factory=lambda: generate_id("msg"))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # 关联的音频（如果有）
    audio_url: Optional[str] = None
    audio_duration_ms: Optional[int] = None

    # ASR 相关（用户消息）
    asr_confidence: Optional[float] = None
    original_asr_text: Optional[str] = None

    # 工具调用（助手消息）
    tool_calls: Optional[List[dict]] = None

    class Config:
        use_enum_values = True


class AgentInputType(str, Enum):
    """Agent 输入类型"""
    AUDIO = "audio"      # 语音输入
    TEXT = "text"        # 文本输入
    COMMAND = "command"  # 命令输入（如中断、确认等）


class AgentInput(BaseModel):
    """Agent 输入

    支持语音或文本输入
    """
    type: AgentInputType

    # 音频输入（type=audio 时必填）
    audio: Optional[AudioData] = None

    # 文本输入
    text: Optional[str] = None

    # ASR 元信息（如果文本来自外部 ASR）
    asr_confidence: Optional[float] = None
    asr_original_text: Optional[str] = None

    # 命令输入
    command: Optional[str] = None  # interrupt, confirm, cancel, repeat

    def validate_input(self):
        """验证输入完整性"""
        if self.type == AgentInputType.AUDIO and not self.audio:
            raise ValueError("音频输入必须提供 audio 字段")
        if self.type == AgentInputType.TEXT and not self.text:
            raise ValueError("文本输入必须提供 text 字段")
        return True


class AgentContext(BaseModel):
    """Agent 上下文

    包含对话历史和用户信息
    """
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list,
        description="对话历史（最近 N 轮）"
    )
    max_history_turns: int = Field(10, description="保留的最大历史轮数")

    # 用户信息
    user_vocabulary: List[str] = Field(
        default_factory=list,
        description="用户常用词汇（提高识别准确率）"
    )
    user_profile: Optional[dict] = Field(
        None,
        description="用户画像（偏好、习惯等）"
    )
    user_name: Optional[str] = Field(None, description="用户称呼")

    # 当前会话状态
    current_intent: Optional[str] = None
    current_slots: dict = Field(default_factory=dict)
    pending_confirmation: Optional[str] = None

    def add_message(self, message: ConversationMessage):
        """添加消息到历史"""
        self.conversation_history.append(message)
        # 保持历史长度
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[-self.max_history_turns * 2:]


class TTSVoice(str, Enum):
    """TTS 声音选项"""
    ZH_FEMALE_CALM = "zh_female_calm"       # 中文女声-平静
    ZH_FEMALE_WARM = "zh_female_warm"       # 中文女声-温暖
    ZH_MALE_CALM = "zh_male_calm"           # 中文男声-平静
    EN_FEMALE_CALM = "en_female_calm"       # 英文女声-平静


class AgentOptions(BaseModel):
    """Agent 选项配置"""

    # TTS 选项
    enable_tts: bool = Field(True, description="是否返回语音回复")
    tts_voice: TTSVoice = Field(TTSVoice.ZH_FEMALE_CALM, description="TTS 声音")
    tts_speed: float = Field(1.0, ge=0.5, le=2.0, description="TTS 语速")

    # LLM 选项
    llm_model: str = Field("claude-3-5-sonnet", description="LLM 模型")
    max_response_tokens: int = Field(500, description="最大回复 token 数")
    temperature: float = Field(0.7, ge=0, le=1, description="LLM temperature")
    streaming: bool = Field(True, description="是否流式输出")

    # 工具选项
    enable_tools: bool = Field(True, description="是否启用工具调用")
    allowed_tools: Optional[List[str]] = Field(
        None,
        description="允许的工具列表，None 表示全部允许"
    )

    # 行为选项
    auto_confirm: bool = Field(
        False,
        description="是否自动确认（不询问用户确认）"
    )
    clarification_threshold: float = Field(
        0.7,
        description="低于此置信度时请求澄清"
    )
    max_clarification_attempts: int = Field(3, description="最大澄清尝试次数")

    class Config:
        use_enum_values = True


class AgentTask(BaseTask):
    """Agent 任务

    语音/文本对话任务
    """
    task_type: str = "agent"
    agent_type: AgentType = Field(AgentType.VOICE_ASSISTANT)

    input: AgentInput = Field(..., description="用户输入")
    context: AgentContext = Field(default_factory=AgentContext)
    options: AgentOptions = Field(default_factory=AgentOptions)


class ToolCall(BaseModel):
    """工具调用"""
    tool_call_id: str = Field(default_factory=lambda: generate_id("tc"))
    tool_name: str = Field(..., description="工具名称")
    arguments: dict = Field(default_factory=dict, description="调用参数")

    # 执行结果
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


class IntentInfo(BaseModel):
    """意图识别结果"""
    intent_name: str = Field(..., description="意图名称")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    slots: dict = Field(default_factory=dict, description="槽位信息")

    # 是否需要澄清
    needs_clarification: bool = Field(False)
    clarification_question: Optional[str] = None


class AgentOutput(BaseModel):
    """Agent 输出"""

    # 文本回复
    text: str = Field(..., description="文本回复内容")

    # 语音输出（如果启用 TTS）
    audio: Optional[AudioData] = None

    # 意图识别结果
    intent: Optional[IntentInfo] = None

    # 工具调用
    tool_calls: List[ToolCall] = Field(default_factory=list)

    # 建议的后续操作/问题
    suggestions: List[str] = Field(
        default_factory=list,
        description="建议的后续问题或操作"
    )

    # 情感标签
    emotion: Optional[str] = Field(None, description="回复的情感标签")

    # 是否需要用户确认
    needs_confirmation: bool = Field(False)
    confirmation_prompt: Optional[str] = None


class AgentResult(BaseResult):
    """Agent 任务结果"""
    output: Optional[AgentOutput] = None

    # 输入的 ASR 结果（如果是语音输入）
    asr_text: Optional[str] = None
    asr_confidence: Optional[float] = None

    # 性能指标
    asr_time_ms: Optional[int] = None
    llm_time_ms: Optional[int] = None
    tts_time_ms: Optional[int] = None
    tool_time_ms: Optional[int] = None
    total_time_ms: Optional[int] = None


class AgentStreamingChunkType(str, Enum):
    """流式 Agent 输出块类型"""
    ASR_PARTIAL = "asr_partial"      # ASR 部分结果
    ASR_FINAL = "asr_final"          # ASR 最终结果
    THINKING = "thinking"             # 思考中
    TEXT_DELTA = "text_delta"         # 文本增量
    TEXT_DONE = "text_done"           # 文本完成
    AUDIO_CHUNK = "audio_chunk"       # 音频块
    AUDIO_DONE = "audio_done"         # 音频完成
    TOOL_CALL = "tool_call"           # 工具调用
    TOOL_RESULT = "tool_result"       # 工具结果
    ERROR = "error"                   # 错误
    DONE = "done"                     # 完成


class AgentStreamingChunk(BaseModel):
    """流式 Agent 输出块"""
    session_id: str
    chunk_type: AgentStreamingChunkType

    # ASR 相关
    asr_text: Optional[str] = None
    asr_confidence: Optional[float] = None

    # 文本相关
    text_delta: Optional[str] = None
    accumulated_text: Optional[str] = None

    # 音频相关
    audio_chunk: Optional[str] = None  # Base64 编码的音频块

    # 工具调用相关
    tool_call: Optional[ToolCall] = None

    # 思考/状态消息
    status_message: Optional[str] = None

    # 错误信息
    error: Optional[str] = None

    # 完成标记
    is_done: bool = False
    final_result: Optional[AgentResult] = None

    # 时间戳
    timestamp_ms: int = Field(..., description="时间戳")

    class Config:
        use_enum_values = True


# ============== 工具定义 ==============

class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = False
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    """工具定义"""
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    parameters: List[ToolParameter] = Field(default_factory=list)

    # 执行配置
    timeout_ms: int = Field(10000, description="执行超时")
    requires_confirmation: bool = Field(False, description="是否需要用户确认")


# ============== 预定义意图 ==============

class PredefinedIntent(str, Enum):
    """预定义意图"""
    # 查询类
    QUERY_WEATHER = "query_weather"
    QUERY_TIME = "query_time"
    QUERY_DATE = "query_date"

    # 任务类
    SET_REMINDER = "set_reminder"
    SET_ALARM = "set_alarm"
    MAKE_CALL = "make_call"
    SEND_MESSAGE = "send_message"

    # 控制类
    PLAY_MUSIC = "play_music"
    STOP_MUSIC = "stop_music"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"

    # 对话类
    GREETING = "greeting"
    GOODBYE = "goodbye"
    THANKS = "thanks"
    HELP = "help"
    REPEAT = "repeat"
    CANCEL = "cancel"
    CONFIRM = "confirm"
    DENY = "deny"

    # 未知
    UNKNOWN = "unknown"
