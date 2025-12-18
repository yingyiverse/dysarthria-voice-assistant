"""
共享数据模型 - ASR 相关
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from .common import BaseTask, BaseResult, generate_id


class ASREngine(str, Enum):
    """ASR 引擎类型"""
    SENSEVOICE = "sensevoice"
    SENSEVOICE_LORA = "sensevoice_lora"  # 带 LoRA 的个性化模型
    WHISPER = "whisper"
    WHISPER_LARGE = "whisper_large"


class AudioFormat(str, Enum):
    """音频格式"""
    WAV = "wav"
    PCM = "pcm"
    MP3 = "mp3"
    WEBM = "webm"
    OGG = "ogg"


class AudioData(BaseModel):
    """音频数据

    用于传输音频数据，支持 Base64 编码或 URL 引用
    """
    # 二选一：直接数据或 URL
    data: Optional[str] = Field(None, description="Base64 编码的音频数据")
    url: Optional[str] = Field(None, description="音频文件 URL")

    format: AudioFormat = Field(AudioFormat.PCM)
    sample_rate: int = Field(16000, description="采样率 (Hz)")
    channels: int = Field(1, description="通道数")
    bit_depth: int = Field(16, description="位深度")
    duration_ms: Optional[int] = Field(None, description="音频时长（毫秒）")

    def validate_source(self):
        """验证数据源"""
        if not self.data and not self.url:
            raise ValueError("必须提供 data 或 url")
        return True


class ASROptions(BaseModel):
    """ASR 选项配置"""

    # 引擎选择
    engine: ASREngine = Field(ASREngine.SENSEVOICE, description="ASR 引擎")
    language: str = Field("zh", description="语言代码 (zh, en, auto)")

    # 功能开关
    enable_ger: bool = Field(True, description="是否启用 GER 纠错")
    enable_punctuation: bool = Field(True, description="是否添加标点符号")
    enable_word_timestamps: bool = Field(False, description="是否返回词级时间戳")
    enable_vad: bool = Field(True, description="是否启用 VAD 预处理")

    # 个性化选项
    use_personal_model: bool = Field(False, description="是否使用个人模型")
    personal_model_id: Optional[str] = Field(None, description="个人模型 ID")
    user_vocabulary: List[str] = Field(default_factory=list, description="用户词汇表")

    # GER 选项
    ger_confidence_threshold: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="触发 GER 的置信度阈值，低于此值触发纠错"
    )
    ger_use_context: bool = Field(True, description="GER 是否使用对话上下文")

    # 高级选项
    beam_size: int = Field(5, ge=1, le=10, description="Beam search 大小")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="解码温度")


class WordInfo(BaseModel):
    """词级信息"""
    word: str = Field(..., description="词汇")
    start_time: float = Field(..., ge=0, description="开始时间（秒）")
    end_time: float = Field(..., ge=0, description="结束时间（秒）")
    confidence: float = Field(..., ge=0, le=1, description="置信度")


class ASRTask(BaseTask):
    """ASR 任务

    用于批量/单次语音识别请求
    """
    task_type: str = "asr"
    audio: AudioData = Field(..., description="音频数据")
    options: ASROptions = Field(default_factory=ASROptions)


class ASRStreamingTask(BaseTask):
    """流式 ASR 任务

    用于实时语音识别，音频通过 Stream 发送
    """
    task_type: str = "asr_streaming"
    options: ASROptions = Field(default_factory=ASROptions)

    # 流式特有配置
    stream_id: str = Field(
        default_factory=lambda: generate_id("stream"),
        description="音频输入流 ID"
    )
    result_stream_id: str = Field(
        default_factory=lambda: generate_id("result"),
        description="结果输出流 ID"
    )

    # 流式选项
    chunk_duration_ms: int = Field(500, description="每个音频块的时长")
    send_partial_results: bool = Field(True, description="是否发送部分结果")


class ASRResult(BaseResult):
    """ASR 结果"""

    # 识别结果
    text: str = Field("", description="识别文本")
    original_text: Optional[str] = Field(
        None,
        description="GER 纠错前的原始文本（如果启用了 GER）"
    )
    confidence: float = Field(0.0, ge=0, le=1, description="整体置信度")
    is_final: bool = Field(True, description="是否最终结果")

    # 词级信息
    words: Optional[List[WordInfo]] = Field(None, description="词级详细信息")

    # 低置信度词汇（用于 UI 高亮显示）
    low_confidence_words: List[str] = Field(
        default_factory=list,
        description="低置信度词汇列表"
    )

    # 性能指标
    audio_duration_ms: Optional[int] = Field(None, description="音频时长")
    inference_time_ms: Optional[int] = Field(None, description="推理耗时")
    ger_time_ms: Optional[int] = Field(None, description="GER 纠错耗时")
    rtf: Optional[float] = Field(
        None,
        description="实时率 (inference_time / audio_duration)"
    )


class ASRStreamingChunk(BaseModel):
    """流式 ASR 结果块

    每个音频块的识别结果
    """
    session_id: str
    chunk_id: int = Field(..., description="块序号")

    # 识别结果
    text: str = Field("", description="当前块的识别文本")
    accumulated_text: str = Field("", description="累积的完整文本")

    # 状态
    is_partial: bool = Field(True, description="是否部分结果（可能会被修正）")
    is_final: bool = Field(False, description="是否最终结果（会话结束）")
    is_endpoint: bool = Field(False, description="是否检测到语音端点（句子结束）")

    # 置信度
    confidence: float = Field(0.0, ge=0, le=1)

    # 时间戳
    timestamp_ms: int = Field(..., description="相对于会话开始的时间戳")

    # VAD 信息
    is_speech: bool = Field(True, description="当前块是否包含语音")
    speech_probability: float = Field(1.0, description="语音概率")


class VADResult(BaseModel):
    """VAD 检测结果"""
    is_speech: bool = Field(..., description="是否包含语音")
    speech_probability: float = Field(..., ge=0, le=1)
    start_ms: Optional[int] = Field(None, description="语音开始时间")
    end_ms: Optional[int] = Field(None, description="语音结束时间")


class GERResult(BaseModel):
    """GER 纠错结果"""
    original_text: str = Field(..., description="原始文本")
    corrected_text: str = Field(..., description="纠正后文本")
    corrections: List[dict] = Field(
        default_factory=list,
        description="纠正详情 [{'original': 'xxx', 'corrected': 'yyy', 'position': 0}]"
    )
    confidence_improvement: float = Field(0.0, description="置信度提升")
