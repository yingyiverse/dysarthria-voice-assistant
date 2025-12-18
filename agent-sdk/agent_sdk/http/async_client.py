"""
Agent SDK - 异步 HTTP 客户端
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime

import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from ..config import HTTPClientConfig

logger = logging.getLogger(__name__)


class AsyncHTTPClient:
    """异步 HTTP 客户端

    负责：
    - 外部 API 调用
    - 连接池管理
    - 重试机制
    - 超时处理
    """

    def __init__(self, config: HTTPClientConfig):
        self.config = config
        self._session: Optional[ClientSession] = None

    async def connect(self):
        """建立连接池"""
        if self._session is None:
            connector = TCPConnector(
                limit=self.config.max_connections,
                keepalive_timeout=self.config.keepalive_timeout
            )
            timeout = ClientTimeout(total=self.config.timeout)

            self._session = ClientSession(
                connector=connector,
                timeout=timeout
            )
            logger.info("HTTP client connected")

    async def disconnect(self):
        """关闭连接池"""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("HTTP client disconnected")

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("HTTP client not connected. Call connect() first.")
        return self._session

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Union[bytes, str]] = None,
        timeout: Optional[int] = None,
        retry: bool = True
    ) -> "HTTPResponse":
        """发送 HTTP 请求

        Args:
            method: HTTP 方法
            url: 请求 URL
            headers: 请求头
            params: URL 参数
            json: JSON 数据
            data: 原始数据
            timeout: 超时时间（秒）
            retry: 是否启用重试

        Returns:
            HTTP 响应
        """
        request_timeout = ClientTimeout(total=timeout) if timeout else None
        max_retries = self.config.max_retries if retry else 1
        last_error = None

        for attempt in range(max_retries):
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    data=data,
                    timeout=request_timeout
                ) as response:
                    body = await response.read()
                    return HTTPResponse(
                        status=response.status,
                        headers=dict(response.headers),
                        body=body
                    )

            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"HTTP request failed (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                    await asyncio.sleep(delay)

        raise HTTPRequestError(f"Request failed after {max_retries} attempts: {last_error}")

    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "HTTPResponse":
        """GET 请求"""
        return await self.request("GET", url, headers=headers, params=params, **kwargs)

    async def post(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Union[bytes, str]] = None,
        **kwargs
    ) -> "HTTPResponse":
        """POST 请求"""
        return await self.request("POST", url, headers=headers, json=json, data=data, **kwargs)

    async def put(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "HTTPResponse":
        """PUT 请求"""
        return await self.request("PUT", url, headers=headers, json=json, **kwargs)

    async def delete(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> "HTTPResponse":
        """DELETE 请求"""
        return await self.request("DELETE", url, headers=headers, **kwargs)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


class HTTPResponse:
    """HTTP 响应封装"""

    def __init__(
        self,
        status: int,
        headers: Dict[str, str],
        body: bytes
    ):
        self.status = status
        self.headers = headers
        self._body = body

    @property
    def body(self) -> bytes:
        """原始响应体"""
        return self._body

    @property
    def text(self) -> str:
        """文本响应"""
        return self._body.decode("utf-8")

    def json(self) -> Any:
        """JSON 响应"""
        import json
        return json.loads(self._body)

    @property
    def ok(self) -> bool:
        """请求是否成功"""
        return 200 <= self.status < 300


class HTTPRequestError(Exception):
    """HTTP 请求错误"""
    pass


class ExternalAPIClient:
    """外部 API 客户端（封装常用的外部服务调用）"""

    def __init__(self, http_client: AsyncHTTPClient):
        self.http = http_client

    async def call_tts_api(
        self,
        text: str,
        voice: str = "default",
        api_url: str = "http://localhost:8000/tts"
    ) -> bytes:
        """调用 TTS API

        Args:
            text: 要转换的文本
            voice: 语音类型
            api_url: TTS 服务地址

        Returns:
            音频数据（bytes）
        """
        response = await self.http.post(
            url=api_url,
            json={"text": text, "voice": voice}
        )

        if not response.ok:
            raise HTTPRequestError(f"TTS API error: {response.status}")

        return response.body

    async def call_llm_api(
        self,
        messages: list,
        model: str = "gpt-4",
        api_url: str = "https://api.openai.com/v1/chat/completions",
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """调用 LLM API

        Args:
            messages: 消息列表
            model: 模型名称
            api_url: API 地址
            api_key: API Key

        Returns:
            LLM 响应
        """
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = await self.http.post(
            url=api_url,
            headers=headers,
            json={
                "model": model,
                "messages": messages
            }
        )

        if not response.ok:
            raise HTTPRequestError(f"LLM API error: {response.status}")

        return response.json()

    async def call_embedding_api(
        self,
        text: str,
        model: str = "text-embedding-ada-002",
        api_url: str = "https://api.openai.com/v1/embeddings",
        api_key: Optional[str] = None
    ) -> list:
        """调用 Embedding API

        Args:
            text: 要嵌入的文本
            model: 模型名称
            api_url: API 地址
            api_key: API Key

        Returns:
            嵌入向量
        """
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = await self.http.post(
            url=api_url,
            headers=headers,
            json={
                "model": model,
                "input": text
            }
        )

        if not response.ok:
            raise HTTPRequestError(f"Embedding API error: {response.status}")

        data = response.json()
        return data["data"][0]["embedding"]
