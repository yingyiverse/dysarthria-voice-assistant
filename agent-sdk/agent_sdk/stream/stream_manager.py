"""
Agent SDK - Redis Stream 管理器
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Callable, Optional, Dict, Any
from datetime import datetime

import redis.asyncio as aioredis
from redis.asyncio import Redis

from .config import RedisConfig

logger = logging.getLogger(__name__)


class StreamManager:
    """Redis Stream 管理器

    负责：
    - 消息发布
    - 消息消费
    - 消费者组管理
    - Stream 生命周期管理
    """

    def __init__(self, config: RedisConfig):
        self.config = config
        self._redis: Optional[Redis] = None
        self._consumer_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self):
        """建立 Redis 连接"""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.config.url,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                decode_responses=True
            )
            logger.info(f"Connected to Redis: {self.config.url}")

    async def disconnect(self):
        """断开 Redis 连接"""
        # 取消所有消费者任务
        for task in self._consumer_tasks.values():
            task.cancel()
        self._consumer_tasks.clear()

        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis")

    @property
    def redis(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._redis

    async def publish(
        self,
        stream: str,
        message: dict,
        max_len: Optional[int] = None
    ) -> str:
        """发布消息到 Stream

        Args:
            stream: Stream 名称
            message: 消息内容（dict）
            max_len: 最大长度限制

        Returns:
            消息 ID
        """
        # 序列化嵌套的 dict/list
        serialized = {}
        for key, value in message.items():
            if isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = str(value) if value is not None else ""

        message_id = await self.redis.xadd(
            stream,
            serialized,
            maxlen=max_len or self.config.max_stream_length
        )

        logger.debug(f"Published message to {stream}: {message_id}")
        return message_id

    async def subscribe(
        self,
        stream: str,
        last_id: str = "0"
    ) -> AsyncIterator[Dict[str, Any]]:
        """订阅 Stream（简单读取）

        Args:
            stream: Stream 名称
            last_id: 起始 ID

        Yields:
            消息内容
        """
        current_id = last_id

        while True:
            try:
                messages = await self.redis.xread(
                    {stream: current_id},
                    count=1,
                    block=self.config.block_timeout_ms
                )

                if messages:
                    for stream_name, message_list in messages:
                        for message_id, message in message_list:
                            current_id = message_id
                            yield self._deserialize_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading from stream {stream}: {e}")
                await asyncio.sleep(1)

    async def create_consumer_group(self, stream: str, group: Optional[str] = None):
        """创建消费者组"""
        group = group or self.config.consumer_group

        try:
            await self.redis.xgroup_create(
                stream,
                group,
                id="0",
                mkstream=True
            )
            logger.info(f"Created consumer group {group} for stream {stream}")
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group {group} already exists")
            else:
                raise

    async def consume(
        self,
        stream: str,
        consumer_name: str,
        handler: Callable[[Dict[str, Any]], Any],
        group: Optional[str] = None,
        batch_size: int = 1
    ):
        """消费 Stream 消息（Worker 使用）

        Args:
            stream: Stream 名称
            consumer_name: 消费者名称
            handler: 消息处理函数（async）
            group: 消费者组名称
            batch_size: 批量大小
        """
        group = group or self.config.consumer_group

        # 确保消费者组存在
        await self.create_consumer_group(stream, group)

        logger.info(f"Starting consumer {consumer_name} for {stream}")

        while True:
            try:
                # 读取消息
                messages = await self.redis.xreadgroup(
                    group,
                    consumer_name,
                    {stream: ">"},
                    count=batch_size,
                    block=self.config.block_timeout_ms
                )

                if not messages:
                    continue

                for stream_name, message_list in messages:
                    for message_id, message in message_list:
                        try:
                            # 反序列化并处理
                            data = self._deserialize_message(message)
                            data["_message_id"] = message_id

                            # 调用处理函数
                            if asyncio.iscoroutinefunction(handler):
                                await handler(data)
                            else:
                                handler(data)

                            # ACK 消息
                            await self.redis.xack(stream, group, message_id)
                            logger.debug(f"Processed and ACKed message {message_id}")

                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            # 不 ACK，消息会被重新投递

            except asyncio.CancelledError:
                logger.info(f"Consumer {consumer_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(1)

    async def start_consumer(
        self,
        stream: str,
        consumer_name: str,
        handler: Callable,
        group: Optional[str] = None
    ) -> asyncio.Task:
        """启动后台消费者任务

        Returns:
            消费者任务
        """
        task_name = f"{stream}:{consumer_name}"

        if task_name in self._consumer_tasks:
            logger.warning(f"Consumer {task_name} already running")
            return self._consumer_tasks[task_name]

        task = asyncio.create_task(
            self.consume(stream, consumer_name, handler, group)
        )
        self._consumer_tasks[task_name] = task
        return task

    async def stop_consumer(self, stream: str, consumer_name: str):
        """停止消费者任务"""
        task_name = f"{stream}:{consumer_name}"

        if task_name in self._consumer_tasks:
            self._consumer_tasks[task_name].cancel()
            del self._consumer_tasks[task_name]
            logger.info(f"Stopped consumer {task_name}")

    async def get_result(
        self,
        result_stream: str,
        timeout_ms: int = 30000
    ) -> Optional[Dict[str, Any]]:
        """获取任务结果（等待结果）

        Args:
            result_stream: 结果 Stream 名称
            timeout_ms: 超时时间

        Returns:
            结果数据或 None（超时）
        """
        start_time = asyncio.get_event_loop().time()
        timeout_seconds = timeout_ms / 1000

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout_seconds:
                return None

            remaining_ms = int((timeout_seconds - elapsed) * 1000)

            messages = await self.redis.xread(
                {result_stream: "0"},
                count=1,
                block=min(remaining_ms, 5000)
            )

            if messages:
                for _, message_list in messages:
                    for _, message in message_list:
                        return self._deserialize_message(message)

        return None

    async def publish_result(
        self,
        task_id: str,
        result: dict,
        ttl_seconds: int = 300
    ):
        """发布任务结果

        Args:
            task_id: 任务 ID
            result: 结果数据
            ttl_seconds: 结果保留时间
        """
        result_stream = f"results:{task_id}"

        await self.publish(result_stream, result, max_len=1)

        # 设置过期时间
        await self.redis.expire(result_stream, ttl_seconds)

    def _deserialize_message(self, message: Dict[str, str]) -> Dict[str, Any]:
        """反序列化消息"""
        result = {}
        for key, value in message.items():
            if value.startswith("{") or value.startswith("["):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value
            else:
                result[key] = value
        return result

    # ==================== 辅助方法 ====================

    async def get_stream_length(self, stream: str) -> int:
        """获取 Stream 长度"""
        return await self.redis.xlen(stream)

    async def get_pending_count(
        self,
        stream: str,
        group: Optional[str] = None
    ) -> int:
        """获取待处理消息数量"""
        group = group or self.config.consumer_group

        try:
            info = await self.redis.xpending(stream, group)
            return info["pending"] if info else 0
        except Exception:
            return 0

    async def delete_stream(self, stream: str):
        """删除 Stream"""
        await self.redis.delete(stream)
        logger.info(f"Deleted stream {stream}")
