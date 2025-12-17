"""
Agent SDK - Worker 池管理器
"""

import asyncio
import logging
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime, timedelta
from enum import Enum

from ..config import WorkerPoolConfig

# 导入共享模型
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from shared.models import WorkerInfo, WorkerStatus, HealthCheckResponse

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(str, Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"


class WorkerPool:
    """Worker 池管理器

    负责：
    - Worker 注册与注销
    - 健康检查
    - 负载均衡
    - Worker 状态跟踪
    """

    def __init__(self, config: WorkerPoolConfig):
        self.config = config

        # Worker 注册表: {worker_id: WorkerInfo}
        self._workers: Dict[str, WorkerInfo] = {}

        # 按类型分组: {worker_type: [worker_id, ...]}
        self._workers_by_type: Dict[str, List[str]] = {}

        # 健康检查任务
        self._health_check_task: Optional[asyncio.Task] = None

        # 重启计数: {worker_id: count}
        self._restart_counts: Dict[str, int] = {}

        # Round Robin 索引
        self._rr_index: Dict[str, int] = {}

    async def start(self):
        """启动 Worker 池管理"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Worker pool manager started")

    async def stop(self):
        """停止 Worker 池管理"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Worker pool manager stopped")

    async def register_worker(self, worker: WorkerInfo):
        """注册 Worker

        Args:
            worker: Worker 信息
        """
        worker_id = worker.worker_id
        worker_type = worker.worker_type

        # 更新注册表
        self._workers[worker_id] = worker

        # 按类型分组
        if worker_type not in self._workers_by_type:
            self._workers_by_type[worker_type] = []

        if worker_id not in self._workers_by_type[worker_type]:
            self._workers_by_type[worker_type].append(worker_id)

        # 初始化重启计数
        self._restart_counts[worker_id] = 0

        logger.info(f"Registered worker: {worker_id} (type={worker_type})")

    async def unregister_worker(self, worker_id: str):
        """注销 Worker

        Args:
            worker_id: Worker ID
        """
        if worker_id not in self._workers:
            return

        worker = self._workers[worker_id]
        worker_type = worker.worker_type

        # 从注册表移除
        del self._workers[worker_id]

        # 从类型分组移除
        if worker_type in self._workers_by_type:
            if worker_id in self._workers_by_type[worker_type]:
                self._workers_by_type[worker_type].remove(worker_id)

        # 清理重启计数
        if worker_id in self._restart_counts:
            del self._restart_counts[worker_id]

        logger.info(f"Unregistered worker: {worker_id}")

    def get_workers(self, worker_type: Optional[str] = None) -> List[WorkerInfo]:
        """获取 Worker 列表

        Args:
            worker_type: 过滤类型（可选）

        Returns:
            Worker 信息列表
        """
        if worker_type is None:
            return list(self._workers.values())

        worker_ids = self._workers_by_type.get(worker_type, [])
        return [self._workers[wid] for wid in worker_ids if wid in self._workers]

    def get_healthy_workers(self, worker_type: str) -> List[WorkerInfo]:
        """获取健康的 Worker 列表

        Args:
            worker_type: Worker 类型

        Returns:
            健康的 Worker 列表
        """
        workers = self.get_workers(worker_type)
        return [w for w in workers if w.status == WorkerStatus.HEALTHY]

    def select_worker(self, worker_type: str) -> Optional[WorkerInfo]:
        """选择一个 Worker（负载均衡）

        Args:
            worker_type: Worker 类型

        Returns:
            选中的 Worker 或 None
        """
        healthy_workers = self.get_healthy_workers(worker_type)

        if not healthy_workers:
            logger.warning(f"No healthy workers available for type: {worker_type}")
            return None

        strategy = self.config.load_balance_strategy

        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._select_round_robin(worker_type, healthy_workers)
        elif strategy == LoadBalanceStrategy.LEAST_LOADED:
            return self._select_least_loaded(healthy_workers)
        else:  # RANDOM
            return self._select_random(healthy_workers)

    def _select_round_robin(
        self,
        worker_type: str,
        workers: List[WorkerInfo]
    ) -> WorkerInfo:
        """轮询选择"""
        if worker_type not in self._rr_index:
            self._rr_index[worker_type] = 0

        index = self._rr_index[worker_type] % len(workers)
        self._rr_index[worker_type] = index + 1

        return workers[index]

    def _select_least_loaded(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """最少负载选择"""
        return min(workers, key=lambda w: w.current_load)

    def _select_random(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """随机选择"""
        import random
        return random.choice(workers)

    async def update_worker_status(
        self,
        worker_id: str,
        status: WorkerStatus,
        current_load: Optional[int] = None
    ):
        """更新 Worker 状态

        Args:
            worker_id: Worker ID
            status: 新状态
            current_load: 当前负载
        """
        if worker_id not in self._workers:
            return

        worker = self._workers[worker_id]
        worker.status = status
        worker.last_heartbeat = datetime.utcnow()

        if current_load is not None:
            worker.current_load = current_load

        logger.debug(f"Updated worker {worker_id}: status={status}, load={current_load}")

    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _perform_health_checks(self):
        """执行健康检查"""
        now = datetime.utcnow()
        timeout_threshold = timedelta(seconds=self.config.health_check_timeout * 3)

        for worker_id, worker in list(self._workers.items()):
            # 检查心跳超时
            if worker.last_heartbeat:
                time_since_heartbeat = now - worker.last_heartbeat

                if time_since_heartbeat > timeout_threshold:
                    logger.warning(f"Worker {worker_id} heartbeat timeout")
                    await self._handle_unhealthy_worker(worker_id)

    async def _handle_unhealthy_worker(self, worker_id: str):
        """处理不健康的 Worker

        Args:
            worker_id: Worker ID
        """
        if worker_id not in self._workers:
            return

        worker = self._workers[worker_id]
        worker.status = WorkerStatus.UNHEALTHY

        if self.config.auto_restart:
            restart_count = self._restart_counts.get(worker_id, 0)

            if restart_count < self.config.max_restart_attempts:
                self._restart_counts[worker_id] = restart_count + 1
                logger.info(f"Attempting to restart worker {worker_id} "
                           f"(attempt {restart_count + 1}/{self.config.max_restart_attempts})")
                # 实际重启逻辑由外部 Worker 管理器处理
                # 这里只是标记需要重启
                worker.status = WorkerStatus.RESTARTING
            else:
                logger.error(f"Worker {worker_id} exceeded max restart attempts")
                worker.status = WorkerStatus.DEAD

    def get_pool_stats(self) -> Dict[str, Any]:
        """获取 Worker 池统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "total_workers": len(self._workers),
            "by_type": {},
            "by_status": {
                "healthy": 0,
                "unhealthy": 0,
                "busy": 0,
                "restarting": 0,
                "dead": 0
            }
        }

        for worker_type, worker_ids in self._workers_by_type.items():
            healthy = sum(
                1 for wid in worker_ids
                if wid in self._workers and
                self._workers[wid].status == WorkerStatus.HEALTHY
            )
            stats["by_type"][worker_type] = {
                "total": len(worker_ids),
                "healthy": healthy
            }

        for worker in self._workers.values():
            status_key = worker.status.value
            if status_key in stats["by_status"]:
                stats["by_status"][status_key] += 1

        return stats


class WorkerHeartbeatManager:
    """Worker 心跳管理器（Worker 端使用）"""

    def __init__(
        self,
        worker_info: WorkerInfo,
        heartbeat_interval: int = 10,
        on_heartbeat: Optional[Callable] = None
    ):
        self.worker_info = worker_info
        self.heartbeat_interval = heartbeat_interval
        self.on_heartbeat = on_heartbeat

        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """启动心跳"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"Heartbeat started for worker {self.worker_info.worker_id}")

    async def stop(self):
        """停止心跳"""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info(f"Heartbeat stopped for worker {self.worker_info.worker_id}")

    async def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # 更新心跳时间
                self.worker_info.last_heartbeat = datetime.utcnow()

                # 调用心跳回调（如发送心跳到 Redis）
                if self.on_heartbeat:
                    if asyncio.iscoroutinefunction(self.on_heartbeat):
                        await self.on_heartbeat(self.worker_info)
                    else:
                        self.on_heartbeat(self.worker_info)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    def update_load(self, current_load: int):
        """更新当前负载

        Args:
            current_load: 当前处理的任务数
        """
        self.worker_info.current_load = current_load
