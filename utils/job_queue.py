"""
Hàng đợi job bất đồng bộ trong process: nhận việc ngay, xử lý lần lượt bằng `workers` worker asyncio.

Trạng thái lưu trong RAM của process API (chạy 1 worker uvicorn) nên mất khi restart container.
Job đã xong được giữ `ttl` giây để client lấy kết quả rồi xóa.
"""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from fastapi import HTTPException

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


@dataclass
class Job:
    id: str
    filename: str
    run: Callable[[], Awaitable[object]] | None  # bỏ đi khi xong để nhả nội dung file
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: object = None
    error: dict | None = None  # {"status_code": ..., "detail": ...}
    done: asyncio.Event = field(default_factory=asyncio.Event)


class QueueFull(Exception):
    pass


class JobQueue:
    def __init__(self, workers: int = 1, max_queued: int = 20, ttl: float = 86400):
        self.workers = workers
        self.max_queued = max_queued
        self.ttl = ttl
        self._jobs: dict[str, Job] = {}  # giữ thứ tự thêm vào -> tính được vị trí trong hàng
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    def start(self):
        self._tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def submit(self, filename: str, run: Callable[[], Awaitable[object]]) -> Job:
        self._prune()
        if self.queued_count() >= self.max_queued:
            raise QueueFull
        job = Job(id=uuid.uuid4().hex, filename=filename, run=run)
        self._jobs[job.id] = job
        self._queue.put_nowait(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, status: str | None = None) -> list[Job]:
        """Các job còn lưu, mới nhất trước."""
        self._prune()
        return [j for j in reversed(self._jobs.values()) if status is None or j.status == status]

    def queued_count(self) -> int:
        return sum(j.status == QUEUED for j in self._jobs.values())

    def is_busy(self) -> bool:
        """Không còn worker rảnh: job mới sẽ phải chờ."""
        active = sum(j.status in (QUEUED, RUNNING) for j in self._jobs.values())
        return active >= self.workers

    def position(self, job: Job) -> int | None:
        """Số job đang chờ trước job này (0 = chạy kế tiếp); None nếu không còn chờ."""
        if job.status != QUEUED:
            return None
        ahead = 0
        for j in self._jobs.values():
            if j is job:
                return ahead
            ahead += j.status == QUEUED
        return None

    def _prune(self):
        now = time.time()
        expired = [j.id for j in self._jobs.values() if j.finished_at and now - j.finished_at > self.ttl]
        for job_id in expired:
            del self._jobs[job_id]

    async def _worker(self):
        while True:
            job = await self._queue.get()
            job.status, job.started_at = RUNNING, time.time()
            try:
                job.result = await job.run()
                job.status = DONE
            except HTTPException as e:
                job.status, job.error = FAILED, {"status_code": e.status_code, "detail": e.detail}
            except Exception as e:  # không để 1 job lỗi làm chết worker
                job.status, job.error = FAILED, {"status_code": 500, "detail": f"Lỗi không mong đợi: {e!r}"}
            finally:
                job.run, job.finished_at = None, time.time()
                job.done.set()
                self._queue.task_done()
