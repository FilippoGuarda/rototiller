#!/usr/bin/env python3
import csv
import os
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskLogRecord:
    timestamp: float
    run_id: str
    task_id: str
    robot_id: str
    event: str          # e.g. "ASSIGNED", "COMPLETED", "CANCELLED"
    status: str         # e.g. "OK", "FAILED"
    allocation_cost: Optional[float]
    duration: Optional[float]
    path: str
    collision_flag: int # 0 / 1
    message: str


class TaskLogger:
    def __init__(self, csv_path: str):
        self._csv_path = csv_path
        self._lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self):
        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
        if not os.path.exists(self._csv_path) or os.path.getsize(self._csv_path) == 0:
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "run_id",
                    "task_id",
                    "robot_id",
                    "event",
                    "status",
                    "allocation_cost",
                    "duration",
                    "path",
                    "collision_flag",
                    "message",
                ])

    def log(self, rec: TaskLogRecord):
        with self._lock:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{rec.timestamp:.3f}",
                    rec.run_id,
                    rec.task_id,
                    rec.robot_id,
                    rec.event,
                    rec.status,
                    "" if rec.allocation_cost is None else f"{rec.allocation_cost:.3f}",
                    "" if rec.duration is None else f"{rec.duration:.3f}",
                    rec.path,
                    int(rec.collision_flag),
                    rec.message,
                ])