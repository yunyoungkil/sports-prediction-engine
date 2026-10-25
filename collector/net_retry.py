"""
collector/net_retry.py
======================
베트맨 서버가 가끔 연결을 끊는 문제(ConnectionResetError, WinError 10054, Errno 104)에 대비한 재시도 헬퍼.
"""

import time
import urllib.request
from typing import Any, Optional


def open_with_retry(opener: urllib.request.OpenerDirector, req: urllib.request.Request, timeout: float = 10,
                    attempts: int = 3, base_wait: float = 2.0, label: str = "") -> Any:
    """opener.open(req)를 최대 attempts번 시도. 대기 시간은 2초, 5초, 8초 … 로 늘린다. 마지막 실패는 예외로 올린다."""
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return opener.open(req, timeout=timeout)
        except Exception as e:      # 연결 끊김·타임아웃 등
            last = e
            if i < attempts - 1:
                wait = base_wait + 3 * i
                if label:
                    print(f"  ↻ {label} 재시도 {i + 1}/{attempts - 1} ({wait:.0f}초 후): {e}")
                time.sleep(wait)
    raise last
