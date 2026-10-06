import logging
import os
import random
import time

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL_NAME", "google/gemma-4-26B-A4B-it")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Retry cho lỗi tạm thời (connection reset by peer, 429, 5xx của gateway...).
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "2"))    # giây, nhân đôi sau mỗi lần
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "30"))     # giây, trần thời gian chờ

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# ConnectTimeout là con của ConnectionError nên cũng được retry.
# ReadTimeout không retry: server đã nhận request, chờ thêm `timeout` giây nữa thường vô ích.
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)


class _RetryableHTTPError(Exception):
    def __init__(self, response: requests.Response):
        self.response = response


def _backoff_delay(attempt: int, response: requests.Response | None = None) -> float:
    """Exponential backoff + full jitter; ưu tiên header Retry-After nếu server gửi."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), BACKOFF_MAX)
    return random.uniform(0, min(BACKOFF_MAX, BACKOFF_BASE * 2 ** attempt))


def _post_with_retry(url: str, payload: dict, headers: dict, timeout: int) -> requests.Response:
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code in RETRYABLE_STATUS:
                raise _RetryableHTTPError(response)
            return response
        except (*RETRYABLE_EXCEPTIONS, _RetryableHTTPError) as e:
            if attempt == MAX_RETRIES:
                if isinstance(e, _RetryableHTTPError):
                    return e.response  # để raise_for_status() báo lỗi HTTP như cũ
                raise
            resp = e.response if isinstance(e, _RetryableHTTPError) else None
            delay = _backoff_delay(attempt, resp)
            reason = f"HTTP {resp.status_code}" if resp is not None else repr(e)
            logger.warning("LLM lỗi tạm thời (%s), thử lại lần %d/%d sau %.1fs",
                           reason, attempt + 1, MAX_RETRIES, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def query_llm(prompt: str, model: str = MODEL, timeout: int = 600,
              temperature: float = TEMPERATURE) -> str:
    """
    Gửi 1 prompt tới endpoint OpenAI-compatible (/chat/completions) và lấy kết quả.
    Tự retry với lỗi kết nối / 429 / 5xx (xem LLM_MAX_RETRIES).

    Args:
        prompt: nội dung prompt.
        model: tên model trên server (vd "google/gemma-4-26B-A4B-it").
        timeout: thời gian chờ tối đa (giây) — văn bản dài cần lâu hơn.
        temperature: mặc định 0 để kết quả ổn định, so sánh được giữa các lần sửa prompt.

    Returns:
        Văn bản phản hồi từ model.
    """
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    try:
        response = _post_with_retry(url, payload, headers, timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Không kết nối được tới LLM endpoint sau {MAX_RETRIES + 1} lần thử: {BASE_URL}") from e
    except requests.exceptions.ChunkedEncodingError as e:
        raise RuntimeError(
            f"Kết nối tới LLM bị ngắt giữa chừng sau {MAX_RETRIES + 1} lần thử.") from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError("LLM phản hồi quá lâu (timeout). Hãy tăng timeout.") from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"LLM endpoint lỗi {response.status_code}: {response.text[:500]}") from e

    data = response.json()
    return (data["choices"][0]["message"]["content"] or "").strip()
