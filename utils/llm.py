import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL_NAME", "google/gemma-4-26B-A4B-it")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))


def query_llm(prompt: str, model: str = MODEL, timeout: int = 600,
              temperature: float = TEMPERATURE) -> str:
    """
    Gửi 1 prompt tới endpoint OpenAI-compatible (/chat/completions) và lấy kết quả.

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
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Không kết nối được tới LLM endpoint: {BASE_URL}") from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError("LLM phản hồi quá lâu (timeout). Hãy tăng timeout.") from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"LLM endpoint lỗi {response.status_code}: {response.text[:500]}") from e

    data = response.json()
    return (data["choices"][0]["message"]["content"] or "").strip()
