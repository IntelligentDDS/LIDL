from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from threading import BoundedSemaphore, Lock

import litellm
from loguru import logger

litellm.suppress_debug_info = True
litellm.success_callback = []
litellm.failure_callback = []

file_dir = os.path.dirname(os.path.abspath(__file__))
global_log_dir = f"{file_dir}/logs"
os.makedirs(global_log_dir, exist_ok=True)


def custom_logger(log_file = f"{global_log_dir}/all.log", level = "INFO"):
    """Configure colorized console and rotating file logging."""
    logger.remove()
    logger.add(
        sink=lambda message: print(message, end=""),
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )
    logger.add(
        Path(log_file),
        level=level,
        rotation="1 day",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] ({name}:{function}:{line}) {message}",
    )
    return logger


global_logger = custom_logger()
_log_locks = {}
_log_locks_guard = Lock()
_llm_semaphore = BoundedSemaphore(4)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temp, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve())
    with _log_locks_guard:
        lock = _log_locks.setdefault(key, Lock())
    with lock, path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")
        file.flush()


def trace(case, stage, event, **details):
    append_jsonl(case / "trace.jsonl", {"timestamp": now(), "stage": stage, "event": event, **details})


def log_case(case, level, message):
    with (case / "run.log").open("a", encoding="utf-8") as file:
        file.write(f"{now()} [{level}] {message}\n")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_relative(path_value):
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] == ".git":
        raise ValueError(f"Unsafe repository-relative path: {path_value}")
    return path


def run_parallel(items, worker, max_workers, item_name):
    if not items:
        return {}
    failures = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        futures = {executor.submit(worker, item): item_name(item) for item in items}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures[futures[future]] = repr(error)
                logger.error(f"{futures[future]} failed: {error}")
    return failures


def configure_llm_concurrency(max_workers):
    global _llm_semaphore
    if max_workers < 1:
        raise ValueError("GPT_MAX_WORKERS and --llm-workers must be at least 1")
    _llm_semaphore = BoundedSemaphore(max_workers)


def limited_completion(completion, *args, **kwargs):
    """Hold the global LLM slot until a complete response or stream is consumed."""
    _llm_semaphore.acquire()
    try:
        response = completion(*args, **kwargs)
    except Exception:
        _llm_semaphore.release()
        raise
    if hasattr(response, "model_dump"):
        _llm_semaphore.release()
        return response

    def limited_stream():
        try:
            yield from response
        finally:
            _llm_semaphore.release()

    return limited_stream()


def llm_settings(args):
    if not args.llm_name or not args.llm_base_url or not args.llm_api_key:
        raise ValueError("Missing LLM name, base URL, or API key")
    return args.llm_name, args.llm_base_url, args.llm_api_key


def call_llm_limited(*args, **kwargs):
    with _llm_semaphore:
        return call_llm(*args, **kwargs)


def truncate(text, max_tokens = 32768, reserved_tokens = 1000):
    """Truncate text using a four-characters-per-token estimate."""
    available_tokens = max_tokens - reserved_tokens
    max_chars = available_tokens * 4
    if len(text) <= max_chars:
        return text
    global_logger.warning(f"Text has {len(text)} chars and exceeds the {max_chars}-char limit; truncating")
    return text[:max_chars] + "...[truncated]"


def log_api(log_file, data):
    """Append one complete API record to a JSONL file."""
    if callable(log_file):
        log_file({"time": datetime.now().isoformat(), "data": data})
        return
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(log_path.resolve())
    with _log_locks_guard:
        lock = _log_locks.setdefault(key, Lock())
    record = {"time": datetime.now().isoformat(), "data": data}
    with lock, log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_format_instructions(pydantic_object):
    """Build concise JSON-only instructions from a Pydantic model schema."""
    if pydantic_object is None:
        return "Return only one valid JSON object without Markdown fences or additional text."
    if hasattr(pydantic_object, "model_json_schema"):
        schema = pydantic_object.model_json_schema()
    else:
        schema = pydantic_object.schema()
    return (
        "Return only one valid JSON object without Markdown fences or additional text. "
        f"The object must conform to this JSON schema: {json.dumps(schema, ensure_ascii=False)}"
    )


def _litellm_model(model, base_url):
    """Route explicit providers natively and compatible proxy aliases through OpenAI."""
    if not model:
        raise ValueError("LLM model is required")
    providers = {
        "anthropic",
        "azure",
        "bedrock",
        "cohere",
        "gemini",
        "groq",
        "mistral",
        "ollama",
        "openai",
        "together_ai",
        "vertex_ai",
    }
    prefix = model.split("/", 1)[0] if "/" in model else ""
    if prefix in providers:
        return model
    if "api.anthropic.com" in (base_url or ""):
        return f"anthropic/{model}"
    return f"openai/{model}"


def _response_dump(response):
    """Convert a LiteLLM response to a JSON-serializable dictionary without dropping fields."""
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "dict"):
        return response.dict()
    raise TypeError(f"Unsupported LiteLLM response type: {type(response).__name__}")


def _extract_content(raw_response):
    """Read text content from a complete LiteLLM response dictionary."""
    choices = raw_response.get("choices") or []
    if not choices:
        raise ValueError("LiteLLM response has no choices")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _parse_json(content):
    """Parse a JSON response, accepting fences, prose prefixes, and one missing closing delimiter."""
    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as direct_error:
        decoder = json.JSONDecoder()
        decoded = []
        repaired = []
        if (
            stripped.startswith("{")
            and stripped.endswith("}")
            and stripped.count("{") == stripped.count("}")
            and stripped.count("[") == stripped.count("]") + 1
        ):
            try:
                value = json.loads(stripped[:-1] + "]}")
                global_logger.warning("LLM JSON response closed an array with a brace; repaired deterministically")
                return value
            except json.JSONDecodeError:
                pass
        for index, character in enumerate(stripped):
            if character not in "{[":
                continue
            candidate = stripped[index:]
            try:
                value, end = decoder.raw_decode(candidate)
                decoded.append((end, index, False, value))
            except json.JSONDecodeError:
                opening = candidate[0]
                closing = "}" if opening == "{" else "]"
                if candidate.count(opening) == candidate.count(closing) + 1:
                    try:
                        value, end = decoder.raw_decode(candidate + closing)
                        repaired.append((min(end, len(candidate)), index, True, value))
                    except json.JSONDecodeError:
                        pass
        candidates = [*decoded, *repaired]
        if candidates:
            selected = max(
                candidates,
                key=lambda item: (
                    isinstance(item[3], dict),
                    isinstance(item[3], list),
                    item[0],
                    -item[1],
                ),
            )
            if selected[2]:
                global_logger.warning("LLM JSON response missed one closing delimiter; repaired deterministically")
            return selected[3]
        raise ValueError(f"LLM returned invalid JSON: {direct_error}; content={content}") from direct_error


def _validate_result(data, pydantic_object):
    """Validate structured output and return a plain dictionary."""
    if pydantic_object is not None:
        if hasattr(pydantic_object, "model_validate"):
            return pydantic_object.model_validate(data).model_dump()
        return pydantic_object.parse_obj(data).dict()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object, got {type(data).__name__}")
    return data


_RETRYABLE_KEYWORDS = (
    "timeout",
    "timed out",
    "rate limit",
    "rate_limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "connection",
    "service unavailable",
    "internal server error",
)


def _retry_delay(error, attempt):
    """Calculate bounded exponential backoff and honor integer retry-after hints."""
    match = re.search(r"retry after (\d+)", str(error).lower())
    if match:
        return min(int(match.group(1)) + 1, 120)
    return min(5 * (2**attempt) + random.uniform(0, 1), 60)


def call_llm(
    messages,
    model,
    base_url,
    api_key,
    pydantic_object = None,
    log_file = "calls_llm.jsonl",
    **kwargs,
):
    """Call an LLM through LiteLLM and return validated structured output."""
    max_retries = int(kwargs.pop("max_retries", 5))
    log_input = kwargs.pop("log_input", True)
    log_metadata = kwargs.pop("log_metadata", None)
    kwargs.setdefault("timeout", 600)
    kwargs.setdefault("drop_params", True)
    model_name = _litellm_model(model, base_url)

    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=model_name,
                messages=messages,
                api_base=base_url,
                api_key=api_key,
                **kwargs,
            )
            raw_response = _response_dump(response)
        except Exception as error:
            record = {
                "error": str(error),
                "error_type": type(error).__name__,
                "attempt": attempt + 1,
            }
            if log_input:
                record["input"] = messages
            if log_metadata is not None:
                record["metadata"] = log_metadata
            log_api(log_file, record)
            retryable = any(keyword in str(error).lower() for keyword in _RETRYABLE_KEYWORDS)
            if retryable and attempt < max_retries - 1:
                wait = _retry_delay(error, attempt)
                global_logger.warning(f"LLM call retry {attempt + 1}/{max_retries} after {wait:.1f}s: {error}")
                time.sleep(wait)
                continue
            global_logger.error(f"LLM call failed: {error}")
            raise

        try:
            result = _validate_result(_parse_json(_extract_content(raw_response)), pydantic_object)
        except Exception as error:
            record = {
                "output": raw_response,
                "error": str(error),
                "error_type": type(error).__name__,
                "attempt": attempt + 1,
            }
            if log_input:
                record["input"] = messages
            if log_metadata is not None:
                record["metadata"] = log_metadata
            log_api(log_file, record)
            if attempt < max_retries - 1:
                wait = _retry_delay(error, attempt)
                global_logger.warning(
                    f"LLM response validation retry {attempt + 1}/{max_retries} after {wait:.1f}s: {error}"
                )
                time.sleep(wait)
                continue
            global_logger.error(f"LLM response validation failed after {max_retries} attempts: {error}")
            raise
        record = {"output": raw_response}
        if log_input:
            record["input"] = messages
        if log_metadata is not None:
            record["metadata"] = log_metadata
        log_api(log_file, record)
        global_logger.info(f"LLM call succeeded: model={model}, usage={raw_response.get('usage')}")
        return result

    raise RuntimeError("LLM call exhausted retries")


def call_embedding(
    text,
    model,
    base_url,
    api_key,
    log_file = "calls_embedding.jsonl",
    **kwargs,
):
    """Call an embedding endpoint through LiteLLM and return the first vector."""
    max_retries = int(kwargs.pop("max_retries", 5))
    kwargs.setdefault("timeout", 600)
    model_name = _litellm_model(model, base_url)

    for attempt in range(max_retries):
        try:
            response = litellm.embedding(
                model=model_name,
                input=[text],
                api_base=base_url,
                api_key=api_key,
                **kwargs,
            )
            raw_response = _response_dump(response)
            log_api(log_file, {"input": text, "output": raw_response})
            data = raw_response.get("data") or []
            if not data or "embedding" not in data[0]:
                raise ValueError("LiteLLM embedding response has no vector")
            global_logger.info(f"Embedding call succeeded: model={model}, usage={raw_response.get('usage')}")
            return data[0]["embedding"]
        except Exception as error:
            log_api(
                log_file,
                {"input": text, "error": str(error), "error_type": type(error).__name__, "attempt": attempt + 1},
            )
            retryable = any(keyword in str(error).lower() for keyword in _RETRYABLE_KEYWORDS)
            if retryable and attempt < max_retries - 1:
                wait = _retry_delay(error, attempt)
                global_logger.warning(f"Embedding retry {attempt + 1}/{max_retries} after {wait:.1f}s: {error}")
                time.sleep(wait)
                continue
            global_logger.error(f"Embedding call failed: {error}")
            raise

    raise RuntimeError("Embedding call exhausted retries")
