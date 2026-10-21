import fcntl
import os
import time
from pathlib import Path
from typing import Dict, Union

import anthropic
import openai
import tiktoken

_llm_call_count = 0


def _max_llm_calls():
    max_llm_calls = int(os.getenv("MAX_LLM_CALLS", "50"))
    if max_llm_calls < 1:
        raise ValueError("MAX_LLM_CALLS must be at least 1")
    return max_llm_calls


class LLMCallLimitExceeded(RuntimeError):
    pass


def _count_llm_call():
    global _llm_call_count
    max_llm_calls = _max_llm_calls()
    count_file = os.getenv("LLM_CALL_COUNT_FILE")
    if not count_file:
        if _llm_call_count >= max_llm_calls:
            raise LLMCallLimitExceeded(f"Maximum LLM call limit reached: {max_llm_calls}")
        _llm_call_count += 1
        return

    path = Path(count_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        raw_count = file.read().strip()
        count = int(raw_count) if raw_count else 0
        if count >= max_llm_calls:
            path.with_name("llm_call_limit_reached").touch()
            raise LLMCallLimitExceeded(f"Maximum LLM call limit reached: {max_llm_calls}")
        file.seek(0)
        file.truncate()
        file.write(str(count + 1))
        file.flush()


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url="https://openkey.cloud/v1/", max_retries=40, timeout=100):
    ret = None
    retries = 0
    client = openai.OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")  # or hardcode it here if needed
    )

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")
            _count_llm_call()
            ret = client.chat.completions.create(**config)

        except LLMCallLimitExceeded:
            raise
        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


def request_anthropic_engine(
    config, logger, max_retries=40, timeout=500, prompt_cache=False
):
    ret = None
    retries = 0

    #client = anthropic.Anthropic()
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Override the base URL for proxy
    client._client.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")  # default = official

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            _count_llm_call()
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                config["messages"][0]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except LLMCallLimitExceeded:
            raise
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10 * retries)
        retries += 1

    return ret
