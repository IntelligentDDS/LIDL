import argparse
import ast
import subprocess
import inspect
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description="Exercise real ChatAgent.step with a deterministic model response and no network")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--field", choices=("refusal", "audio"), required=True)
    parser.add_argument("--check-baseline", action="store_true")
    args = parser.parse_args()
    if args.check_baseline:
        original = subprocess.run(["git", "show", "HEAD:camel/messages/chat_messages.py"], cwd=args.repo, capture_output=True, text=True, check=True).stdout
        tree = ast.parse(original)
        message_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChatMessage")
        if any(isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == args.field for node in message_class.body):
            print("LIDL_BASELINE_ALREADY_ACCEPTS_FIELD", file=sys.stderr)
            return 2
    sys.path.insert(0, str(args.repo.resolve()))
    os.environ["OPENAI_API_KEY"] = "offline-test-placeholder"

    def deny_network(*unused_args, **unused_kwargs):
        raise RuntimeError("External network is disabled for this response-conversion test")

    with patch.object(socket.socket, "connect", deny_network):
        from camel.agents.chat_agent import ChatAgent, ChatCompletion
        from camel.messages import ChatMessage, SystemMessage
        from camel.typing import RoleType

        class ResponseBackend:
            def __init__(self, payload):
                self.payload = payload
                self.calls = 0

            def run(self, **kwargs):
                self.calls += 1
                choice = SimpleNamespace(message=self.payload, finish_reason="stop")
                return ChatCompletion.model_construct(id="offline-response", choices=[choice], usage={})

        results = []
        for extra_field in (False, True):
            payload = {"role": "assistant", "content": "offline response"}
            if extra_field:
                payload[args.field] = None
            backend = ResponseBackend(payload)
            try:
                with patch("camel.agents.chat_agent.ModelFactory.create", return_value=backend), patch("camel.agents.chat_agent.num_tokens_from_messages", return_value=10):
                    system = SystemMessage(role_name="Assistant", role_type=RoleType.ASSISTANT, content="Test response conversion")
                    agent = ChatAgent(system)
                    message = ChatMessage(role_name="User", role_type=RoleType.USER, meta_dict={}, role="user", content="Test")
                    response = inspect.unwrap(ChatAgent.step)(agent, message)
                    assert backend.calls == 1, "Real response conversion was not reached"
                    assert response.msgs and response.msgs[0].content == payload["content"], "Response content was lost"
                results.append({"extra_field": extra_field, "passed": True, "backend_calls": backend.calls})
            except Exception as error:
                print(f"Response conversion failed: {type(error).__name__}: {error}", file=sys.stderr)
                results.append({"extra_field": extra_field, "passed": False, "backend_calls": backend.calls, "error_type": type(error).__name__, "error": str(error)})
        print(json.dumps({"field": args.field, "cases": results}))
        return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
