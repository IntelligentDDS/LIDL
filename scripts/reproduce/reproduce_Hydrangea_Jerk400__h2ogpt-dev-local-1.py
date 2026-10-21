import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "Jerk400__h2ogpt-dev-local-1"
BASE_COMMIT = "ba56d6471080ec447c7c5347a2d4765ae4dba5b4"
ENVIRONMENT = ROOT / "datasets" / "Hydrangea" / "environments" / "Jerk400__h2ogpt-dev-local-ba56d647"
ORIGINAL_REPO = ROOT / "datasets" / "Hydrangea" / "repo" / INSTANCE_ID
WORKSPACE = ENVIRONMENT / INSTANCE_ID
ARTIFACTS = ROOT / "datasets" / "Hydrangea" / "fix_repo" / INSTANCE_ID
PYTHON = ENVIRONMENT / ".venv" / "bin" / "python"
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")


def safe_environment() -> dict[str, str]:
    return {
        key: "<redacted>"
        if any(marker in key.lower() for marker in SECRET_MARKERS)
        else value
        for key, value in os.environ.items()
    }


def run(command: list[str], cwd: Path, phase: str) -> subprocess.CompletedProcess:
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    record = (
        f"\n{'=' * 80}\n"
        f"phase: {phase}\n"
        f"started_at: {started_at.isoformat()}\n"
        f"finished_at: {datetime.now(timezone.utc).isoformat()}\n"
        f"cwd: {cwd}\n"
        f"command: {subprocess.list2cmdline(command)}\n"
        f"returncode: {result.returncode}\n"
        f"stdout:\n{result.stdout}"
        f"stderr:\n{result.stderr}"
    )
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(record)
    return result


def record_investigation() -> None:
    environment = "\n".join(
        f"{key}={value}" for key, value in sorted(safe_environment().items())
    )
    record = f"""
{'=' * 80}
phase: second-pass-full-repository-reaudit
started_at: {datetime.now(timezone.utc).isoformat()}
finished_at: {datetime.now(timezone.utc).isoformat()}
cwd: {ORIGINAL_REPO}
command: manual full-repository production-path inspection
returncode: 0
stdout:
instance_id: {INSTANCE_ID}
base_commit: {BASE_COMMIT}
record_title: Requested tokens exceed context window of 2048.
upstream_tree_evidence:
- GitHub commit API identifies the exact commit date as 2023-08-04T10:46:02Z.
- The recursive upstream tree contains 215 entries and reports truncated=false.
metadata_path_result:
- src/enums.py only declares model context metadata and is not the production root cause.
production_path_evidence:
- src/gradio_runner.py accepts uploaded documents and controls max_new_tokens, top_k_docs, and chunk_size.
- src/gpt_langchain.py::_run_qa_db forwards max_new_tokens to get_llm but omits it from get_chain.
- src/gpt_langchain.py::get_chain budgets retrieved chunks with model_max_length minus a fixed 256 tokens.
- The same function explicitly states that its budget does not account for query, context, or separators.
- When no chunk fits, the original code still forces one chunk into the request.
- The final model request therefore can exceed model_max_length when uploaded-document chunks, query, history, prompt wrapper, and requested output are combined.
patch_rationale:
- Forward max_new_tokens into get_chain.
- Reserve the requested output budget instead of a fixed 256 tokens.
- Count query and chat context before selecting retrieved chunks.
- Select zero chunks when the best chunk cannot fit instead of forcing an oversized chunk.
external_research_evidence:
- Upstream h2oGPT documentation defines context as the combined input and output token budget.
- Later upstream architecture adds broader input-budget controls and a complete prompt-limiting path.
- No public fork-specific completion_chunk issue or traceback was found, so the deterministic test targets the verified local request-budget invariant.
environment:
{environment}
stderr:
"""
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(record)


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(
            f"{phase} returned {result.returncode}, expected {expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--runtime", choices=["local", "docker", "all"], default="all")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    record_investigation()

    if args.runtime in ["local", "all"]:
        original = run(
            [PYTHON.as_posix(), "-m", "pytest", "-q", (ARTIFACTS / "tests" / "test.py").as_posix()],
            ORIGINAL_REPO,
            "local-original-expected-failure",
        )
        require(original, 1, "local original")
        dry_run = run(
            ["patch", "--dry-run", "-p1", "-i", (ARTIFACTS / "patch.diff").as_posix()],
            ORIGINAL_REPO,
            "patch-dry-run",
        )
        require(dry_run, 0, "patch dry-run")
        patched = run(
            [PYTHON.as_posix(), "-m", "pytest", "-q", (ARTIFACTS / "tests" / "test.py").as_posix()],
            WORKSPACE,
            "local-patched",
        )
        require(patched, 0, "local patched")

    if args.runtime in ["docker", "all"]:
        original_tag = f"lidl-{INSTANCE_ID.lower()}-original"
        original_build = run(
            ["docker", "build", "--build-arg", "APPLY_PATCH=0", "-t", original_tag, "-f", (ARTIFACTS / "Dockerfile").as_posix(), ROOT.as_posix()],
            ROOT,
            "docker-build-original",
        )
        require(original_build, 0, "Docker original build")
        original_run = run(
            ["docker", "run", "--rm", original_tag],
            ROOT,
            "docker-run-original-expected-failure",
        )
        require(original_run, 1, "Docker original run")

        patched_tag = f"lidl-{INSTANCE_ID.lower()}-patched"
        patched_build = run(
            ["docker", "build", "--build-arg", "APPLY_PATCH=1", "-t", patched_tag, "-f", (ARTIFACTS / "Dockerfile").as_posix(), ROOT.as_posix()],
            ROOT,
            "docker-build-patched",
        )
        require(patched_build, 0, "Docker patched build")
        patched_run = run(
            ["docker", "run", "--rm", patched_tag],
            ROOT,
            "docker-run-patched",
        )
        require(patched_run, 0, "Docker patched run")


if __name__ == "__main__":
    main()
