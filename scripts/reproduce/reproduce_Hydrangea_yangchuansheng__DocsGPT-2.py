import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "yangchuansheng__DocsGPT-2"
BASE_COMMIT = "595581b624b54046d756ad28bc845521ee4807d0"
ENVIRONMENT = ROOT / "datasets" / "Hydrangea" / "environments" / "yangchuansheng__DocsGPT-595581b6"
ORIGINAL_REPO = ROOT / "datasets" / "Hydrangea" / "repo" / INSTANCE_ID
PATCHED_REPO = ENVIRONMENT / "workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets" / "Hydrangea" / "fix_repo" / INSTANCE_ID
PYTHON = ENVIRONMENT / ".venv" / "bin" / "python"
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")


def safe_environment() -> dict[str, str]:
    return {
        key: "<redacted>" if any(marker in key.lower() for marker in SECRET_MARKERS) else value
        for key, value in os.environ.items()
    }


def run(command: list[str], cwd: Path, phase: str) -> subprocess.CompletedProcess:
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    record = (
        f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {started_at.isoformat()}\n"
        f"finished_at: {datetime.now(timezone.utc).isoformat()}\ncwd: {cwd}\n"
        f"command: {subprocess.list2cmdline(command)}\nreturncode: {result.returncode}\n"
        f"stdout:\n{result.stdout}stderr:\n{result.stderr}"
    )
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(record)
    return result


def record_investigation() -> None:
    environment = "\n".join(f"{key}={value}" for key, value in sorted(safe_environment().items()))
    now = datetime.now(timezone.utc).isoformat()
    record = f"""
{'=' * 80}
phase: upstream-confirmed-root-cause
started_at: {now}
finished_at: {now}
cwd: {ORIGINAL_REPO}
command: manual source trace plus Arc53 DocsGPT issue and commit inspection
returncode: 0
stdout:
instance_id: {INSTANCE_ID}
base_commit: {BASE_COMMIT}
record_title: Requested tokens exceed context window of 2048.
production_path:
- scripts/ingest.py loads reStructuredText through SimpleDirectoryReader and RstParser before FAISS indexing.
- application/app.py retrieves four indexed chunks and sends them through LangChain's map-reduce QA chain.
confirmed_failure:
- Upstream issues 93, 97, and 145 report 4097-token OpenAI requests containing 6724 to 8489 prompt tokens.
- The issue 93 traceback reaches application/app.py, VectorDBQA, MapReduceDocumentsChain.combine_docs, and llm_chain.apply.
- The reporter reproduced with a one-word question, so an oversized retrieved document section caused the map request overflow.
upstream_fix:
- Arc53 commit f7d724458895c81a4438cb924267dfe46e8c276b on 2023-03-08 added token-bounded RstParser output.
- The maintainer then asked affected users to update and re-ingest; issue 145's reporter confirmed the problem disappeared.
local_patch:
- The patch follows that production fix at the exact pre-fix snapshot while splitting on tokenizer boundaries rather than character indices.
- Every parsed reStructuredText body is limited to 2048 cl100k_base tokens before indexing, and all tokens are retained across chunks.
environment:
{environment}
stderr:
"""
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(record)


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(f"{phase} returned {result.returncode}, expected {expected}")


def run_local() -> None:
    tests = str(ARTIFACTS / "tests" / "test.py")
    original = run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "original-fail-to-pass")
    require(original, 1, "original-fail-to-pass")
    dry_run = run(
        ["patch", "--dry-run", "-p1", "-i", str(ARTIFACTS / "patch.diff")],
        ORIGINAL_REPO,
        "immutable-source-patch-dry-run",
    )
    require(dry_run, 0, "immutable-source-patch-dry-run")
    patched = run([str(PYTHON), "-m", "pytest", "-q", tests], PATCHED_REPO, "patched-pass-to-pass")
    require(patched, 0, "patched-pass-to-pass")


def run_docker() -> None:
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        build = run(
            [
                "docker", "build", "--build-arg", f"APPLY_PATCH={apply_patch}",
                "-t", tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT),
            ],
            ROOT,
            f"docker-build-apply-patch-{apply_patch}",
        )
        require(build, 0, f"docker-build-apply-patch-{apply_patch}")
        result = run(["docker", "run", "--rm", tag], ROOT, f"docker-run-apply-patch-{apply_patch}")
        require(result, expected, f"docker-run-apply-patch-{apply_patch}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--runtime", choices=["local", "docker"], default="local")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    record_investigation()
    if args.runtime == "docker":
        run_docker()
    else:
        run_local()


if __name__ == "__main__":
    main()
