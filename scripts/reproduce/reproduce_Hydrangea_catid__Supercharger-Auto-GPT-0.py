import argparse
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "catid__Supercharger-Auto-GPT-0"
BASE_COMMIT = "2f8066fb4121ba0f724cbe92c4cbef91ecf5cc29"
ENVIRONMENT = ROOT / "datasets" / "Hydrangea" / "environments" / "catid__Supercharger-Auto-GPT-2f8066fb"
ORIGINAL_REPO = ROOT / "datasets" / "Hydrangea" / "repo" / INSTANCE_ID
PATCHED_REPO = ENVIRONMENT / "workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets" / "Hydrangea" / "fix_repo" / INSTANCE_ID
PYTHON = ENVIRONMENT / ".venv" / "bin" / "python"
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}\b"),
)


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


def append_record(phase: str, stdout: str, returncode: int = 0) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {now}\nfinished_at: {now}\n"
            f"cwd: {ORIGINAL_REPO}\ncommand: manual production-contract inspection\n"
            f"returncode: {returncode}\nstdout:\n{stdout}stderr:\n"
        )


def record_investigation() -> None:
    environment = "\n".join(f"{key}={value}" for key, value in sorted(safe_environment().items()))
    append_record(
        "memory-conversation-deduplication-investigation",
        f"""instance_id: {INSTANCE_ID}
base_commit: {BASE_COMMIT}
record_title: Relevant memory is essentially a duplicate of the conversation.
complete_snapshot:
- The exact commit has a 43-entry scripts-based application with process-local permanent memory rather than a vector provider.
production_path:
- commands.py memory_add appends an explicit string to memory.py::permanent_memory.
- main.py passes the same list and full_message_history to chat.chat_with_ai on every interaction.
- chat.py previously serialized every permanent-memory item and then independently inserted recent history into the same request.
confirmed_root_cause:
- When a memory item is still present verbatim in recent conversation history, the request sends identical content in both the Permanent memory system message and active history.
minimal_patch:
- Build a set of non-empty active-history content and exclude exact duplicates only from the per-request memory view.
- Preserve the process-local memory list, unrelated memory order, command behavior, recent history, and model selection.
environment:
{environment}
""",
    )


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(f"{phase} returned {result.returncode}, expected {expected}")


def run_local() -> None:
    tests = str(ARTIFACTS / "tests" / "test.py")
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "original-fail-to-pass"), 1, "original-fail-to-pass")
    require(
        run(
            ["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")],
            ORIGINAL_REPO,
            "immutable-source-patch-dry-run",
        ),
        0,
        "immutable-source-patch-dry-run",
    )
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], PATCHED_REPO, "patched-pass-to-pass"), 0, "patched-pass-to-pass")


def run_docker() -> None:
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        require(
            run(
                [
                    "docker",
                    "build",
                    "--build-arg",
                    f"APPLY_PATCH={apply_patch}",
                    "-t",
                    tag,
                    "-f",
                    str(ARTIFACTS / "Dockerfile"),
                    str(ROOT),
                ],
                ROOT,
                f"docker-build-apply-patch-{apply_patch}",
            ),
            0,
            f"docker-build-apply-patch-{apply_patch}",
        )
        require(run(["docker", "run", "--rm", tag], ROOT, f"docker-run-apply-patch-{apply_patch}"), expected, f"docker-run-apply-patch-{apply_patch}")


def validate_artifacts() -> None:
    python_files = [ARTIFACTS / "tests" / "test.py", Path(__file__), PATCHED_REPO / "scripts" / "chat.py"]
    for path in python_files:
        require(run([str(PYTHON), "-m", "py_compile", str(path)], ROOT, f"compile-{path.name}"), 0, f"compile-{path.name}")
    scanned = [ARTIFACTS / "tests" / "test.py", ARTIFACTS / "Dockerfile", ARTIFACTS / "patch.diff", ARTIFACTS / "run.log", Path(__file__)]
    hits = []
    for path in scanned:
        text = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(f"{path}: {pattern.pattern}")
    append_record(
        "compile-and-secret-scan",
        f"compiled_python_files={len(python_files)}\nscanned_files={len(scanned)}\nsecret_hits={len(hits)}\n"
        + ("\n".join(hits) + "\n" if hits else ""),
        1 if hits else 0,
    )
    if hits:
        raise SystemExit("Secret scan found credentials")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--runtime", choices=["local", "docker", "all"], default="local")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "run.log").write_text("", encoding="utf-8")
    record_investigation()
    if args.runtime in ("local", "all"):
        run_local()
    if args.runtime in ("docker", "all"):
        run_docker()
    validate_artifacts()


if __name__ == "__main__":
    main()
