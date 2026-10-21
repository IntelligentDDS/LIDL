import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "edik7333__Auto-GPT-llama-cpp-8"
BASE_COMMIT = "2b4bf189825c0abe28bf6f020b87921d82636d67"
ENVIRONMENT = ROOT / "datasets" / "Hydrangea" / "environments" / "edik7333__Auto-GPT-llama-cpp-2b4bf189"
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
phase: gpt3-memory-production-path-repair
started_at: {now}
finished_at: {now}
cwd: {ORIGINAL_REPO}
command: manual defect, upstream issue, and production path inspection
returncode: 0
stdout:
instance_id: {INSTANCE_ID}
base_commit: {BASE_COMMIT}
record_title: Memory Feature seems to not work on gpt3only mode.
production_path:
- --gpt3only selects the fast model, while main.py still constructs the configured memory provider.
- main.py stores every assistant reply and command result in long-term memory and also appends the same result to recent message history.
- chat.py queries memory from the recent nine messages and includes both retrieved memory and recent history in the same GPT-3 request.
confirmed_root_cause:
- Recent command output, including a long read_file result, is duplicated as both active conversation and retrieved past memory.
- This consumes the smaller GPT-3 context and makes the model lose the task or emit invalid command JSON, matching the supplied reproduction and upstream AutoGPT issue 2232.
local_patch:
- Keep ordinary command results in both short-term history and long-term memory, but filter retrieved items containing active-history content before prompt construction.
- Preserve unrelated retrieved memories in provider order and prevent the explicit memory_add command from being written a second time by the interaction loop.
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
    parser.add_argument("--runtime", choices=["local", "docker", "all"], default="local")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "run.log").write_text("", encoding="utf-8")
    record_investigation()
    if args.runtime in ("local", "all"):
        run_local()
    if args.runtime in ("docker", "all"):
        run_docker()


if __name__ == "__main__":
    main()
