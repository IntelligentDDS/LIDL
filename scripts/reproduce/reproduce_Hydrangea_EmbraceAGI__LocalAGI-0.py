import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "EmbraceAGI__LocalAGI-0"
ENVIRONMENT = ROOT / "datasets" / "Hydrangea" / "environments" / "EmbraceAGI__LocalAGI-f2fd7824"
ORIGINAL_REPO = ROOT / "datasets" / "Hydrangea" / "repo" / INSTANCE_ID
PATCHED_REPO = ENVIRONMENT / "workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets" / "Hydrangea" / "fix_repo" / INSTANCE_ID
PYTHON = ENVIRONMENT / ".venv" / "bin" / "python"
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")


def safe_environment() -> dict[str, str]:
    return {key: "<redacted>" if any(marker in key.lower() for marker in SECRET_MARKERS) else value for key, value in os.environ.items()}


def run(command: list[str], cwd: Path, phase: str) -> subprocess.CompletedProcess:
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    record = f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {started_at.isoformat()}\nfinished_at: {datetime.now(timezone.utc).isoformat()}\ncwd: {cwd}\ncommand: {subprocess.list2cmdline(command)}\nreturncode: {result.returncode}\nstdout:\n{result.stdout}stderr:\n{result.stderr}"
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(record)
    return result


def record_investigation() -> None:
    log_path = ARTIFACTS / "run.log"
    if log_path.exists():
        return
    environment = "\n".join(f"{key}={value}" for key, value in sorted(safe_environment().items()))
    log_path.write_text(
        f"instance_id: {INSTANCE_ID}\nstatus: reproducible\nbase_commit: f2fd7824ee270b8d45b336366ec885a14b809b8b\nrecord_title: The program cannot stop.\nrecord_problem_statement:\nThe program cannot stop.\n1. Set up LocalAGI according to the project README.\n2. Set LLM_MODEL=chatglm-6b, OBJECTIVE=Implement a Python code to accept input and print it in the terminal, and INITIAL_TASK=Build a task list.\n3. LocalAGI starts creating and completing tasks in an infinite loop and never stops.\nmetadata: null\nproduction_path_inspection:\n- The main loop can continuously consume tasks while task_creation_agent replenishes task_list.\n- The existing empty-queue branch still stops naturally, but there is no upper bound when the queue never empties.\n- The patch adds one environment-configurable total-task limit and increments it only after a completed task result.\n- The same boundary is applied to the English and Chinese entry points without launching models, Redis, or a vector store.\nenvironment:\n{environment}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--expect", choices=["fail", "pass"], required=True)
    parser.add_argument("--runtime", choices=["local", "docker"], default="local")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    record_investigation()
    expected = 1 if args.expect == "fail" else 0
    if args.runtime == "docker":
        tag = f"lidl-{INSTANCE_ID.lower()}-{args.expect}"
        command = ["docker", "build", "-t", tag]
        if args.expect == "fail":
            command.extend(["--build-arg", "APPLY_PATCH=0"])
        command.extend(["-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)])
        build = run(command, ROOT, f"docker-build-{args.expect}")
        if build.returncode != 0:
            raise SystemExit(f"Docker build failed with return code {build.returncode}")
        result = run(["docker", "run", "--rm", tag], ROOT, f"docker-run-{args.expect}")
    else:
        repo = ORIGINAL_REPO if args.expect == "fail" else PATCHED_REPO
        result = run([str(PYTHON), "-m", "pytest", "-q", str(ARTIFACTS / "tests" / "test.py")], repo, args.expect)
        if args.expect == "pass":
            dry_run = run(["patch", "--dry-run", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL_REPO, "patch-dry-run")
            if dry_run.returncode != 0:
                raise SystemExit(f"Patch dry-run failed with return code {dry_run.returncode}")
    if result.returncode != expected:
        raise SystemExit(f"Expected return code {expected}, got {result.returncode}")


if __name__ == "__main__":
    main()
