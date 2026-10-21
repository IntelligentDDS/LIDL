import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "edik7333__Auto-GPT-llama-cpp-0"
ENVIRONMENT = (
    ROOT
    / "datasets"
    / "Hydrangea"
    / "environments"
    / "edik7333__Auto-GPT-llama-cpp-2b4bf189"
)
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
    log_path = ARTIFACTS / "run.log"
    if log_path.exists():
        return
    environment = "\n".join(
        f"{key}={value}" for key, value in sorted(safe_environment().items())
    )
    log_path.write_text(
        f"instance_id: {INSTANCE_ID}\n"
        "status: reproducible\n"
        "investigation:\n"
        "- Agent.start_interaction_loop stores each assistant reply, command result, and human feedback in vector memory.\n"
        "- chat.chat_with_ai queries that memory using the last nine conversation messages and also inserts those same messages into the prompt.\n"
        "- Retrieved memory records can therefore repeat text already visible in recent conversation history.\n"
        "- The patch filters only retrieved records containing non-empty recent message content before prompt construction.\n"
        "- Distinct older memories and behavior with no recent history remain unchanged.\n"
        "- The test executes the production filter extracted from scripts/chat.py without importing optional model or vector-store dependencies.\n"
        f"environment:\n{environment}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--expect", choices=["fail", "pass"], required=True)
    parser.add_argument("--runtime", choices=["local", "docker"], default="local")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    record_investigation()

    expected_code = 1 if args.expect == "fail" else 0
    if args.runtime == "docker":
        tag = f"lidl-{INSTANCE_ID.lower()}-{args.expect}"
        command = ["docker", "build", "-t", tag]
        if args.expect == "fail":
            command.extend(["--build-arg", "APPLY_PATCH=0"])
        command.extend(["-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)])
        build = run(command, ROOT, f"docker-build-{args.expect}")
        if build.returncode != 0:
            raise SystemExit(f"Docker build failed with return code {build.returncode}")
        result = run(
            ["docker", "run", "--rm", tag],
            ROOT,
            f"docker-run-{args.expect}",
        )
    else:
        repo = ORIGINAL_REPO if args.expect == "fail" else PATCHED_REPO
        result = run(
            [
                str(PYTHON),
                "-m",
                "pytest",
                "-q",
                str(ARTIFACTS / "tests" / "test.py"),
            ],
            repo,
            args.expect,
        )
        if args.expect == "pass":
            dry_run = run(
                ["patch", "--dry-run", "-p1", "-i", str(ARTIFACTS / "patch.diff")],
                ORIGINAL_REPO,
                "patch-dry-run",
            )
            if dry_run.returncode != 0:
                raise SystemExit(
                    f"Patch dry-run failed with return code {dry_run.returncode}"
                )
    if result.returncode != expected_code:
        raise SystemExit(f"Expected return code {expected_code}, got {result.returncode}")


if __name__ == "__main__":
    main()
