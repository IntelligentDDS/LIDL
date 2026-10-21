import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "catid__Supercharger-Auto-GPT-2"
ENVIRONMENT = (
    ROOT
    / "datasets"
    / "Hydrangea"
    / "environments"
    / "catid__Supercharger-Auto-GPT-2f8066fb"
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
        "- The defect record points to scripts/config.py, but that file only configures the OpenAI API key.\n"
        "- The production failure path is scripts/chat.py: chat_with_ai catches every HTTP 429 RateLimitError and retries after ten seconds.\n"
        "- openai-python 0.27.2 preserves the server JSON body on RateLimitError, including error.type and error.code.\n"
        "- An insufficient_quota response is permanent until account billing changes, while ordinary rate limiting remains retryable.\n"
        "- The test replaces only the OpenAI request and sleep boundaries and executes the production functions extracted from scripts/chat.py.\n"
        "references:\n"
        "- https://github.com/Significant-Gravitas/AutoGPT/issues/75\n"
        "- https://github.com/Significant-Gravitas/AutoGPT/issues/22\n"
        "- https://github.com/openai/openai-python/blob/v0.27.2/openai/api_requestor.py\n"
        "- https://github.com/openai/openai-python/blob/v0.27.2/openai/error.py\n"
        "- https://github.com/Significant-Gravitas/AutoGPT/issues/6593\n"
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
