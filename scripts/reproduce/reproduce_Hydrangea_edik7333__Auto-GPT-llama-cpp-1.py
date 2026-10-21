import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "edik7333__Auto-GPT-llama-cpp-1"
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
        "- The reported HTML template contains byte 0xA0, which is invalid as a standalone UTF-8 byte and is a non-breaking space in Windows-1252.\n"
        "- scripts/file_operations.py opens files as UTF-8 text, so read_file catches the decoding exception and returns it as an error string.\n"
        "- The patch reads bytes, decodes valid UTF-8 first, and falls back to Windows-1252 only when UTF-8 decoding fails.\n"
        "- The behavioral tests execute the production safe_join and read_file functions without starting AutoGPT or contacting an external service.\n"
        "- Three defect cases cover Windows-1252 non-breaking space, accented text, and the euro sign; three regression cases preserve valid UTF-8 behavior.\n"
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
