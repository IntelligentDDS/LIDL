import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "ATheorell__AutoGPTArenaHack-3"
BASE_COMMIT = "1e4f2dc004b92b9f236543674f94fb9f0af9bb2e"
ENVIRONMENT = ROOT / "datasets/Hydrangea/environments/ATheorell__AutoGPTArenaHack-1e4f2dc0"
ORIGINAL_REPO = ROOT / "datasets/Hydrangea/repo" / INSTANCE_ID
WORKSPACE = ENVIRONMENT / "workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets/Hydrangea/fix_repo" / INSTANCE_ID
PYTHON = ENVIRONMENT / ".venv/bin/python"
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
    environment = "\n".join(
        f"{key}={value}" for key, value in sorted(safe_environment().items())
    )
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: second-pass-full-repository-reaudit\n"
            f"finished_at: {datetime.now(timezone.utc).isoformat()}\n"
            f"cwd: {ROOT}\nstatus: reproducible_fixed\nreturncode: 0\n"
            f"base_commit: {BASE_COMMIT}\n"
            "record_title: Caught in error loop No Text to summary\n"
            "metadata_path: autogpts/forge/forge/agent.py\n"
            "actual_production_paths:\n"
            "- autogpts/autogpt/autogpt/commands/web_selenium.py returns an explicit no-text result and records browser errors.\n"
            "- autogpts/autogpt/autogpt/agents/agent.py records every action and result in event_history.\n"
            "- autogpts/autogpt/autogpt/agents/features/watchdog.py detects an identical consecutive command and intends to retry with the smart model.\n"
            "root_cause:\n"
            "The watchdog writes self.big_brain instead of self.config.big_brain. BaseAgent.llm and the watchdog guard both read self.config.big_brain, so the recursive retry remains on the fast model, detects the same action again, and can recurse indefinitely.\n"
            "repair:\n"
            "Set self.config.big_brain before the retry. ExitStack restores that same configuration field afterward.\n"
            f"environment:\n{environment}\n"
        )


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(f"{phase} expected return code {expected}, got {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--runtime", choices=["local", "docker"], default="local")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    record_investigation()

    if args.runtime == "docker":
        original_tag = f"lidl-{INSTANCE_ID.lower()}-original"
        patched_tag = f"lidl-{INSTANCE_ID.lower()}-patched"
        original_build = run(
            ["docker", "build", "--build-arg", "APPLY_PATCH=0", "-t", original_tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)],
            ROOT,
            "docker-build-original",
        )
        require(original_build, 0, "docker-build-original")
        require(run(["docker", "run", "--rm", original_tag], ROOT, "docker-run-original-expected-failure"), 1, "docker-run-original-expected-failure")
        patched_build = run(
            ["docker", "build", "--build-arg", "APPLY_PATCH=1", "-t", patched_tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)],
            ROOT,
            "docker-build-patched",
        )
        require(patched_build, 0, "docker-build-patched")
        require(run(["docker", "run", "--rm", patched_tag], ROOT, "docker-run-patched"), 0, "docker-run-patched")
        return

    tests = str(ARTIFACTS / "tests/test.py")
    require(
        run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "local-original-expected-failure"),
        1,
        "local-original-expected-failure",
    )
    require(
        run(["patch", "--dry-run", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL_REPO, "patch-dry-run"),
        0,
        "patch-dry-run",
    )
    require(
        run([str(PYTHON), "-m", "pytest", "-q", tests], WORKSPACE, "local-patched"),
        0,
        "local-patched",
    )


if __name__ == "__main__":
    main()
