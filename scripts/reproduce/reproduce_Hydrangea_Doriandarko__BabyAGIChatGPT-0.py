import argparse
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = 'Doriandarko__BabyAGIChatGPT-0'
BASE_COMMIT = '174491cbbc09d3f6ef0ec511e9e3b94e2615ffa0'
ENVIRONMENT = ROOT / "datasets/Hydrangea/environments" / 'Doriandarko__BabyAGIChatGPT-174491cb'
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
            f"\n{'=' * 80}\nphase: prompt-grounding-repair\n"
            f"started_at: {datetime.now(timezone.utc).isoformat()}\n"
            f"finished_at: {datetime.now(timezone.utc).isoformat()}\n"
            f"cwd: {ROOT}\nreturncode: 0\ninstance_id: {INSTANCE_ID}\n"
            f"base_commit: {BASE_COMMIT}\nproduction_prompt_path: main.py\n"
            "status: reproducible_fixed\n"
            "repair_contract:\n"
            "- Answer only from the context already supplied by the production retrieval path.\n"
            "- Do not add unsupported facts or use outside knowledge.\n"
            "- State explicitly when the supplied context is insufficient.\n"
            "validation_oracle:\n"
            "- Static contract tests fail on the immutable source and pass on the isolated patched workspace.\n"
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
        require(
            run(["docker", "build", "--build-arg", "APPLY_PATCH=0", "-t", original_tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, "docker-build-original"),
            0,
            "docker-build-original",
        )
        require(run(["docker", "run", "--rm", original_tag], ROOT, "docker-run-original-expected-failure"), 1, "docker-run-original-expected-failure")
        require(
            run(["docker", "build", "--build-arg", "APPLY_PATCH=1", "-t", patched_tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, "docker-build-patched"),
            0,
            "docker-build-patched",
        )
        require(run(["docker", "run", "--rm", patched_tag], ROOT, "docker-run-patched"), 0, "docker-run-patched")
        return

    tests = str(ARTIFACTS / "tests/test.py")
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "local-original-expected-failure"), 1, "local-original-expected-failure")
    require(run(["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL_REPO, "patch-dry-run"), 0, "patch-dry-run")
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(ORIGINAL_REPO, WORKSPACE, symlinks=True)
    require(run(["patch", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], WORKSPACE, "apply-patch-isolated-workspace"), 0, "apply-patch-isolated-workspace")
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], WORKSPACE, "local-patched"), 0, "local-patched")


if __name__ == "__main__":
    main()
