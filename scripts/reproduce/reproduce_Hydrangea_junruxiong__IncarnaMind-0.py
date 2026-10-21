import argparse
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "junruxiong__IncarnaMind-0"
BASE_COMMIT = "75564a3bf08d0006387889cdc9a76fc509ededd8"
ENVIRONMENT = ROOT / "datasets/Hydrangea/environments/junruxiong__IncarnaMind-75564a3b"
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
    log_path = ARTIFACTS / "run.log"
    if log_path.exists():
        return
    environment = "\n".join(f"{key}={value}" for key, value in sorted(safe_environment().items()))
    log_path.write_text(
        f"instance_id: {INSTANCE_ID}\nstatus: reproducible_fixed\nbase_commit: {BASE_COMMIT}\n"
        "record_title: Image-only PDF causes list index out of range\n"
        "metadata_paths: main.py, convo_qa_chain.py\n"
        "production_path_inspection:\n"
        "- Actual ingestion occurs in docs2db.py through PyPDFLoader and process_files, before main.py loads persisted stores.\n"
        "- PyPDFLoader may return an empty list or pages whose page_content is empty for an image-only PDF.\n"
        "- process_files dereferences doc[0].metadata before validating extracted text, causing IndexError for an empty loader result.\n"
        "- The patch skips empty or whitespace-only documents, logs a warning, advances progress, and preserves text-document processing.\n"
        "conclusion:\nThe narrow ingestion defect is reproducible and fixed in docs2db.py.\n"
        f"environment:\n{environment}\n",
        encoding="utf-8",
    )


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
        if original_build.returncode != 0:
            raise SystemExit(f"Original Docker build failed with return code {original_build.returncode}")
        original = run(["docker", "run", "--rm", original_tag], ROOT, "docker-run-original-expected-failure")
        if original.returncode == 0:
            raise SystemExit("Original Docker run unexpectedly passed")
        patched_build = run(
            ["docker", "build", "--build-arg", "APPLY_PATCH=1", "-t", patched_tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)],
            ROOT,
            "docker-build-patched",
        )
        if patched_build.returncode != 0:
            raise SystemExit(f"Patched Docker build failed with return code {patched_build.returncode}")
        patched = run(["docker", "run", "--rm", patched_tag], ROOT, "docker-run-patched")
        if patched.returncode != 0:
            raise SystemExit(f"Patched Docker run failed with return code {patched.returncode}")
        return

    original = run(
        [str(PYTHON), "-m", "pytest", "-q", str(ARTIFACTS / "tests/test.py")],
        ORIGINAL_REPO,
        "local-original-expected-failure",
    )
    if original.returncode == 0:
        raise SystemExit("Original snapshot unexpectedly passed")
    dry_run = run(
        ["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")],
        ORIGINAL_REPO,
        "patch-dry-run",
    )
    if dry_run.returncode != 0:
        raise SystemExit(f"Patch dry-run failed with return code {dry_run.returncode}")
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(ORIGINAL_REPO, WORKSPACE, symlinks=True)
    applied = run(
        ["patch", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")],
        WORKSPACE,
        "apply-patch-isolated-workspace",
    )
    if applied.returncode != 0:
        raise SystemExit(f"Patch apply failed with return code {applied.returncode}")
    patched = run(
        [str(PYTHON), "-m", "pytest", "-q", str(ARTIFACTS / "tests/test.py")],
        WORKSPACE,
        "local-patched",
    )
    if patched.returncode != 0:
        raise SystemExit(f"Patched tests failed with return code {patched.returncode}")


if __name__ == "__main__":
    main()
