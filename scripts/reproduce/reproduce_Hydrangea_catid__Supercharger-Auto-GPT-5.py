import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "catid__Supercharger-Auto-GPT-5"
BASE_COMMIT = "2f8066fb4121ba0f724cbe92c4cbef91ecf5cc29"
ENVIRONMENT = ROOT / "datasets/Hydrangea/environments/catid__Supercharger-Auto-GPT-2f8066fb"
ORIGINAL_REPO = ROOT / "datasets/Hydrangea/repo" / INSTANCE_ID
PATCHED_REPO = Path("/tmp") / f"lidl-{INSTANCE_ID}-patched"
ARTIFACTS = ROOT / "datasets/Hydrangea/fix_repo" / INSTANCE_ID
DEFECTS = ROOT / "datasets/Hydrangea/hydrangea_defect.json"
PYTHON = ENVIRONMENT / ".venv/bin/python"
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}\b"),
)


def redact_value(value: str) -> str:
    return "<redacted>" if any(pattern.search(value) for pattern in SECRET_PATTERNS) else value


def safe_environment() -> dict[str, str]:
    return {
        key: "<redacted>" if any(marker in key.lower() for marker in SECRET_MARKERS) else redact_value(value)
        for key, value in os.environ.items()
    }


def run(command: list[str], cwd: Path, phase: str) -> subprocess.CompletedProcess:
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {started_at.isoformat()}\n"
            f"finished_at: {datetime.now(timezone.utc).isoformat()}\ncwd: {cwd}\n"
            f"command: {subprocess.list2cmdline(command)}\nreturncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}stderr:\n{result.stderr}"
        )
    return result


def append_record(phase: str, stdout: str, returncode: int = 0) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {now}\nfinished_at: {now}\n"
            f"cwd: {ORIGINAL_REPO}\ncommand: manual complete-record and production-path inspection\n"
            f"returncode: {returncode}\nstdout:\n{stdout}stderr:\n"
        )


def complete_record() -> dict:
    return next(
        record
        for record in json.loads(DEFECTS.read_text(encoding="utf-8"))
        if record["instance_id"] == INSTANCE_ID
    )


def record_investigation() -> None:
    environment = "\n".join(f"{key}={value}" for key, value in sorted(safe_environment().items()))
    append_record(
        "repeated-file-action-production-contract-investigation",
        f"""instance_id: {INSTANCE_ID}
base_commit: {BASE_COMMIT}
complete_record:
{json.dumps(complete_record(), ensure_ascii=False, indent=2)}
defect_alignment:
- The configured memory backend is not the write boundary; commands.execute_command dispatches every generated write action directly to file_operations.write_to_file.
actual_production_path:
- write_to_file opens the same workspace path with mode w on every action, so a second generated action discards the first action's text before any later append command can preserve it.
root_cause:
- The agent's advertised Write to file action has destructive replacement semantics even though iterative autonomous actions are expected to accumulate their generated report.
patch:
- Open the target with append mode while retaining directory creation, Unicode handling, safe path joining, command dispatch, and all generated text.
behavioral_verification:
- AST-isolated tests execute the exact production write_to_file function and prove two write actions preserve both ordered results while nested files and Unicode remain supported.
environment:
{environment}
""",
    )


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(f"{phase} returned {result.returncode}, expected {expected}")


def prepare_workspace() -> None:
    if PATCHED_REPO.exists():
        shutil.rmtree(PATCHED_REPO)
    shutil.copytree(ORIGINAL_REPO, PATCHED_REPO, symlinks=True)
    require(
        run(["patch", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], PATCHED_REPO, "apply-patch-isolated-workspace"),
        0,
        "apply-patch-isolated-workspace",
    )


def run_local() -> None:
    tests = str(ARTIFACTS / "tests/test.py")
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "original-repeated-write-fail-to-pass"), 1, "original-repeated-write-fail-to-pass")
    require(run(["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL_REPO, "immutable-source-patch-dry-run"), 0, "immutable-source-patch-dry-run")
    prepare_workspace()
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], PATCHED_REPO, "patched-workspace-repeated-write"), 0, "patched-workspace-repeated-write")


def run_docker() -> None:
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        require(run(["docker", "build", "--build-arg", f"APPLY_PATCH={apply_patch}", "-t", tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, f"docker-build-apply-patch-{apply_patch}"), 0, f"docker-build-apply-patch-{apply_patch}")
        require(run(["docker", "run", "--rm", tag], ROOT, f"docker-run-apply-patch-{apply_patch}"), expected, f"docker-run-apply-patch-{apply_patch}")


def validate_artifacts() -> None:
    python_files = [ARTIFACTS / "tests/test.py", Path(__file__), ORIGINAL_REPO / "scripts/file_operations.py", ORIGINAL_REPO / "scripts/commands.py"]
    for path in python_files:
        require(run([str(PYTHON), "-m", "py_compile", str(path)], ROOT, f"compile-{path.name}"), 0, f"compile-{path.name}")
    scanned = [ARTIFACTS / "tests/test.py", ARTIFACTS / "Dockerfile", ARTIFACTS / "patch.diff", ARTIFACTS / "run.log", Path(__file__)]
    hits = [f"{path}: {pattern.pattern}" for path in scanned for pattern in SECRET_PATTERNS if pattern.search(path.read_text(encoding="utf-8"))]
    append_record(
        "compile-and-secret-scan",
        f"compiled_python_files={len(python_files)}\nscanned_files={len(scanned)}\nsecret_hits={len(hits)}\npatch_bytes={(ARTIFACTS / 'patch.diff').stat().st_size}\n" + ("\n".join(hits) + "\n" if hits else ""),
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
