import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "h2oai__h2ogpt-6"
BASE_COMMIT = "15f3a2d564541020f78cfc83bc90722dcda8a960"
ENVIRONMENT = ROOT / "datasets" / "Hydrangea" / "environments" / "h2oai__h2ogpt-15f3a2d5"
ORIGINAL_REPO = ROOT / "datasets" / "Hydrangea" / "repo" / INSTANCE_ID
WORKSPACE = ENVIRONMENT / "workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets" / "Hydrangea" / "fix_repo" / INSTANCE_ID
DEFECTS = ROOT / "datasets" / "Hydrangea" / "hydrangea_defect.json"
PYTHON = ENVIRONMENT / ".venv" / "bin" / "python"
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}\b"),
)


def redact(value):
    return "<redacted>" if any(pattern.search(value) for pattern in SECRET_PATTERNS) else value


def safe_environment():
    return {key: "<redacted>" if any(marker in key.lower() for marker in SECRET_MARKERS) else redact(value) for key, value in os.environ.items()}


def run(command, cwd, phase):
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {started_at.isoformat()}\nfinished_at: {datetime.now(timezone.utc).isoformat()}\ncwd: {cwd}\ncommand: {subprocess.list2cmdline(command)}\nreturncode: {result.returncode}\nstdout:\n{redact(result.stdout)}stderr:\n{redact(result.stderr)}")
    return result


def append_record(phase, text, returncode=0):
    now = datetime.now(timezone.utc).isoformat()
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {now}\nfinished_at: {now}\ncwd: {ORIGINAL_REPO}\ncommand: production-path inspection\nreturncode: {returncode}\nstdout:\n{text}stderr:\n")


def require(result, expected, phase):
    if result.returncode != expected:
        raise SystemExit(f"{phase} returned {result.returncode}, expected {expected}")


def prepare_workspace():
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(ORIGINAL_REPO, WORKSPACE, symlinks=True)
    require(run(["patch", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], WORKSPACE, "apply-patch-isolated-workspace"), 0, "apply-patch-isolated-workspace")


def record_investigation():
    record = next(item for item in json.loads(DEFECTS.read_text(encoding="utf-8")) if item["instance_id"] == INSTANCE_ID)
    environment = "\n".join(f"{key}={value}" for key, value in sorted(safe_environment().items()))
    append_record(
        "custom-database-investigation",
        f"instance_id: {INSTANCE_ID}\nbase_commit: {BASE_COMMIT}\ncomplete_record:\n{json.dumps(record, ensure_ascii=False, indent=2)}\n"
        "production_path:\n"
        "- src/gen.py reads LANGCHAIN_MODE and visible_langchain_modes before initializing every visible database.\n"
        "- The original fixed LangChainMode list rejects a selected custom mode and later filters custom visible modes out of the Gradio selector.\n"
        "- Existing db_dir_<custom mode> directories therefore cannot be selected through the documented runtime configuration.\n"
        "patch:\n"
        "- Merge explicitly selected and visible mode names into langchain_modes while preserving order and uniqueness.\n"
        "- Existing initialization and UI code then load and expose those configured databases without changing built-in defaults.\n"
        f"environment:\n{environment}\n",
    )


def run_local():
    tests = str(ARTIFACTS / "tests" / "test.py")
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "local-original-fail-to-pass"), 1, "local-original-fail-to-pass")
    require(run(["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL_REPO, "patch-dry-run"), 0, "patch-dry-run")
    prepare_workspace()
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], WORKSPACE, "local-patched"), 0, "local-patched")


def run_docker():
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        require(run(["docker", "build", "--build-arg", f"APPLY_PATCH={apply_patch}", "-t", tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, f"docker-build-apply-patch-{apply_patch}"), 0, f"docker-build-apply-patch-{apply_patch}")
        require(run(["docker", "run", "--rm", tag], ROOT, f"docker-run-apply-patch-{apply_patch}"), expected, f"docker-run-apply-patch-{apply_patch}")


def validate_artifacts():
    paths = [ARTIFACTS / "tests" / "test.py", ARTIFACTS / "Dockerfile", ARTIFACTS / "patch.diff", ARTIFACTS / "run.log", Path(__file__)]
    require(run([str(PYTHON), "-m", "py_compile", str(ARTIFACTS / "tests" / "test.py"), str(Path(__file__))], ROOT, "compile"), 0, "compile")
    hits = [f"{path}: {pattern.pattern}" for path in paths for pattern in SECRET_PATTERNS if pattern.search(path.read_text(encoding="utf-8"))]
    append_record("secret-scan", f"scanned_files={len(paths)}\nsecret_hits={len(hits)}\npatch_bytes={(ARTIFACTS / 'patch.diff').stat().st_size}\n" + ("\n".join(hits) if hits else ""), 1 if hits else 0)
    if hits:
        raise SystemExit("Secret scan found credentials")


def main():
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
