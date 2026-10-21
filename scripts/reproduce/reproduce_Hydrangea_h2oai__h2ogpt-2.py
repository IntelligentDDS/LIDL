import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "h2oai__h2ogpt-2"
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
    return {
        key: "<redacted>" if any(marker in key.lower() for marker in SECRET_MARKERS) else redact(value)
        for key, value in os.environ.items()
    }


def run(command, cwd, phase):
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {started_at.isoformat()}\n"
            f"finished_at: {datetime.now(timezone.utc).isoformat()}\ncwd: {cwd}\n"
            f"command: {subprocess.list2cmdline(command)}\nreturncode: {result.returncode}\n"
            f"stdout:\n{redact(result.stdout)}stderr:\n{redact(result.stderr)}"
        )
    return result


def append_record(phase, text, returncode=0):
    now = datetime.now(timezone.utc).isoformat()
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {now}\nfinished_at: {now}\n"
            f"cwd: {ORIGINAL_REPO}\ncommand: production-path inspection\nreturncode: {returncode}\n"
            f"stdout:\n{text}stderr:\n"
        )


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
        "source-order-investigation",
        f"instance_id: {INSTANCE_ID}\nbase_commit: {BASE_COMMIT}\ncomplete_record:\n"
        f"{json.dumps(record, ensure_ascii=False, indent=2)}\n"
        "production_path:\n"
        "- src/gpt_langchain.py retrieves documents by semantic distance and reverses them when reverse_docs is enabled so the most relevant context is closest to the question.\n"
        "- get_sources_answer receives answer['input_documents'] in that actual LLM context order.\n"
        "- The original function deduplicates those sources and then sorts them by score again, so its displayed order differs from the order supplied to the LLM.\n"
        "patch:\n"
        "- Preserve insertion order while retaining the best score for duplicate source URLs.\n"
        "- Rename Ranked Sources to Sources because the list now represents context order rather than similarity rank.\n"
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
