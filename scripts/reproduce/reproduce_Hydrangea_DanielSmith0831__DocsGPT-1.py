import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "DanielSmith0831__DocsGPT-1"
BASE_COMMIT = "6b6737613ad934a28ae4251ffa9716465e6e35dd"
ENVIRONMENT = ROOT / "datasets" / "Hydrangea" / "environments" / "DanielSmith0831__DocsGPT-6b673761"
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
    environment = "\n".join(f"{key}={value}" for key, value in sorted(safe_environment().items()))
    now = datetime.now(timezone.utc).isoformat()
    record = f"""
{'=' * 80}
phase: production-token-path-repair
started_at: {now}
finished_at: {now}
cwd: {ORIGINAL_REPO}
command: manual full-source ingestion and request path inspection
returncode: 0
stdout:
instance_id: {INSTANCE_ID}
base_commit: {BASE_COMMIT}
record_title: Requested tokens exceed context window of 2048.
production_path:
- application/worker.py parses uploads and invokes application/parser/token_func.py with max_tokens=1250.
- application/app.py retrieves those chunks through conversational or map-reduce QA chains.
confirmed_defects:
- split_documents measures tokens but splits by character count and assumes every document begins with three newline-terminated header lines.
- A long plain-text upload raises AttributeError before it can be safely token-bounded.
- group_split catches every error and returns the original oversized document, silently defeating its context-limit contract.
- application/app.py also silently suppresses exceptions while formatting the production answer.
local_patch:
- Split directly on cl100k_base token boundaries, retain every token and metadata, and propagate ingestion failures instead of indexing oversized input.
- Remove the silent answer-format exception because splitting a string is deterministic after answer assignment.
environment:
{environment}
stderr:
"""
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(record)


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(f"{phase} returned {result.returncode}, expected {expected}")


def run_local() -> None:
    tests = str(ARTIFACTS / "tests" / "test.py")
    original = run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "original-fail-to-pass")
    require(original, 1, "original-fail-to-pass")
    dry_run = run(
        ["patch", "--dry-run", "-p1", "-i", str(ARTIFACTS / "patch.diff")],
        ORIGINAL_REPO,
        "immutable-source-patch-dry-run",
    )
    require(dry_run, 0, "immutable-source-patch-dry-run")
    patched = run([str(PYTHON), "-m", "pytest", "-q", tests], PATCHED_REPO, "patched-pass-to-pass")
    require(patched, 0, "patched-pass-to-pass")


def run_docker() -> None:
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        build = run(
            [
                "docker", "build", "--build-arg", f"APPLY_PATCH={apply_patch}",
                "-t", tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT),
            ],
            ROOT,
            f"docker-build-apply-patch-{apply_patch}",
        )
        require(build, 0, f"docker-build-apply-patch-{apply_patch}")
        result = run(["docker", "run", "--rm", tag], ROOT, f"docker-run-apply-patch-{apply_patch}")
        require(result, expected, f"docker-run-apply-patch-{apply_patch}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--runtime", choices=["local", "docker"], default="local")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    record_investigation()
    if args.runtime == "docker":
        run_docker()
    else:
        run_local()


if __name__ == "__main__":
    main()
