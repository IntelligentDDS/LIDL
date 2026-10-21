import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "edik7333__Auto-GPT-llama-cpp-4"
BASE_COMMIT = "2b4bf189825c0abe28bf6f020b87921d82636d67"
ENVIRONMENT = ROOT / "datasets/Hydrangea/environments/edik7333__Auto-GPT-llama-cpp-2b4bf189"
ORIGINAL_REPO = ROOT / "datasets/Hydrangea/repo" / INSTANCE_ID
PATCHED_REPO = ENVIRONMENT / "workspaces" / INSTANCE_ID
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


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(f"{phase} returned {result.returncode}, expected {expected}")


def record_investigation() -> None:
    environment = "\n".join(f"{key}={value}" for key, value in sorted(safe_environment().items()))
    now = datetime.now(timezone.utc).isoformat()
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(f"""
{'=' * 80}
phase: preseeded-memory-production-path-repair
started_at: {now}
finished_at: {now}
cwd: {ORIGINAL_REPO}
command: manual defect, upstream issue 2076, and production path inspection
returncode: 0
stdout:
instance_id: {INSTANCE_ID}
base_commit: {BASE_COMMIT}
record_title: Failing to use pre-seeded data and/or chunk a large JSON file.
upstream_issue_evidence:
- The reporter verified that Redis dump.rdb grew after data ingestion, but the agent ignored those entries and downloaded the repository instead.
- A maintainer response identified the missing capability: the agent exposed memory_add but no command for retrieving stored memory.
production_path:
- RedisMemory.get_relevant already queries the configured Redis index and returns stored data.
- chat.py performs implicit per-turn retrieval, but the agent cannot deliberately query the pre-seeded corpus for separate FAQ topics.
- prompt.py advertises long-term memory as a resource without exposing a retrieval command.
- config.py defaults WIPE_REDIS_ON_START to True despite the supplied environment template documenting False, risking deletion of pre-seeded entries when the variable is omitted.
local_patch:
- Expose a memory_get command that calls the active provider's get_relevant method with a query and requested result count.
- Advertise memory_get in the production command prompt.
- Align the Redis wipe default with the documented pre-seeded-data-safe False setting.
environment:
{environment}
stderr:
""")


def run_local() -> None:
    tests = str(ARTIFACTS / "tests/test.py")
    original = run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "original-fail-to-pass")
    require(original, 1, "original-fail-to-pass")
    dry_run = run(
        ["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")],
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
    parser.add_argument("--runtime", choices=["local", "docker", "all"], default="local")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "run.log").write_text("", encoding="utf-8")
    record_investigation()
    if args.runtime in ("local", "all"):
        run_local()
    if args.runtime in ("docker", "all"):
        run_docker()


if __name__ == "__main__":
    main()
