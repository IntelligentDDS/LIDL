import argparse
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "StanGirard__quivr-0"
BASE_COMMIT = "b6f38f7aff5f026eb46a071362c9362e057119df"
ENVIRONMENT = ROOT / "datasets" / "Hydrangea" / "environments" / "StanGirard__quivr-b6f38f7a"
ORIGINAL_REPO = ROOT / "datasets" / "Hydrangea" / "repo" / INSTANCE_ID
PATCHED_REPO = ENVIRONMENT / "workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets" / "Hydrangea" / "fix_repo" / INSTANCE_ID
PYTHON = ENVIRONMENT / ".venv" / "bin" / "python"
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}\b"),
)


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


def append_record(phase: str, stdout: str, returncode: int = 0) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {now}\nfinished_at: {now}\n"
            f"cwd: {ORIGINAL_REPO}\ncommand: manual production-contract inspection\n"
            f"returncode: {returncode}\nstdout:\n{stdout}stderr:\n"
        )


def record_investigation() -> None:
    environment = "\n".join(
        f"{key}={value}" for key, value in sorted(safe_environment().items())
    )
    append_record(
        "knowledge-deletion-performance-investigation",
        f"""instance_id: {INSTANCE_ID}
base_commit: {BASE_COMMIT}
record_title: Knowledge fetching is very slow.
complete_record:
- The record says to create a brain, upload multiple files, ask questions, and check whether indexing takes too long.
- Its metadata names backend/models/databases/supabase/brains.py and backend/routes/knowledge_routes.py without supplying edits.
upstream_issue_evidence:
- The exact title comes from Quivr issue #1724, created 2023-11-27 after this 2023-09-27 base commit.
- The issue body identifies DELETE /knowledge/{{knowledge_id}}, links a Sentry performance transaction, and says to inspect a brain with multiple files.
- Upstream pull request #1733 and merge commit 10e94e5a913b76ba446859efd68f482f3fa70c8e are titled "improve delete knowledge performance" and modify the same metadata paths.
- The Hydrangea reproduction steps were therefore mislabeled as upload/indexing behavior, while the source issue and upstream patch identify deletion latency.
confirmed_root_cause:
- SupabaseDB.delete_file_from_brain first selects all vector IDs for a file.
- It then performs one brains_vectors delete, one brains_vectors select, and potentially one vectors delete for every chunk.
- A file with N chunks therefore incurs 1 + 2N to 3N database requests, making deletion latency scale linearly with chunk count and network round trips.
minimal_patch:
- Bulk-delete current-brain links with one PostgREST in_ filter.
- Bulk-select links that remain after unlinking and bulk-delete only vectors no longer used by any brain.
- Preserve shared vectors, the route contract, storage deletion, and the existing response message.
- Return immediately for an empty vector set so an empty PostgREST in-list is never sent.
complexity_change:
- Nonempty files now use four database requests regardless of chunk count: select vectors, unlink links, select remaining links, and delete unreferenced vectors.
environment:
{environment}
""",
    )


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(f"{phase} returned {result.returncode}, expected {expected}")


def run_local() -> None:
    tests = str(ARTIFACTS / "tests" / "test.py")
    require(
        run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "original-fail-to-pass"),
        1,
        "original-fail-to-pass",
    )
    require(
        run(
            ["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")],
            ORIGINAL_REPO,
            "immutable-source-patch-dry-run",
        ),
        0,
        "immutable-source-patch-dry-run",
    )
    require(
        run([str(PYTHON), "-m", "pytest", "-q", tests], PATCHED_REPO, "patched-pass-to-pass"),
        0,
        "patched-pass-to-pass",
    )


def run_docker() -> None:
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        require(
            run(
                [
                    "docker",
                    "build",
                    "--build-arg",
                    f"APPLY_PATCH={apply_patch}",
                    "-t",
                    tag,
                    "-f",
                    str(ARTIFACTS / "Dockerfile"),
                    str(ROOT),
                ],
                ROOT,
                f"docker-build-apply-patch-{apply_patch}",
            ),
            0,
            f"docker-build-apply-patch-{apply_patch}",
        )
        require(
            run(
                ["docker", "run", "--rm", tag],
                ROOT,
                f"docker-run-apply-patch-{apply_patch}",
            ),
            expected,
            f"docker-run-apply-patch-{apply_patch}",
        )


def validate_artifacts() -> None:
    python_files = [
        ARTIFACTS / "tests" / "test.py",
        Path(__file__),
        PATCHED_REPO / "backend" / "models" / "databases" / "supabase" / "brains.py",
    ]
    for path in python_files:
        require(
            run([str(PYTHON), "-m", "py_compile", str(path)], ROOT, f"compile-{path.name}"),
            0,
            f"compile-{path.name}",
        )
    scanned = [
        ARTIFACTS / "tests" / "test.py",
        ARTIFACTS / "Dockerfile",
        ARTIFACTS / "patch.diff",
        ARTIFACTS / "run.log",
        Path(__file__),
    ]
    hits = []
    for path in scanned:
        text = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(f"{path}: {pattern.pattern}")
    append_record(
        "compile-and-secret-scan",
        f"compiled_python_files={len(python_files)}\nscanned_files={len(scanned)}\n"
        f"secret_hits={len(hits)}\npatch_bytes={(ARTIFACTS / 'patch.diff').stat().st_size}\n"
        + ("\n".join(hits) + "\n" if hits else ""),
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
