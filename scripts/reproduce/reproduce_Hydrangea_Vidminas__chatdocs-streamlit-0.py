import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "Vidminas__chatdocs-streamlit-0"
BASE_COMMIT = "cf4053a1b20116fb0d05e64778e3c4c28b902f2a"
ENVIRONMENT = ROOT / "datasets/Hydrangea/environments/Vidminas__chatdocs-streamlit-cf4053a1"
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
        "large-chroma-parquet-production-contract-investigation",
        f"""instance_id: {INSTANCE_ID}
base_commit: {BASE_COMMIT}
complete_record:
{json.dumps(complete_record(), ensure_ascii=False, indent=2)}
defect_alignment:
- The large Parquet files are Chroma's internal chroma-collections.parquet and chroma-embeddings.parquet artifacts, not a user-upload extension; the reported upload steps are inaccurate while the large-database failure is real.
actual_production_path:
- embeddings_viz enables View DB by default, calls load_db_data, and the loader asks Chroma for every metadata, document, and high-dimensional embedding before constructing a pandas dataframe.
root_cause:
- Merely viewing ordinary database rows unnecessarily materializes the full embeddings Parquet payload in memory, so collection growth multiplies memory use and can fail-stop the Streamlit page.
patch:
- Exclude embeddings from the ordinary table loader by default and request them explicitly only when the user enables embedding visualization; no rows or pipeline data are sampled or truncated.
behavioral_verification:
- AST-isolated tests execute the exact loader with a fake Chroma collection, prove the default path does not fetch a 4096-dimensional embedding, and prove explicit visualization access still returns it.
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
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "original-large-parquet-fail-to-pass"), 1, "original-large-parquet-fail-to-pass")
    require(run(["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL_REPO, "immutable-source-patch-dry-run"), 0, "immutable-source-patch-dry-run")
    prepare_workspace()
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], PATCHED_REPO, "patched-workspace-large-parquet"), 0, "patched-workspace-large-parquet")


def run_docker() -> None:
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        require(run(["docker", "build", "--build-arg", f"APPLY_PATCH={apply_patch}", "-t", tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, f"docker-build-apply-patch-{apply_patch}"), 0, f"docker-build-apply-patch-{apply_patch}")
        require(run(["docker", "run", "--rm", tag], ROOT, f"docker-run-apply-patch-{apply_patch}"), expected, f"docker-run-apply-patch-{apply_patch}")


def validate_artifacts() -> None:
    python_files = [ARTIFACTS / "tests/test.py", Path(__file__), ORIGINAL_REPO / "chatdocs/st_utils.py", ORIGINAL_REPO / "chatdocs/pages/embeddings_viz.py"]
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
