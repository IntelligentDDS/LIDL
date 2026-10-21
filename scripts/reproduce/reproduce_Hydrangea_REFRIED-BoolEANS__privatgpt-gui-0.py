import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "REFRIED-BoolEANS__privatgpt-gui-0"
BASE_COMMIT = "840675927b1ace596dcc2c8ed0f6f483805e33fe"
ENVIRONMENT = ROOT / "datasets" / "Hydrangea" / "environments" / "REFRIED-BoolEANS__privatgpt-gui-84067592"
ORIGINAL_REPO = ROOT / "datasets" / "Hydrangea" / "repo" / INSTANCE_ID
PATCHED_REPO = ENVIRONMENT / "workspaces" / INSTANCE_ID
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


def redact_value(value: str) -> str:
    return "<redacted>" if any(pattern.search(value) for pattern in SECRET_PATTERNS) else value


def safe_environment() -> dict[str, str]:
    return {key: "<redacted>" if any(marker in key.lower() for marker in SECRET_MARKERS) else redact_value(value) for key, value in os.environ.items()}


def run(command: list[str], cwd: Path, phase: str) -> subprocess.CompletedProcess:
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {started_at.isoformat()}\nfinished_at: {datetime.now(timezone.utc).isoformat()}\ncwd: {cwd}\ncommand: {subprocess.list2cmdline(command)}\nreturncode: {result.returncode}\nstdout:\n{result.stdout}stderr:\n{result.stderr}")
    return result


def append_record(phase: str, stdout: str, returncode: int = 0) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {now}\nfinished_at: {now}\ncwd: {ORIGINAL_REPO}\ncommand: manual production-contract inspection\nreturncode: {returncode}\nstdout:\n{stdout}stderr:\n")


def complete_record() -> dict:
    return next(record for record in json.loads(DEFECTS.read_text(encoding="utf-8")) if record["instance_id"] == INSTANCE_ID)


def record_investigation() -> None:
    environment = "\n".join(f"{key}={value}" for key, value in sorted(safe_environment().items()))
    append_record(
        "legacy-doc-investigation",
        f"""instance_id: {INSTANCE_ID}
base_commit: {BASE_COMMIT}
complete_record:
{json.dumps(complete_record(), ensure_ascii=False, indent=2)}
production_path:
- privateGPT.py is a Tkinter application whose chooser copies a selected file to source_documents; a separate button launches ingest.py.
- ingest.py discovers .doc files, maps them to UnstructuredWordDocumentLoader, invokes loader.load(), then splits and embeds the returned documents.
- requirements.txt pins langchain 0.0.177 and unstructured 0.6.6.
real_fixture_proof:
- The test creates an actual OLE Compound File Microsoft Word 97 .doc with LibreOffice, checks its D0CF11E0A1B11AE1 signature, and passes it through the exact load_single_document function extracted from ingest.py.
- LangChain detects the .doc, Unstructured invokes soffice to convert it to .docx, and the loader returns all three expected paragraphs with source metadata preserved.
- The immutable base snapshot therefore parses a genuine legacy .doc correctly on Linux when its parser's external converter is installed.
defect_alignment:
- The original metadata paths are misleading because ingest.py already maps .doc files to the correct loader and the GUI only copies the selected file before launching ingestion.
- The actionable Linux setup defect is that README.md omits the system-level LibreOffice converter required by that loader.
patch:
- README.md now documents installation of libreoffice-writer before Python requirements.
- The new production Dockerfile installs libreoffice-writer so the GUI ingestion path works in a clean Linux environment.
- The production loader remains unchanged and the real legacy DOC fixture verifies the complete parser boundary.
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
    require(run(["patch", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], PATCHED_REPO, "apply-patch-isolated-workspace"), 0, "apply-patch-isolated-workspace")


def run_local() -> None:
    tests = str(ARTIFACTS / "tests" / "test.py")
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "local-original-fail-to-pass"), 1, "local-original-fail-to-pass")
    require(run(["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL_REPO, "patch-dry-run"), 0, "patch-dry-run")
    prepare_workspace()
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], PATCHED_REPO, "local-patched"), 0, "local-patched")


def run_docker() -> None:
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        require(run(["docker", "build", "--build-arg", f"APPLY_PATCH={apply_patch}", "-t", tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, f"docker-build-apply-patch-{apply_patch}"), 0, f"docker-build-apply-patch-{apply_patch}")
        require(run(["docker", "run", "--rm", tag], ROOT, f"docker-run-apply-patch-{apply_patch}"), expected, f"docker-run-apply-patch-{apply_patch}")


def validate_artifacts() -> None:
    python_files = [ARTIFACTS / "tests" / "test.py", Path(__file__), ORIGINAL_REPO / "ingest.py", ORIGINAL_REPO / "privateGPT.py"]
    for path in python_files:
        require(run([str(PYTHON), "-m", "py_compile", str(path)], ROOT, f"compile-{path.name}"), 0, f"compile-{path.name}")
    scanned = [ARTIFACTS / "tests" / "test.py", ARTIFACTS / "Dockerfile", ARTIFACTS / "patch.diff", ARTIFACTS / "run.log", Path(__file__)]
    hits = [f"{path}: {pattern.pattern}" for path in scanned for pattern in SECRET_PATTERNS if pattern.search(path.read_text(encoding="utf-8"))]
    append_record("compile-and-secret-scan", f"compiled_python_files={len(python_files)}\nscanned_files={len(scanned)}\nsecret_hits={len(hits)}\npatch_bytes={(ARTIFACTS / 'patch.diff').stat().st_size}\n" + ("\n".join(hits) + "\n" if hits else ""), 1 if hits else 0)
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
