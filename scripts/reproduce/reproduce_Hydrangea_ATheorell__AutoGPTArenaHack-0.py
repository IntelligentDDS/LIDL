import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "ATheorell__AutoGPTArenaHack-0"
BASE_COMMIT = "1e4f2dc004b92b9f236543674f94fb9f0af9bb2e"
ENVIRONMENT = ROOT / "datasets/Hydrangea/environments/ATheorell__AutoGPTArenaHack-1e4f2dc0"
ORIGINAL = ROOT / "datasets/Hydrangea/repo" / INSTANCE_ID
WORKSPACE = ENVIRONMENT / "workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets/Hydrangea/fix_repo" / INSTANCE_ID
DEFECTS = ROOT / "datasets/Hydrangea/hydrangea_defect.json"
PYTHON = ENVIRONMENT / ".venv/bin/python"
PROMPT_PATH = "autogpts/autogpt/autogpt/agents/prompt_strategies/one_shot.py"
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}\b"),
)


def write_record(phase: str, cwd: Path, command: list[str], result: subprocess.CompletedProcess) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {now}\nfinished_at: {now}\n"
            f"cwd: {cwd}\ncommand: {subprocess.list2cmdline(command)}\nreturncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}stderr:\n{result.stderr}"
        )


def run(command: list[str], cwd: Path, phase: str) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    write_record(phase, cwd, command, result)
    return result


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(f"{phase} returned {result.returncode}, expected {expected}")


def record_defect() -> None:
    records = json.loads(DEFECTS.read_text(encoding="utf-8"))
    record = next(item for item in records if item["instance_id"] == INSTANCE_ID)
    now = datetime.now(timezone.utc).isoformat()
    text = (
        f"instance_id: {INSTANCE_ID}\nbase_commit: {BASE_COMMIT}\n"
        f"complete_record:\n{json.dumps(record, ensure_ascii=False, indent=2)}\n"
        "root_cause: compile_progress discards every older action once detailed history reaches the prompt budget.\n"
        "patch: reserve part of the existing budget for compact earlier-action records and keep recent actions detailed.\n"
    )
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"{'=' * 80}\nphase: defect-evidence\nstarted_at: {now}\nfinished_at: {now}\n"
            f"cwd: {ORIGINAL}\ncommand: inspect defect and runtime data flow\nreturncode: 0\nstdout:\n{text}stderr:\n"
        )


def run_local() -> None:
    tests = str(ARTIFACTS / "tests/test.py")
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL, "original-fail-to-pass"), 1, "original-fail-to-pass")
    require(run(["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL, "patch-dry-run"), 0, "patch-dry-run")
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], WORKSPACE, "patched-fail-to-pass"), 0, "patched-fail-to-pass")


def run_docker() -> None:
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        require(run(["docker", "build", "--build-arg", f"APPLY_PATCH={apply_patch}", "-t", tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, f"docker-build-{apply_patch}"), 0, f"docker-build-{apply_patch}")
        require(run(["docker", "run", "--rm", tag], ROOT, f"docker-run-{apply_patch}"), expected, f"docker-run-{apply_patch}")


def validate() -> None:
    for path in (WORKSPACE / PROMPT_PATH, ARTIFACTS / "tests/test.py", Path(__file__)):
        require(run([str(PYTHON), "-m", "py_compile", str(path)], ROOT, f"compile-{path.name}"), 0, f"compile-{path.name}")
    scanned = [ARTIFACTS / "tests/test.py", ARTIFACTS / "Dockerfile", ARTIFACTS / "patch.diff", ARTIFACTS / "run.log", Path(__file__)]
    hits = [f"{path}: {pattern.pattern}" for path in scanned for pattern in SECRET_PATTERNS if pattern.search(path.read_text(encoding="utf-8"))]
    now = datetime.now(timezone.utc).isoformat()
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(f"\n{'=' * 80}\nphase: secret-scan\nstarted_at: {now}\nfinished_at: {now}\ncwd: {ROOT}\ncommand: scan artifacts\nreturncode: {int(bool(hits))}\nstdout:\nsecret_hits={len(hits)}\npatch_bytes={(ARTIFACTS / 'patch.diff').stat().st_size}\n" + "\n".join(hits) + "\nstderr:\n")
    if hits:
        raise SystemExit("Secret scan found credentials")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--runtime", choices=["local", "docker", "all"], default="local")
    args = parser.parse_args()
    (ARTIFACTS / "run.log").write_text("", encoding="utf-8")
    record_defect()
    if args.runtime in ("local", "all"):
        run_local()
    if args.runtime in ("docker", "all"):
        run_docker()
    validate()


if __name__ == "__main__":
    main()
